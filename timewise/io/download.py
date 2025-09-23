import time
import threading
import logging
from queue import Empty
from typing import Dict
from itertools import product

import pandas as pd
import numpy as np
from astropy.table import Table
from pyvo.utils.http import create_session

from .stable_tap import StableTAPService
from .config import DownloadConfig
from ..types import TAPJobMeta, TaskID, TYPE_MAP
from ..query.base import Query
from ..util.error_threading import ErrorQueue, ExceptionSafeThread
from ..chunking import Chunker, Chunk


logger = logging.getLogger(__name__)


class Downloader:
    def __init__(self, cfg: DownloadConfig):
        self.cfg = cfg

        # Shared state
        self.job_lock = threading.Lock()
        # (chunk_id, query_hash) -> job meta
        self.jobs: Dict[TaskID, TAPJobMeta] = {}

        self.stop_event = threading.Event()
        self.submit_queue: ErrorQueue = ErrorQueue(stop_event=self.stop_event)
        self.submit_thread = ExceptionSafeThread(
            error_queue=self.submit_queue, target=self._submission_worker, daemon=True
        )
        self.poll_thread = ExceptionSafeThread(
            error_queue=self.submit_queue, target=self._polling_worker, daemon=True
        )
        self.all_chunks_queued = False
        self.all_chunks_submitted = False

        self.session = create_session()
        self.service: StableTAPService = StableTAPService(
            cfg.service_url, session=self.session
        )

        self.backend = cfg.backend
        self.chunker = Chunker(input_csv=cfg.input_csv, chunk_size=cfg.chunk_size)

    # ----------------------------
    # helpers
    # ----------------------------
    @staticmethod
    def get_task_id(chunk: Chunk, query: Query) -> TaskID:
        return TaskID(
            namespace="download", key=f"chunk{chunk.chunk_id:04d}_{query.hash}"
        )

    def iter_tasks(self) -> TaskID:
        for chunk in self.chunker:
            for q in self.cfg.queries:
                yield self.get_task_id(chunk, q)

    def load_job_meta(self):
        backend = self.backend
        for task in self.iter_tasks():
            if backend.meta_exists(task):
                logger.debug(f"found job metadata {task}")
                if task not in self.jobs:
                    try:
                        jm = TAPJobMeta(**backend.load_meta(task))
                        logger.debug(f"loaded {jm}")
                        logger.debug(f"setting {task}")
                        with self.job_lock:
                            self.jobs[task] = jm
                    except Exception:
                        continue

    # ----------------------------
    # TAP submission and download
    # ----------------------------
    def get_chunk_data(self, chunk: Chunk) -> pd.DataFrame:
        start = min(chunk.row_numbers) + 1  # plus one to always skip header line
        nrows = max(chunk.row_numbers) - start + 2  # plus one: skip header, plus one:

        columns = pd.read_csv(self.cfg.input_csv, nrows=0).columns
        return pd.read_csv(
            self.cfg.input_csv, skiprows=start, nrows=nrows, names=columns
        )

    def submit_tap_job(self, query: Query, chunk: Chunk) -> TAPJobMeta:
        adql = query.adql
        chunk_df = self.get_chunk_data(chunk)

        assert all(chunk_df.index.isin(chunk.indices)), (
            "Some inputs loaded from wrong chunk!"
        )
        assert all(np.isin(chunk.indices, chunk_df.index)), (
            f"Some indices are missing in chunk {chunk.chunk_id}!"
        )
        logger.debug(f"loaded {len(chunk_df)} objects")

        try:
            upload = Table(
                {
                    key: np.array(chunk_df[key]).astype(TYPE_MAP[dtype])
                    for key, dtype in query.input_columns.items()
                }
            )
        except KeyError as e:
            print(chunk_df)
            raise KeyError(e)

        logger.debug(f"uploading {len(upload)} objects.")
        job = self.service.submit_job(adql, uploads={query.upload_name: upload})
        job.run()
        logger.debug(job.url)

        return TAPJobMeta(
            url=job.url,
            query=adql,
            query_config=query.model_dump(),
            input_length=len(chunk_df),
            submitted=time.time(),
            last_checked=time.time(),
            status=job.phase,
            completed_at=0,
        )

    def check_job_status(self, job_meta: TAPJobMeta) -> str:
        return self.service.get_job_from_url(url=job_meta["url"]).phase

    def download_job_result(self, job_meta: TAPJobMeta) -> Table:
        logger.info(f"downloading {job_meta['url']}")
        job = self.service.get_job_from_url(url=job_meta["url"])
        job.wait()
        logger.info(f"{job_meta['url']}: Done!")
        return job.fetch_result().to_table()

    # ----------------------------
    # Submission thread
    # ----------------------------
    def _submission_worker(self):
        while not self.stop_event.is_set():
            try:
                chunk, query = self.submit_queue.get(timeout=1.0)  # type: Chunk, Query
            except Empty:
                if self.all_chunks_queued:
                    self.all_chunks_submitted = True
                    break
                continue

            # Wait until we have capacity
            while not self.stop_event.is_set():
                with self.job_lock:
                    running = sum(
                        1
                        for j in self.jobs.values()
                        if j.get("status") in ("QUEUED", "EXECUTING", "RUNNING")
                    )
                if running < self.cfg.max_concurrent_jobs:
                    break
                time.sleep(1.0)

            task = self.get_task_id(chunk, query)
            logger.info(f"submitting {task}")
            job_meta = self.submit_tap_job(query, chunk)
            self.backend.save_meta(task, job_meta)
            with self.job_lock:
                self.jobs[task] = job_meta

            self.submit_queue.task_done()

    # ----------------------------
    # Polling thread
    # ----------------------------
    def _polling_worker(self):
        logger.debug("starting polling worker")
        backend = self.backend
        while not self.stop_event.is_set():
            # reload job infos
            self.load_job_meta()

            with self.job_lock:
                items = list(self.jobs.items())

            for task, meta in items:  # type: TaskID, TAPJobMeta
                if meta.get("status") in ("COMPLETED", "ERROR", "ABORTED"):
                    logger.debug(f"{task} was already {meta['status']}")
                    continue

                status = self.check_job_status(meta)
                if status == "COMPLETED":
                    logger.info(f"completed {task}")
                    payload_table = self.download_job_result(meta)
                    logger.debug(payload_table.columns)
                    backend.save_data(task, payload_table)
                    meta["status"] = "COMPLETED"
                    meta["completed_at"] = time.time()
                    backend.save_meta(task, meta)
                    backend.mark_done(task)
                    with self.job_lock:
                        self.jobs[task] = meta
                elif status in ("ERROR", "ABORTED"):
                    logger.warning(f"failed {task}: {status}")
                    meta["status"] = status
                    with self.job_lock:
                        self.jobs[task] = meta
                    backend.save_meta(task, meta)
                else:
                    with self.job_lock:
                        self.jobs[task]["status"] = status
                        snapshot = self.jobs[task]
                    backend.save_meta(task, snapshot)

            if self.all_chunks_submitted:
                with self.job_lock:
                    all_done = (
                        all(
                            j.get("status") in ("COMPLETED", "ERROR", "ABORTED")
                            for j in self.jobs.values()
                        )
                        if len(self.jobs) > 0
                        else False
                    )
                if all_done:
                    logger.info("All tasks done! Exiting polling thread")
                    break

            time.sleep(self.cfg.poll_interval)

    # ----------------------------
    # Main run loop
    # ----------------------------
    def run(self):
        # load existing job metadata
        self.load_job_meta()

        # start threads
        self.submit_thread.start()
        self.poll_thread.start()

        # enqueue all chunks & queries
        backend = self.backend
        for chunk, q in product(self.chunker, self.cfg.queries):
            task = self.get_task_id(chunk, q)

            # skip if the download is done, or the job is queued
            if backend.is_done(task) or (task in self.jobs):
                continue

            self.submit_queue.put((chunk, q))
        self.all_chunks_queued = True
        # wait until all jobs are submitted
        self.submit_queue.join()
        # wait for the submit thread
        self.submit_thread.join()
        # the polling thread will exit ones all results are downloaded
        self.poll_thread.join()
        # the stop event will stop also the submit thread
        self.stop_event.set()
        # if any thread exited with an error report it
        self.submit_queue.raise_errors()
        logger.info("Done running downloader!")
