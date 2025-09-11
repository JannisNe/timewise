import os
import json
import time
import threading
import logging
from pathlib import Path
from queue import Empty
from typing import Dict, List

import pandas as pd
import numpy as np
from astropy.table import Table
from pydantic import BaseModel, Field, model_validator
from pyvo.utils.http import create_session
from pyvo.dal.tap import TAPService
from six import BytesIO

from .stable_tap import StableTAPService, StableAsyncTAPJob
from ..types import TAPJobMeta
from ..query import QueryConfig
from ..util.error_threading import ErrorQueue, ExceptionSafeThread
from ..util.csv_utils import get_n_rows


logger = logging.getLogger(__name__)


class DownloadConfig(BaseModel):
    input_csv: Path
    base_dir: Path
    chunk_size: int = 500_000
    raw_dir: str = "raw"
    max_concurrent_jobs: int = 4
    poll_interval: float = 10.0
    dry_run: bool = False
    queries: List[QueryConfig] = Field(..., description="One or more queries per chunk")

    service_url: str = "https://irsa.ipac.caltech.edu/TAP"

    @model_validator(mode="after")
    def validate_input_csv_columns(self) -> "DownloadConfig":
        """Ensure that the input CSV contains all columns required by queries."""
        # only validate if the CSV actually exists
        if not self.input_csv.exists():
            raise ValueError(f"CSV file does not exist: {self.input_csv}")

        # read just the header, avoid loading the entire file
        header = pd.read_csv(self.input_csv, nrows=0).columns

        missing_columns = set()
        for qc in self.queries:
            for col in qc.query.input_columns.keys():
                if col not in header:
                    missing_columns.add(col)

        if missing_columns:
            raise ValueError(
                f"CSV file {self.input_csv} is missing required columns: {sorted(missing_columns)}"
            )

        return self


class Downloader:
    def __init__(self, cfg: DownloadConfig):
        self.cfg = cfg
        self.raw_path = Path(self.cfg.base_dir) / self.cfg.raw_dir
        self.raw_path.mkdir(parents=True, exist_ok=True)

        # Shared state
        self.job_lock = threading.Lock()
        # (chunk_id, query_idx) -> job meta
        self.jobs: Dict[tuple[int, int], TAPJobMeta] = {}

        self.stop_event = threading.Event()
        self.submit_queue: ErrorQueue = ErrorQueue(stop_event=self.stop_event)
        self.submit_thread = ExceptionSafeThread(
            error_queue=self.submit_queue, target=self._submission_worker, daemon=True
        )
        self.poll_thread = ExceptionSafeThread(
            error_queue=self.submit_queue, target=self._polling_worker, daemon=True
        )
        self.all_chunks_queued = False

        self.session = create_session()
        self.service: TAPService = StableTAPService(
            cfg.service_url, session=self.session
        )

    # ----------------------------
    # Disk helpers (atomic writes)
    # ----------------------------
    @staticmethod
    def _atomic_write(target: Path, content: str, mode: str = "w"):
        tmp = target.with_suffix(target.suffix + ".tmp")
        logger.debug(f"writing {tmp}")
        with open(tmp, mode) as f:
            f.write(content)
        logger.debug(f"moving {tmp} to {target}")
        os.replace(tmp, target)

    def _job_path(self, chunk_id: int, query_idx: int) -> Path:
        return self.raw_path / f"chunk_{chunk_id:04d}_q{query_idx}.job.json"

    def _marker_path(self, chunk_id: int, query_idx: int) -> Path:
        return self.raw_path / f"chunk_{chunk_id:04d}_q{query_idx}.ok"

    def _chunk_path(self, chunk_id: int, query_idx: int) -> Path:
        return self.raw_path / f"chunk_{chunk_id:04d}_q{query_idx}.json"

    # ----------------------------
    # TAP submission and download
    # ----------------------------
    def submit_tap_job(self, query_config: QueryConfig, chunk_id: int) -> TAPJobMeta:
        adql = query_config.query.build()
        cs = self.cfg.chunk_size
        sr = list(range(1, chunk_id * cs + 1))
        chunk_df = pd.read_csv(self.cfg.input_csv, skiprows=sr, nrows=cs)
        if self.cfg.dry_run:
            return TAPJobMeta(
                url=f"dry-{int(time.time() * 1000)}",
                query=adql,
                input_length=len(chunk_df),
                submitted=time.time(),
                last_checked=time.time(),
                status="RUNNING",
                query_config=query_config.model_dump(),
                completed_at=0,
            )

        upload = Table(
            {
                key: np.array(chunk_df[key]).astype(dtype)
                for key, dtype in query_config.query.input_columns.items()
            }
        )

        logger.debug(f"uploading {len(upload)} objects.")
        job = self.service.submit_job(adql, uploads={"input": upload})
        job.run()
        logger.debug(job.url)

        return TAPJobMeta(
            url=job.url,
            query=adql,
            query_config=query_config.model_dump(),
            input_length=len(chunk_df),
            submitted=time.time(),
            last_checked=time.time(),
            status=job.phase,
            completed_at=0,
        )

    def check_job_status(self, job_meta: TAPJobMeta) -> str:
        if self.cfg.dry_run:
            return "COMPLETED"
        return StableAsyncTAPJob(url=job_meta["url"], session=self.session).phase

    def download_job_result(self, job_meta: TAPJobMeta) -> Table:
        if self.cfg.dry_run:
            return Table(
                {k: [2, 5, 1] for k in job_meta["query_config"]["input_columns"]}
            )
        logger.info(f"downloading {job_meta}")
        job = StableAsyncTAPJob(url=job_meta["url"], session=self.session)
        job.wait()
        logger.info(f"{job_meta}: Done!")
        return job.fetch_result().to_table()

    # ----------------------------
    # Submission thread
    # ----------------------------
    def _submission_worker(self):
        while not self.stop_event.is_set():
            try:
                chunk_id, query_idx, query_config = self.submit_queue.get(timeout=1.0)  # type: int, int, QueryConfig
            except Empty:
                if self.all_chunks_queued:
                    break
                continue

            # Wait until we have capacity
            while not self.stop_event.is_set():
                with self.job_lock:
                    running = sum(
                        1
                        for j in self.jobs.values()
                        if j.get("status") in ("QUEUED", "EXECUTING", "RUN")
                    )
                if running < self.cfg.max_concurrent_jobs:
                    break
                time.sleep(1.0)

            logger.info(f"submitting chunk {chunk_id}, query {query_idx}")
            job_meta = self.submit_tap_job(query_config, chunk_id)

            job_path = self._job_path(chunk_id, query_idx)
            self._atomic_write(job_path, json.dumps(job_meta, indent=2))

            with self.job_lock:
                self.jobs[(chunk_id, query_idx)] = job_meta

            self.submit_queue.task_done()

    # ----------------------------
    # Polling thread
    # ----------------------------
    def _polling_worker(self):
        logger.debug("starting polling worker")
        while not self.stop_event.is_set():
            # reload job files from disk
            for job_file in sorted(self.raw_path.glob("chunk_*_q*.job.json")):
                logger.debug(f"found job file {job_file}")
                parts = job_file.stem.split("_")
                chunk_id = int(parts[1])
                query_idx = int(parts[2][1:])
                key = (chunk_id, query_idx)
                if key not in self.jobs:
                    try:
                        with open(job_file) as f:
                            jm = TAPJobMeta(**json.load(f))
                        logger.debug(f"loaded {jm}")
                        logger.debug(f"setting {key}")
                        with self.job_lock:
                            self.jobs[key] = jm
                    except Exception:
                        continue

            with self.job_lock:
                items = list(self.jobs.items())

            for (chunk_id, query_idx), meta in items:  # type: tuple[int, int], TAPJobMeta
                if meta.get("status") in ("COMPLETED", "ERROR", "ABORTED"):
                    logger.debug(f"{meta} was already {meta['status']}")
                    continue

                status = self.check_job_status(meta)
                if status == "COMPLETED":
                    logger.info(f"completed {chunk_id}, query {query_idx}")
                    payload_table = self.download_job_result(meta)
                    with BytesIO() as io:
                        payload = payload_table.write(io, format="fits")
                        payload.seek(0)
                        self._atomic_write(
                            self._chunk_path(chunk_id, query_idx), payload, mode="wb"
                        )
                    self._atomic_write(self._marker_path(chunk_id, query_idx), "done")
                    meta["status"] = "COMPLETED"
                    meta["completed_at"] = time.time()
                    self._atomic_write(
                        self._job_path(chunk_id, query_idx), json.dumps(meta, indent=2)
                    )
                    with self.job_lock:
                        self.jobs[(chunk_id, query_idx)] = meta
                elif status in ("ERROR", "ABORTED"):
                    logger.warning(f"failed {chunk_id}, query {query_idx}: {status}")
                    meta["status"] = status
                    with self.job_lock:
                        self.jobs[(chunk_id, query_idx)] = meta
                    self._atomic_write(
                        self._job_path(chunk_id, query_idx), json.dumps(meta, indent=2)
                    )
                else:
                    with self.job_lock:
                        self.jobs[(chunk_id, query_idx)]["status"] = status
                        snapshot = self.jobs[(chunk_id, query_idx)]
                    self._atomic_write(
                        self._job_path(chunk_id, query_idx),
                        json.dumps(snapshot, indent=2),
                    )

            if self.all_chunks_queued:
                with self.job_lock:
                    all_done = all(
                        j.get("status") in ("COMPLETED", "ERROR", "ABORTED")
                        for j in self.jobs.values()
                    )
                if all_done:
                    logger.info("All tasks done! Exiting polling thread")
                    break

            time.sleep(self.cfg.poll_interval)

    # ----------------------------
    # Main run loop
    # ----------------------------
    def run(self):
        chunk_size = self.cfg.chunk_size
        n = get_n_rows(self.cfg.input_csv) - 1  # one header row

        # load existing job metadata
        for job_file in sorted(self.raw_path.glob("chunk_*_q*.job.json")):
            try:
                parts = job_file.stem.split("_")
                chunk_id = int(parts[1])
                query_idx = int(parts[2][1:])
                with open(job_file) as f:
                    jm = json.load(f)
                self.jobs[(chunk_id, query_idx)] = jm
            except Exception:
                continue

        # start threads
        self.submit_thread.start()
        self.poll_thread.start()

        # enqueue all chunks & queries
        for i in range(0, n, chunk_size):
            chunk_id = i // chunk_size

            for query_idx, qcfg in enumerate(self.cfg.queries):
                marker = self._marker_path(chunk_id, query_idx)
                if marker.exists():
                    continue
                job_file = self._job_path(chunk_id, query_idx)
                if job_file.exists():
                    try:
                        with open(job_file) as f:
                            jm = json.load(f)
                        with self.job_lock:
                            self.jobs[(chunk_id, query_idx)] = jm
                        continue
                    except Exception:
                        pass

                self.submit_queue.put((chunk_id, query_idx, qcfg))

        self.all_chunks_queued = True
        self.submit_queue.join()
        self.stop_event.set()
        self.submit_thread.join()
        self.poll_thread.join()
