# timewise/io/tap_client.py
"""
Downloader with threaded TAP submission + polling.

Behavior implemented:
- Read parent CSV and split into chunks.
- A **submission** thread submits TAP queries for chunks and writes a per-chunk job file
  (`chunk_####.job.json`) containing `job_id` and `job_url`.
- A **polling** thread loops over all known job files, checks job status, and when a job
  is finished downloads the result into `chunk_####.json` and writes the `.ok` marker.
- On startup existing `.job.json` files are reloaded to resume monitoring previously
  submitted jobs.

The TAP-specific functions (`submit_tap_job`, `check_job_status`, `download_job_result`)
are placeholders and should be implemented with `pyvo` calls against IRSA.
"""

import os
import json
import time
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, Any

import pandas as pd
from pydantic import BaseModel, Field


class DownloadConfig(BaseModel):
    input_csv: Path
    base_dir: Path
    chunk_size: int = Field(default=500, ge=1)
    raw_dir: str = "raw"
    max_concurrent_jobs: int = 4
    poll_interval: float = 10.0  # seconds
    dry_run: bool = False


class Downloader:
    def __init__(self, cfg: DownloadConfig):
        self.cfg = cfg
        self.raw_path = Path(self.cfg.base_dir) / self.cfg.raw_dir
        self.raw_path.mkdir(parents=True, exist_ok=True)

        # Shared state
        self.job_lock = threading.Lock()
        self.jobs: Dict[int, Dict[str, Any]] = {}  # chunk_id -> job metadata

        # Queues for submission
        self.submit_queue: Queue = Queue()
        self.stop_event = threading.Event()

    # ----------------------------
    # Disk helpers (atomic writes)
    # ----------------------------
    def _atomic_write(self, target: Path, content: str):
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "w") as f:
            f.write(content)
        os.replace(tmp, target)  # atomic rename

    def _marker_path(self, chunk_id: int) -> Path:
        return self.raw_path / f"chunk_{chunk_id:04d}.ok"

    def _chunk_path(self, chunk_id: int) -> Path:
        return self.raw_path / f"chunk_{chunk_id:04d}.json"

    def _job_path(self, chunk_id: int) -> Path:
        return self.raw_path / f"chunk_{chunk_id:04d}.job.json"

    # ----------------------------
    # TAP placeholders (replace with pyvo)
    # ----------------------------
    def submit_tap_job(self, chunk_df: pd.DataFrame) -> Dict[str, str]:
        """Submit a TAP job for the chunk and return a dict with job_id and job_url.

        Replace this with code using pyvo.TAPService(...).submit_job(...) or the async API.
        For now we simulate by returning a fake job id and URL.
        """
        if self.cfg.dry_run:
            return {"job_id": f"dry-{int(time.time()*1000)}", "job_url": "dry://job", "n_rows": len(chunk_df)}

        # Real implementation note (pyvo):
        # svc = TAPService(SERVICE_URL)
        # job = svc.submit_job(adql, uploads=upload_table)
        # job_id = job.jobid
        # job_url = job.href
        # return {"job_id": job_id, "job_url": job_url}

        return {"job_id": f"sim-{int(time.time()*1000)}", "job_url": "http://example.com/job", "n_rows": len(chunk_df)}

    def check_job_status(self, job_meta: Dict[str, Any]) -> str:
        """Return one of: 'PENDING', 'RUNNING', 'COMPLETED', 'ERROR'.

        Replace with real status checking via pyvo or HTTP GET on job_url.
        """
        if self.cfg.dry_run:
            # pretend it completes quickly
            return "COMPLETED"
        # Placeholder behavior: jobs take ~20s to complete
        submitted = job_meta.get("submitted_at", time.time())
        if time.time() - submitted > 20:
            return "COMPLETED"
        return "RUNNING"

    def download_job_result(self, job_meta: Dict[str, Any]) -> str:
        """Download the job result and return the serialized content to write to disk.

        Replace with code to fetch the result table (VOTable/CSV) and serialize to JSON/CSV.
        """
        if self.cfg.dry_run:
            return json.dumps({"rows": job_meta.get("n_rows", 0), "status": "dry-complete"}, indent=2)
        # Simulate a small payload
        return json.dumps({"rows": job_meta.get("n_rows", 0), "status": "sim-complete"}, indent=2)

    # ----------------------------
    # Submission thread
    # ----------------------------
    def _submission_worker(self):
        """Worker that consumes the submit_queue and submits TAP jobs while keeping
        no more than max_concurrent_jobs running.
        """
        print("[downloader] Submission thread started")
        while not self.stop_event.is_set():
            try:
                chunk_id, chunk_df = self.submit_queue.get(timeout=1.0)
            except Empty:
                # queue empty: break if no more expected
                if getattr(self, "all_chunks_queued", False):
                    break
                continue

            # Wait until we have capacity
            while not self.stop_event.is_set():
                with self.job_lock:
                    running = sum(1 for j in self.jobs.values() if j.get("status") in ("PENDING", "RUNNING"))
                if running < self.cfg.max_concurrent_jobs:
                    break
                time.sleep(1.0)

            # Submit job
            print(f"[downloader] Submitting chunk {chunk_id}")
            job_info = self.submit_tap_job(chunk_df)
            job_meta = {
                "job_id": job_info["job_id"],
                "job_url": job_info.get("job_url"),
                "status": "PENDING",
                "submitted_at": time.time(),
                "chunk_size": len(chunk_df),
            }

            # persist job meta to disk
            job_path = self._job_path(chunk_id)
            self._atomic_write(job_path, json.dumps(job_meta, indent=2))

            with self.job_lock:
                self.jobs[chunk_id] = job_meta

            self.submit_queue.task_done()

        print("[downloader] Submission thread exiting")

    # ----------------------------
    # Polling thread
    # ----------------------------
    def _polling_worker(self):
        """Poll known job files and download completed results."""
        print("[downloader] Polling thread started")
        while not self.stop_event.is_set():
            # reload job files from disk to pick up jobs submitted in previous runs
            for job_file in sorted(self.raw_path.glob("chunk_*.job.json")):
                chunk_id = int(job_file.stem.split("_")[1])
                if chunk_id not in self.jobs:
                    try:
                        with open(job_file, "r") as f:
                            jm = json.load(f)
                        with self.job_lock:
                            self.jobs[chunk_id] = jm
                    except Exception:
                        continue

            # Iterate over known jobs and check status
            to_download = []
            with self.job_lock:
                jobs_items = list(self.jobs.items())

            for chunk_id, meta in jobs_items:
                if meta.get("status") in ("COMPLETED", "ERROR"):
                    continue

                status = self.check_job_status(meta)
                if status == "COMPLETED":
                    print(f"[downloader] Job for chunk {chunk_id} completed; downloading result")
                    # download result
                    payload = self.download_job_result(meta)
                    # write chunk payload atomically
                    self._atomic_write(self._chunk_path(chunk_id), payload)
                    # write marker
                    self._atomic_write(self._marker_path(chunk_id), "done")
                    # update job meta
                    meta["status"] = "COMPLETED"
                    meta["completed_at"] = time.time()
                    # persist updated job meta
                    self._atomic_write(self._job_path(chunk_id), json.dumps(meta, indent=2))
                    with self.job_lock:
                        self.jobs[chunk_id] = meta
                elif status == "ERROR":
                    print(f"[downloader] Job for chunk {chunk_id} errored")
                    meta["status"] = "ERROR"
                    with self.job_lock:
                        self.jobs[chunk_id] = meta
                    self._atomic_write(self._job_path(chunk_id), json.dumps(meta, indent=2))
                else:
                    # still running / pending
                    with self.job_lock:
                        self.jobs[chunk_id]["status"] = status
                    # persist to disk occasionally
                    self._atomic_write(self._job_path(chunk_id), json.dumps(self.jobs[chunk_id], indent=2))

            # If all jobs are completed and all chunks were queued, we can exit
            if getattr(self, "all_chunks_queued", False):
                with self.job_lock:
                    all_done = all(j.get("status") in ("COMPLETED", "ERROR") for j in self.jobs.values())
                if all_done:
                    break

            time.sleep(self.cfg.poll_interval)

        print("[downloader] Polling thread exiting")

    # ----------------------------
    # Main run loop
    # ----------------------------
    def run(self):
        df = pd.read_csv(self.cfg.input_csv)
        n = len(df)
        chunk_size = self.cfg.chunk_size

        # load existing job metadata to resume
        for job_file in sorted(self.raw_path.glob("chunk_*.job.json")):
            try:
                with open(job_file, "r") as f:
                    jm = json.load(f)
                chunk_id = int(job_file.stem.split("_")[1])
                self.jobs[chunk_id] = jm
            except Exception:
                continue

        # Start threads
        self.submit_thread = threading.Thread(target=self._submission_worker, daemon=True)
        self.poll_thread = threading.Thread(target=self._polling_worker, daemon=True)
        self.submit_thread.start()
        self.poll_thread.start()

        # Enqueue chunks for submission (but don't block — submission thread governs concurrency)
        for i in range(0, n, chunk_size):
            chunk_id = i // chunk_size
            marker = self._marker_path(chunk_id)
            if marker.exists():
                print(f"[download] Skipping chunk {chunk_id}, already downloaded")
                continue

            # If there is an existing .job.json for this chunk and its status is not ERROR/COMPLETED,
            # it will be picked up by the polling thread; but we still ensure job record exists.
            job_file = self._job_path(chunk_id)
            if job_file.exists():
                try:
                    with open(job_file, "r") as f:
                        jm = json.load(f)
                    with self.job_lock:
                        self.jobs[chunk_id] = jm
                    continue
                except Exception:
                    pass

            chunk_df = df.iloc[i : i + chunk_size]
            # put on submission queue
            self.submit_queue.put((chunk_id, chunk_df))

        # Signal that all chunks have been queued
        self.all_chunks_queued = True

        # Wait for submission queue to be empty and threads to finish
        self.submit_queue.join()
        # Wait for threads to observe completion
        self.stop_event.set()
        self.submit_thread.join()
        self.poll_thread.join()

        print("[downloader] All done")


# Entrypoint for CLI

def download_all_chunks(cfg_dict: dict):
    # Build DownloadConfig from general cfg dict structure
    download_cfg = cfg_dict.get("download", {})
    storage_cfg = cfg_dict.get("storage", {})
    cfg = DownloadConfig(
        input_csv=cfg_dict["input_csv"],
        base_dir=storage_cfg.get("base_dir"),
        chunk_size=download_cfg.get("chunk_size", 500),
        max_concurrent_jobs=download_cfg.get("max_concurrent_jobs", 4),
        poll_interval=download_cfg.get("poll_interval", 10.0),
        dry_run=download_cfg.get("dry_run", False),
    )
    downloader = Downloader(cfg)
    downloader.run()
