import os
import json
import time
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, Any, List

import pandas as pd
from pydantic import BaseModel, Field

from timewise.query import QueryConfig


class DownloadConfig(BaseModel):
    input_csv: Path
    base_dir: Path
    chunk_size: int = 500
    raw_dir: str = "raw"
    max_concurrent_jobs: int = 4
    poll_interval: float = 10.0
    dry_run: bool = False
    queries: List[QueryConfig] = Field(..., description="One or more queries per chunk")


class Downloader:
    def __init__(self, cfg: DownloadConfig):
        self.cfg = cfg
        self.raw_path = Path(self.cfg.base_dir) / self.cfg.raw_dir
        self.raw_path.mkdir(parents=True, exist_ok=True)

        # Shared state
        self.job_lock = threading.Lock()
        # (chunk_id, query_idx) -> job meta
        self.jobs: Dict[Any, Dict[str, Any]] = {}

        self.submit_queue: Queue = Queue()
        self.stop_event = threading.Event()

    # ----------------------------
    # Disk helpers (atomic writes)
    # ----------------------------
    def _atomic_write(self, target: Path, content: str):
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "w") as f:
            f.write(content)
        os.replace(tmp, target)

    def _job_path(self, chunk_id: int, query_idx: int) -> Path:
        return self.raw_path / f"chunk_{chunk_id:04d}_q{query_idx}.job.json"

    def _marker_path(self, chunk_id: int, query_idx: int) -> Path:
        return self.raw_path / f"chunk_{chunk_id:04d}_q{query_idx}.ok"

    def _chunk_path(self, chunk_id: int, query_idx: int) -> Path:
        return self.raw_path / f"chunk_{chunk_id:04d}_q{query_idx}.json"

    # ----------------------------
    # TAP placeholders
    # ----------------------------
    def submit_tap_job(self, query_config: QueryConfig, chunk_df: pd.DataFrame) -> Dict[str, Any]:
        adql = query_config.query.build()
        if self.cfg.dry_run:
            return {"job_id": f"dry-{int(time.time()*1000)}", "job_url": "dry://job", "query": adql, "n_rows": len(chunk_df)}
        # TODO: real pyvo TAP submission
        return {"job_id": f"sim-{int(time.time()*1000)}", "job_url": "http://example.com/job", "query": adql, "n_rows": len(chunk_df)}

    def check_job_status(self, job_meta: Dict[str, Any]) -> str:
        if self.cfg.dry_run:
            return "COMPLETED"
        submitted = job_meta.get("submitted_at", time.time())
        if time.time() - submitted > 20:
            return "COMPLETED"
        return "RUNNING"

    def download_job_result(self, job_meta: Dict[str, Any]) -> str:
        if self.cfg.dry_run:
            return json.dumps({"rows": job_meta.get("n_rows", 0), "status": "dry-complete"}, indent=2)
        return json.dumps({"rows": job_meta.get("n_rows", 0), "status": "sim-complete"}, indent=2)

    # ----------------------------
    # Submission thread
    # ----------------------------
    def _submission_worker(self):
        while not self.stop_event.is_set():
            try:
                chunk_id, query_idx, query_config, chunk_df = self.submit_queue.get(timeout=1.0)
            except Empty:
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

            job_info = self.submit_tap_job(query_config, chunk_df)
            job_meta = {
                "job_id": job_info["job_id"],
                "job_url": job_info.get("job_url"),
                "status": "PENDING",
                "submitted_at": time.time(),
                "chunk_size": len(chunk_df),
            }

            job_path = self._job_path(chunk_id, query_idx)
            self._atomic_write(job_path, json.dumps(job_meta, indent=2))

            with self.job_lock:
                self.jobs[(chunk_id, query_idx)] = job_meta

            self.submit_queue.task_done()

    # ----------------------------
    # Polling thread
    # ----------------------------
    def _polling_worker(self):
        while not self.stop_event.is_set():
            # reload job files from disk
            for job_file in sorted(self.raw_path.glob("chunk_*_q*.job.json")):
                parts = job_file.stem.split("_")
                chunk_id = int(parts[1])
                query_idx = int(parts[2][1:])
                key = (chunk_id, query_idx)
                if key not in self.jobs:
                    try:
                        with open(job_file) as f:
                            jm = json.load(f)
                        with self.job_lock:
                            self.jobs[key] = jm
                    except Exception:
                        continue

            with self.job_lock:
                items = list(self.jobs.items())

            for (chunk_id, query_idx), meta in items:
                if meta.get("status") in ("COMPLETED", "ERROR"):
                    continue

                status = self.check_job_status(meta)
                if status == "COMPLETED":
                    payload = self.download_job_result(meta)
                    self._atomic_write(self._chunk_path(chunk_id, query_idx), payload)
                    self._atomic_write(self._marker_path(chunk_id, query_idx), "done")
                    meta["status"] = "COMPLETED"
                    meta["completed_at"] = time.time()
                    self._atomic_write(self._job_path(chunk_id, query_idx), json.dumps(meta, indent=2))
                    with self.job_lock:
                        self.jobs[(chunk_id, query_idx)] = meta
                elif status == "ERROR":
                    meta["status"] = "ERROR"
                    with self.job_lock:
                        self.jobs[(chunk_id, query_idx)] = meta
                    self._atomic_write(self._job_path(chunk_id, query_idx), json.dumps(meta, indent=2))
                else:
                    with self.job_lock:
                        self.jobs[(chunk_id, query_idx)]["status"] = status
                    self._atomic_write(self._job_path(chunk_id, query_idx), json.dumps(self.jobs[(chunk_id, query_idx)], indent=2))

            if getattr(self, "all_chunks_queued", False):
                with self.job_lock:
                    all_done = all(j.get("status") in ("COMPLETED", "ERROR") for j in self.jobs.values())
                if all_done:
                    break

            time.sleep(self.cfg.poll_interval)

    # ----------------------------
    # Main run loop
    # ----------------------------
    def run(self):
        df = pd.read_csv(self.cfg.input_csv)
        n = len(df)
        chunk_size = self.cfg.chunk_size

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
        self.submit_thread = threading.Thread(target=self._submission_worker, daemon=True)
        self.poll_thread = threading.Thread(target=self._polling_worker, daemon=True)
        self.submit_thread.start()
        self.poll_thread.start()

        # enqueue all chunks & queries
        for i in range(0, n, chunk_size):
            chunk_id = i // chunk_size
            chunk_df = df.iloc[i:i + chunk_size]

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

                self.submit_queue.put((chunk_id, query_idx, qcfg, chunk_df))

        self.all_chunks_queued = True
        self.submit_queue.join()
        self.stop_event.set()
        self.submit_thread.join()
        self.poll_thread.join()
