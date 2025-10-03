from pathlib import Path
from typing import List
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from .download import Downloader
from ..query import QueryType
from ..backend import BackendType
from ..types import TYPE_MAP


class DownloadConfig(BaseModel):
    input_csv: Path
    chunk_size: int = 500_000
    max_concurrent_jobs: int = 4
    poll_interval: float = 10.0
    queries: List[QueryType] = Field(..., description="One or more queries per chunk")
    backend: BackendType = Field(..., discriminator="type")

    service_url: str = "https://irsa.ipac.caltech.edu/TAP"

    @model_validator(mode="after")
    def validate_input_csv_columns(self) -> "DownloadConfig":
        """Ensure that the input CSV contains all columns required by queries."""
        # only validate if the CSV actually exists
        if not self.input_csv.exists():
            raise ValueError(f"CSV file does not exist: {self.input_csv}")

        # read just the header and first 10 lines
        input_table = pd.read_csv(self.input_csv, nrows=10)

        missing_columns = set()
        wrong_dtype = set()
        for qc in self.queries:
            for col, dtype in qc.input_columns.items():
                if col not in input_table.columns:
                    missing_columns.add(col)
                else:
                    try:
                        input_table[col].astype(TYPE_MAP[dtype])
                    except Exception:
                        wrong_dtype.add(col)

        msg = f"CSV file {self.input_csv}: "
        if missing_columns:
            raise KeyError(msg + f"Missing required columns: {sorted(missing_columns)}")
        if wrong_dtype:
            raise TypeError(
                msg
                + f"Columns not convertable to right data type: {sorted(wrong_dtype)}"
            )

        return self

    def build_downloader(self) -> Downloader:
        return Downloader(
            service_url=self.service_url,
            input_csv=self.input_csv,
            chunk_size=self.chunk_size,
            backend=self.backend,
            queries=self.queries,
            max_concurrent_jobs=self.max_concurrent_jobs,
            poll_interval=self.poll_interval,
        )
