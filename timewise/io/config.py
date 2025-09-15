from pathlib import Path
from typing import List

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from ..query import QueryType
from ..backend import BackendType


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

        # read just the header, avoid loading the entire file
        header = pd.read_csv(self.input_csv, nrows=0).columns

        missing_columns = set()
        for qc in self.queries:
            for col in qc.input_columns.keys():
                if col not in header:
                    missing_columns.add(col)

        if missing_columns:
            raise ValueError(
                f"CSV file {self.input_csv} is missing required columns: {sorted(missing_columns)}"
            )

        return self
