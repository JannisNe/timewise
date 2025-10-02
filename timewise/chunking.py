from typing import Iterator
from pathlib import Path
import numpy as np
from numpy import typing as npt
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Chunk:
    def __init__(
        self, chunk_id: int, indices: npt.ArrayLike, row_indices: npt.ArrayLike
    ):
        self.chunk_id = chunk_id
        self.indices = indices
        self.row_numbers = row_indices


class Chunker:
    def __init__(self, input_csv: Path, chunk_size: int):
        self.input_csv = input_csv
        self.chunk_size = chunk_size
        self._n_rows = self._count_rows()
        logger.debug(f"found {self._n_rows} rows in {self.input_csv}")

    def _count_rows(self) -> int:
        chunk = 1024 * 1024  # Process 1 MB at a time.
        f = np.memmap(self.input_csv)
        num_newlines = sum(
            np.sum(f[i : i + chunk] == ord("\n")) for i in range(0, len(f), chunk)
        )
        del f
        return num_newlines - 1  # one header row

    def __len__(self) -> int:
        return int(np.ceil(self._n_rows / self.chunk_size))

    def __iter__(self) -> Iterator[Chunk]:
        for chunk_id in range(len(self)):
            yield self.get_chunk(chunk_id)

    def get_chunk(self, chunk_id: int) -> Chunk:
        if chunk_id >= len(self):
            raise IndexError(f"Invalid chunk_id {chunk_id}")
        start = chunk_id * self.chunk_size
        stop = min(start + self.chunk_size, self._n_rows)
        indices = pd.read_csv(self.input_csv, skiprows=start, nrows=stop - start).index
        logger.debug(f"chunk {chunk_id}: from {start} to {stop}")
        return Chunk(chunk_id, indices, np.arange(start=start, stop=stop))
