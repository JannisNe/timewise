from pathlib import Path

import pytest
from itertools import product
import pandas as pd
import numpy as np

from timewise.types import TAPJobMeta
from timewise.io.download import DownloadConfig, Downloader
from timewise.chunking import Chunker
from dummy_tap import DummyTAPService, get_table_from_query_and_chunk


@pytest.fixture
def cfg(tmp_path) -> DownloadConfig:
    input_csv = Path(__file__).parent / "data" / "test_sample.csv"
    return DownloadConfig.model_validate(
        dict(
            input_csv=input_csv,
            chunk_size=32,
            max_concurrent_jobs=1,
            poll_interval=1,
            backend={"type": "filesystem", "base_path": tmp_path / "test/download"},
            queries=[
                {
                    "type": "positional",
                    "radius_arcsec": 6,
                    "table": {"name": "allwise_p3as_mep"},
                    "columns": [
                        "ra",
                        "dec",
                        "mjd",
                        "cntr_mf",
                        "w1mpro_ep",
                        "w1sigmpro_ep",
                        "w2mpro_ep",
                        "w2sigmpro_ep",
                        "w1flux_ep",
                        "w1sigflux_ep",
                        "w2flux_ep",
                        "w2sigflux_ep",
                    ],
                },
                {
                    "type": "positional",
                    "radius_arcsec": 6,
                    "table": {"name": "neowiser_p1bs_psd"},
                    "columns": [
                        "ra",
                        "dec",
                        "mjd",
                        "allwise_cntr",
                        "w1mpro",
                        "w1sigmpro",
                        "w2mpro",
                        "w2sigmpro",
                        "w1flux",
                        "w1sigflux",
                        "w2flux",
                        "w2sigflux",
                    ],
                },
            ],
        )
    )


def test_chunking(cfg):
    dl = Downloader(cfg)
    chunks = Chunker(input_csv=cfg.input_csv, chunk_size=cfg.chunk_size)
    for i, chunk in enumerate(chunks):
        chunk_data = dl.get_chunk_data(chunk)
        assert (
            len(pd.Index(chunk_data.orig_id).difference(range(i * 32, (i + 1) * 32)))
            == 0
        )


def test_downloader_creates_files(cfg):
    dl = Downloader(cfg)
    dl.service = DummyTAPService(baseurl="", chunksize=cfg.chunk_size)
    dl.run()

    chunks = Chunker(input_csv=cfg.input_csv, chunk_size=cfg.chunk_size)

    for q, c in product(cfg.queries, chunks):
        task = dl.get_task_id(c, q)
        b = dl.backend
        assert b.is_done(task)
        assert b.data_exists(task)

        assert b.meta_exists(task)
        assert TAPJobMeta(**b.load_meta(task))

        produced = dl.backend.load_data(task)
        reference = get_table_from_query_and_chunk(q.adql, c.chunk_id)

        for col in reference.colnames:
            if hasattr(reference[col], "mask"):
                m = ~(reference[col].mask & produced[col].mask)
            else:
                m = [True] * len(reference)
            assert sum(np.array(reference[col] - produced[col])[m]) == 0
