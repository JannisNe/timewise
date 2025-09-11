from pathlib import Path

import numpy as np
import pytest
import json
from itertools import product

from timewise.types import TAPJobMeta
from timewise.io.download import DownloadConfig, Downloader
from timewise.util.csv_utils import get_n_rows
from dummy_tap import DummyTAPService


@pytest.fixture
def cfg(tmp_path) -> DownloadConfig:
    input_csv = Path(__file__).parent / "data" / "test_sample.csv"
    return DownloadConfig.model_validate(
        dict(
            input_csv=input_csv,
            base_dir=tmp_path,
            chunk_size=32,
            max_concurrent_jobs=1,
            poll_interval=1,
            queries=[
                {
                    "query": {
                        "type": "positional_allwise_p3as_mep",
                        "radius_arcsec": 6,
                        "magnitudes": True,
                        "fluxes": True,
                    },
                },
                {
                    "query": {
                        "type": "positional_neowiser_p1bs_psd",
                        "radius_arcsec": 6,
                        "magnitudes": True,
                        "fluxes": True,
                    },
                },
            ],
        )
    )


def test_downloader_creates_files(cfg):
    dl = Downloader(cfg)
    dl.service = DummyTAPService(baseurl="", chunksize=cfg.chunk_size)
    dl.run()

    n_chunks = int(np.ceil(get_n_rows(cfg.input_csv) / cfg.chunk_size))
    n_queries = len(cfg.queries)

    for q, c in product(range(n_queries), range(n_chunks)):
        assert dl._marker_path(c, q).exists()
        assert dl._chunk_path(c, q).exists()

        job_path = dl._job_path(c, q)
        assert job_path.exists()
        with job_path.open("r") as f:
            TAPJobMeta(**json.load(f))
