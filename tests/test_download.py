from pathlib import Path

import numpy as np
import pytest
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
            chunk_size=32,
            max_concurrent_jobs=1,
            poll_interval=1,
            backend={"type": "filesystem", "base_path": tmp_path / "test/download"},
            queries=[
                {
                    "type": "positional_allwise_p3as_mep",
                    "radius_arcsec": 6,
                    "magnitudes": True,
                    "fluxes": True,
                },
                {
                    "type": "positional_neowiser_p1bs_psd",
                    "radius_arcsec": 6,
                    "magnitudes": True,
                    "fluxes": True,
                },
            ],
        )
    )


def test_downloader_creates_files(cfg):
    dl = Downloader(cfg)
    dl.service = DummyTAPService(baseurl="", chunksize=cfg.chunk_size)
    dl.run()

    n_chunks = int(np.ceil(get_n_rows(cfg.input_csv) / cfg.chunk_size))

    for q, c in product(cfg.queries, range(n_chunks)):
        task = dl.get_task_id(c, q.hash)
        b = dl.backend
        assert b.is_done(task)
        assert b.data_exists(task)

        assert b.meta_exists(task)
        assert TAPJobMeta(**b.load_meta(task))
