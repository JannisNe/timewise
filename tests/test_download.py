from pathlib import Path
import pytest
import json

from timewise.types import TAPJobMeta
from timewise.io.download import DownloadConfig, Downloader
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
            dry_run=False,
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

    # check job metadata files exist
    job_files = list((cfg.output_dir / "raw").glob("chunk_*_q*.json"))
    assert job_files, "No job files created"

    # check marker files exist
    marker_files = list((cfg.output_dir / "raw").glob("chunk_*_q*.ok"))
    assert marker_files, "No marker files created"

    # check contents are valid JSON
    with open(job_files[0]) as f:
        data = TAPJobMeta(**json.load(f))
    assert "job_id" in data
