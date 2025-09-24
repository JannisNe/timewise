from pathlib import Path
from itertools import product

import pytest

from tests.dummy_tap import get_table_from_query_and_chunk
from timewise.io.config import DownloadConfig
from timewise.io.download import Downloader
from timewise.config import TimewiseConfig
from timewise.process.ampel import make_ampel_job_file


DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def download_cfg(tmp_path) -> DownloadConfig:
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


@pytest.fixture
def ampel_job_path(tmp_path) -> Path:
    timewise_config_template_path = DATA_DIR / "test_download.yml"
    with timewise_config_template_path.open("r") as f:
        timewise_config = f.read()
    timewise_config = timewise_config.replace("BASE_PATH", str(tmp_path))
    timewise_config_path = tmp_path / "timewise_config.yml"
    with timewise_config_path.open("w") as f:
        f.write(timewise_config)

    dl = Downloader(TimewiseConfig.from_yaml(timewise_config_path).download)
    for q, c in product(dl.cfg.queries, dl.chunker):
        data = get_table_from_query_and_chunk(q.adql, c.chunk_id)
        task = dl.get_task_id(c, q)
        dl.backend.save_data(task, data)

    return make_ampel_job_file(timewise_config_path)
