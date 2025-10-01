from pathlib import Path
from itertools import product

import pytest

from tests.dummy_tap import get_table_from_query_and_chunk
from timewise.io import DownloadConfig
from timewise.config import TimewiseConfig
from timewise.process import AmpelInterface
from tests.constants import DATA_DIR, INPUT_CSV_PATH, V0_KEYMAP


@pytest.fixture
def download_cfg(tmp_path) -> DownloadConfig:
    return DownloadConfig.model_validate(
        dict(
            input_csv=INPUT_CSV_PATH,
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
def timewise_config_path(tmp_path) -> Path:
    timewise_config_template_path = DATA_DIR / "test_download.yml"
    with timewise_config_template_path.open("r") as f:
        timewise_config = f.read()
    timewise_config = timewise_config.replace("BASE_PATH", str(tmp_path)).replace(
        "INPUT_CSV", str(INPUT_CSV_PATH)
    )
    timewise_config_path = tmp_path / "timewise_config.yml"
    with timewise_config_path.open("w") as f:
        f.write(timewise_config)

    return timewise_config_path


@pytest.fixture
def ampel_interface(timewise_config_path) -> AmpelInterface:
    cfg = TimewiseConfig.from_yaml(timewise_config_path)
    dl = cfg.download.build_downloader()
    for q, c in product(dl.queries, dl.chunker):
        data = get_table_from_query_and_chunk(q.adql, c.chunk_id)
        for ol, nl in V0_KEYMAP:
            if ol in data.columns:
                data.rename_column(ol, nl)
        task = dl.get_task_id(c, q)
        dl.backend.save_data(task, data)

    return cfg.build_ampel_interface()
