from pathlib import Path
from itertools import product
from typing import Generator
import re
import os
import tempfile

import pytest
from pymongo import MongoClient

from tests.dummy_tap import get_table_from_query_and_chunk
from timewise.io import DownloadConfig
from timewise.config import TimewiseConfig
from timewise.process import AmpelInterface
from tests.constants import DATA_DIR, INPUT_CSV_PATH, V0_KEYMAP

from ampel.log.AmpelLogger import AmpelLogger


# reset AmpelLoggers to make sure it is not trying to write to closed sys.stdout/sys.stderr
@pytest.fixture(autouse=True)
def reset_ampel_logger():
    AmpelLogger.loggers.clear()  # if available


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
def tmp_db_name(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Generator[str, None, None]:
    """Return a temporary Path with base root removed from the string representation."""

    # Create a safe name from the test function
    name = re.sub(r"[\W]", "_", request.node.name)[:30]

    # Create temp dir
    path = tmp_path_factory.mktemp(name, numbered=True)

    # Strip the base root from the path
    from_env = os.environ.get("PYTEST_DEBUG_TEMPROOT")
    temproot = str(Path(from_env or tempfile.gettempdir()).resolve())
    db_name = (
        str(path).replace(temproot, "").replace("/", "_").replace("_pytest-of-", "")
    )

    # Yield to test
    yield db_name

    try:
        client = MongoClient()
        for d in client.list_database_names():
            if d.startswith(db_name):
                client.drop_database(d)
    except Exception:
        pass


@pytest.fixture
def timewise_config_path(tmp_path, tmp_db_name) -> Path:
    timewise_config_template_path = DATA_DIR / "test_download.yml"
    with timewise_config_template_path.open("r") as f:
        timewise_config = f.read()
    timewise_config = (
        timewise_config.replace("BASE_PATH", str(tmp_path))
        .replace("INPUT_CSV", str(INPUT_CSV_PATH))
        .replace("MONGODB_NAME", tmp_db_name)
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
