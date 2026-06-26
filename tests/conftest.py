from pathlib import Path
from itertools import product
from typing import Generator
import re
import os
import tempfile

import pymongo
import yaml

import mongomock
import pytest
from ampel.core.AmpelContext import AmpelContext
from ampel.secret.AmpelVault import AmpelVault
from ampel.config.builder.DisplayOptions import DisplayOptions
from pymongo import MongoClient

from tests.dummy_tap import get_table_from_query_and_chunk
from timewise.io import DownloadConfig
from ampel.dev.DevAmpelContext import DevAmpelContext
from ampel.config.builder.DistConfigBuilder import DistConfigBuilder
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
def timewise_config_path(tmp_path, tmp_db_name, request) -> Path:
    timewise_config_template_path = DATA_DIR / "test_download.yml"
    with timewise_config_template_path.open("r") as f:
        timewise_config = f.read()
    timewise_config = (
        timewise_config.replace("BASE_PATH", str(tmp_path))
        .replace("INPUT_CSV", str(INPUT_CSV_PATH))
        .replace("MONGODB_NAME", tmp_db_name)
    )
    if (
        marker := request.node.get_closest_marker("ampel_template_filename")
    ) is not None:
        timewise_config += f"  template_path: {DATA_DIR / marker.args[0]}\n"
    timewise_config_path = tmp_path / "timewise_config.yml"
    with timewise_config_path.open("w") as f:
        f.write(timewise_config)

    return timewise_config_path


@pytest.fixture(scope="session")
def ampel_timewise_testing_config(tmp_path_factory, pytestconfig):
    """Path to an Ampel config file suitable for testing."""
    config_path = tmp_path_factory.mktemp("config") / "testing-config.yaml"
    if (config := pytestconfig.cache.get("testing_config", None)) is None:
        # build a config from all available ampel distributions
        cb = DistConfigBuilder(
            DisplayOptions(verbose=False, debug=False),
        )
        cb.load_distributions(prefixes=["ampel", "timewise"])
        config = cb.build_config(
            stop_on_errors=2,
            config_validator="ConfigValidator",
            get_unit_env=False,
        )
        assert config is not None
        # massage db settings for use with mongomock
        for db in config["mongo"]["databases"]:
            for collection in db["collections"]:
                # remove unsuported storageEngine options
                if "args" in collection and "storageEngine" in collection["args"]:
                    collection["args"].pop("storageEngine")
            # ensure that r and w modes share a client
            db["role"]["r"] = db["role"]["w"]
        pytestconfig.cache.set("testing_config", config)
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    return config_path


def pytest_addoption(parser):
    parser.addoption(
        "--mongodb-uri",
        action="store",
        default=None,
        help="URI of the MongoDB instance to use for tests",
    )


@pytest.fixture
def mongomock_client(
    monkeypatch, request
) -> mongomock.MongoClient | pymongo.MongoClient:
    mongodb_uri = request.config.getoption("--mongodb-uri")

    if mongodb_uri:
        # User provided a real MongoDB URI, don't patch.
        return MongoClient(mongodb_uri)

    # ignore codec_options in DataLoader
    monkeypatch.setattr("mongomock.codec_options.is_supported", lambda *args: None)
    # work around https://github.com/mongomock/mongomock/issues/912
    add_update = mongomock.collection.BulkOperationBuilder.add_update

    def _add_update(self, *args, sort=None, **kwargs):
        if sort is not None:
            raise NotImplementedError("sort not implemented in mongomock")
        return add_update(self, *args, **kwargs)

    monkeypatch.setattr(
        "mongomock.collection.BulkOperationBuilder.add_update", _add_update
    )

    client = mongomock.MongoClient()

    def get_client(*args, **kwargs):
        return client

    monkeypatch.setattr("ampel.core.AmpelDB.MongoClient", get_client)
    monkeypatch.setattr("pymongo.MongoClient", get_client)
    monkeypatch.setattr("timewise.process.interface.AmpelInterface.client", client)
    monkeypatch.setattr("ampel.timewise.t1.T1HDBSCAN.MongoClient", get_client)

    return client


@pytest.fixture
def ampel_interface(
    timewise_config_path,
    monkeypatch,
    ampel_timewise_testing_config: Path,
    ampel_vault: AmpelVault,
    mongomock_client: mongomock.MongoClient,
) -> AmpelInterface:
    cfg = TimewiseConfig.from_yaml(timewise_config_path)

    def get_mock_context(*args, db_prefix: str, **kwargs) -> AmpelContext:
        return DevAmpelContext.load(
            config=str(ampel_timewise_testing_config),
            vault=ampel_vault,
            purge_db=True,
            one_db=True,
            db_prefix=db_prefix,
        )

    monkeypatch.setattr(
        "ampel.cli.AbsCoreCommand.AbsCoreCommand.get_context", get_mock_context
    )

    dl = cfg.download.build_downloader()
    for q, c in product(dl.queries, dl.chunker):
        data = get_table_from_query_and_chunk(q.adql, c.chunk_id)
        ext = "_eq" if "allwise_p3as_mep" in q else ""
        for ol, nl in V0_KEYMAP:
            if ol in data.columns:
                data.rename_column(ol, nl + ext)
        task = dl.get_task_id(c, q)
        dl.backend.save_data(task, data)

    return cfg.build_ampel_interface()
