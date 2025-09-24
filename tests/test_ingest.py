import logging
from pathlib import Path
from itertools import product

import pytest
from astropy.table import Table, vstack
from ampel.cli.JobCommand import JobCommand
from pymongo import MongoClient

from timewise.process.ampel import make_ampel_job_file
from timewise.config import TimewiseConfig
from timewise.io.download import Downloader

from tests.dummy_tap import get_table_from_query_and_chunk

DATA_DIR = Path(__file__).parent / "data"
AMPEL_CONFIG_PATH = Path(__file__).parent.parent / "ampel_config.yml"


logger = logging.Logger(__name__)


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


def test_ingest(ampel_job_path):
    cmd = JobCommand()
    parser = cmd.get_parser()
    args = vars(
        parser.parse_args(
            ["--schema", str(ampel_job_path), "--config", str(AMPEL_CONFIG_PATH)]
        )
    )
    logger.debug(args)
    JobCommand().run(args, unknown_args=())

    client = MongoClient()
    t0 = client["test_ampel"].get_collection("t0")
    n_in_db = t0.count_documents({})

    file_contents = vstack(
        [
            Table.read(f, format="fits")
            for f in Path("tmp/test/download").glob("download_chunk*.fits")
        ]
    )

    missing = []
    duplicates = []
    for r in file_contents:
        f = {
            "body.ra": r["ra"],
            "body.dec": r["dec"],
            "body.mjd": r["mjd"],
        }
        n_doc = t0.count_documents(f)
        if n_doc == 0:
            missing.append(r)
        if n_doc > 1:
            duplicates.append(list(t0.find(f)))

    assert len(missing) == 0, f"Missing {len(missing)} documents in DB!\n{missing}"
    assert len(duplicates) == 0, f"Duplicate documents!\n{duplicates}"

    assert n_in_db == len(file_contents)
