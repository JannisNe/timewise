import logging
from pathlib import Path

from astropy.table import Table, vstack
from ampel.cli.JobCommand import JobCommand
from pymongo import MongoClient

from timewise.config import TimewiseConfig


AMPEL_CONFIG_PATH = Path(__file__).parent.parent / "ampel_config.yml"


logger = logging.Logger(__name__)


def test_ingest(ampel_job_path, timewise_config_path):
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

    bp = TimewiseConfig.from_yaml(timewise_config_path).download.backend.base_path
    file_contents = vstack(
        [Table.read(f, format="fits") for f in bp.glob("download_chunk*.fits")]
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
