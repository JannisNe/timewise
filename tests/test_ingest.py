import logging
from pathlib import Path

import pandas as pd
from ampel.cli.JobCommand import JobCommand
from pymongo import MongoClient

DATA_DIR = Path(__file__).parent / "data"
AMPEL_CONFIG_PATH = Path(__file__).parent.parent / "ampel_config.yml"


logger = logging.Logger(__name__)


def test_ingest():
    job_path = DATA_DIR / "test_ingest.yml"
    cmd = JobCommand()
    parser = cmd.get_parser()
    ampel_config_path = DATA_DIR.parent.parent / "ampel_config.yml"
    args = vars(
        parser.parse_args(
            ["--schema", str(job_path), "--config", str(ampel_config_path)]
        )
    )
    logger.debug(args)
    JobCommand().run(args, unknown_args=())

    client = MongoClient()
    t0 = client["test_ampel"].get_collection("t0")
    n_in_db = t0.count_documents({})

    file_contents = pd.concat(
        [pd.read_csv(f) for f in Path("tmp/test/download").glob("download_chunk*.csv")]
    )

    missing = []
    duplicates = []
    for i, r in file_contents.iterrows():
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
