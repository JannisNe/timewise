from pathlib import Path
import logging

import pandas as pd
from pymongo import MongoClient


logger = logging.getLogger(__name__)


class AmpelPrepper:
    def __init__(
        self,
        mongo_db_name: str,
        orig_id_key: str,
        input_csv: Path,
        input_mongo_db_name: str,
        template_path: str | Path,
        uri: str,
    ):
        self.mongo_db_name = mongo_db_name
        self.orig_id_key = orig_id_key
        self.input_csv = input_csv
        self.input_mongo_db_name = input_mongo_db_name
        self.template_path = Path(template_path)
        self.uri = uri

    def import_input(self):
        logger.debug(f"importing {self.input_csv} into {self.input_mongo_db_name}")
        client = MongoClient(host=self.uri)
        client[self.input_mongo_db_name]["input"].insert_many(
            pd.read_csv(self.input_csv).to_dict(orient="records")
        )

    def make_ampel_job_file(self, cfg_path: Path) -> Path:
        logger.debug(f"loading ampel job template from {self.template_path}")
        with self.template_path.open("r") as f:
            template = f.read()

        ampel_job = (
            template.replace("TIMEWISE_CONFIG_PATH", str(cfg_path))
            .replace("ORIGINAL_ID_KEY", self.orig_id_key)
            .replace("MONGODB_NAME", self.mongo_db_name)
        )

        ampel_job_path = cfg_path.parent / f"{cfg_path.stem}_ampel_job.yml"
        logger.info(f"writing ampel job to {ampel_job_path}")
        with ampel_job_path.open("w") as f:
            f.write(ampel_job)

        return ampel_job_path

    def prepare(self, cfg_path: Path) -> Path:
        self.import_input()
        return self.make_ampel_job_file(cfg_path)
