from pathlib import Path
import logging

import pandas as pd
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from ampel.cli.JobCommand import JobCommand
from ampel.types import StockId


logger = logging.getLogger(__name__)


class AmpelInterface:
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
        # if collection already exists, assume import was already done
        if "input" in self.client[self.input_mongo_db_name].list_collection_names():
            logger.debug(
                f"'input' collection already exists in '{self.input_mongo_db_name}'."
            )
            return

        logger.debug(f"importing {self.input_csv} into {self.input_mongo_db_name}")
        col = self.client[self.input_mongo_db_name]["input"]

        # create an index from stock id
        col.create_index([(self.orig_id_key, ASCENDING)], unique=True)
        col.insert_many(pd.read_csv(self.input_csv).to_dict(orient="records"))

    def make_ampel_job_file(self, cfg_path: Path) -> Path:
        logger.debug(f"loading ampel job template from {self.template_path}")
        with self.template_path.open("r") as f:
            template = f.read()

        ampel_job = (
            template.replace("TIMEWISE_CONFIG_PATH", str(cfg_path))
            .replace("ORIGINAL_ID_KEY", self.orig_id_key)
            .replace("INPUT_MONGODB_NAME", self.input_mongo_db_name)
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

    def run(self, timewise_cfg_path: Path, ampel_config_path: Path):
        ampel_job_path = self.prepare(timewise_cfg_path)
        cmd = JobCommand()
        parser = cmd.get_parser()
        args = vars(
            parser.parse_args(
                ["--schema", str(ampel_job_path), "--config", str(ampel_config_path)]
            )
        )
        logger.debug(args)
        cmd.run(args, unknown_args=())

    @property
    def client(self) -> MongoClient:
        return MongoClient(self.uri)

    @property
    def db(self) -> Database:
        return self.client[self.mongo_db_name]

    @property
    def t0(self) -> Collection:
        return self.db["t0"]

    @property
    def t1(self) -> Collection:
        return self.db["t1"]

    def extract_stacked_lightcurve(self, stock_id: StockId) -> pd.DataFrame:
        records = []
        for i, ic in enumerate(
            self.t1.find({"stock": stock_id, "unit": "T1StackVisits"})
        ):
            assert i == 0, f"More than one stacked lightcurve found for {stock_id}!"
            assert len(ic["body"]) == 1, (
                f"None or more than one stacking result found for {stock_id}!"
            )
            records = ic["body"][0]
        return pd.DataFrame(records)

    def extract_datapoints(self, stock_id: StockId) -> pd.DataFrame:
        records = []
        index = []
        for ic in self.t0.find({"stock": stock_id}):
            records.append(ic["body"])
            index.append(ic["id"])
        return pd.DataFrame(records, index=index)
