import logging

from pymongo import MongoClient
import pandas as pd
from ampel.types import StockId

logger = logging.getLogger(__name__)


class ResultsExtractor:
    def __init__(self, mongo_db_name: str, mongo_db_uri: str):
        self.mongo_db_name = mongo_db_name
        self.mongo_db_uri = mongo_db_uri

    @property
    def client(self) -> MongoClient:
        return MongoClient(self.mongo_db_uri)

    def extract_stacked_lightcurves(self, stock_ids: list[StockId]) -> pd.DataFrame:
        col = self.client[self.mongo_db_name]["t1"]
        records = []
        index = []
        for i in col.find({"stock": {"$in": stock_ids}}):
            records.append(i["body"])
            index.append(i["stock"])
        return pd.DataFrame(records, index=index)
