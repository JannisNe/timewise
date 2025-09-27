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

    def extract_stacked_lightcurve(self, stock_id: StockId) -> pd.DataFrame:
        col = self.client[self.mongo_db_name]["t1"]
        records = []
        for i, ic in enumerate(col.find({"stock": stock_id, "unit": "T1StackVisits"})):
            assert i == 0, f"More than one stacked lightcurve found for {stock_id}!"
            assert len(ic["body"]) == 1, (
                f"None or more than one stacking result found for {stock_id}!"
            )
            records = ic["body"][0]
        return pd.DataFrame(records)
