from pathlib import Path
import logging

from pydantic import BaseModel

from .prepare import AmpelPrepper
from .results import ResultsExtractor


logger = logging.getLogger(__name__)
DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "template.yml"


class AmpelConfig(BaseModel):
    mongo_db_name: str
    template_path: Path = DEFAULT_TEMPLATE_PATH
    uri: str = "localhost:27017"

    @property
    def input_mongo_db_name(self) -> str:
        return self.mongo_db_name + "_input"

    def build_prepper(self, original_id_key: str, input_csv: Path) -> AmpelPrepper:
        return AmpelPrepper(
            mongo_db_name=self.mongo_db_name,
            orig_id_key=original_id_key,
            input_csv=input_csv,
            input_mongo_db_name=self.input_mongo_db_name,
            template_path=self.template_path,
            uri=self.uri,
        )

    def build_extractor(self) -> ResultsExtractor:
        return ResultsExtractor(mongo_db_uri=self.uri, mongo_db_name=self.mongo_db_name)
