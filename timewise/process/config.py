from pathlib import Path
import logging

from pydantic import BaseModel

from .interface import AmpelInterface


logger = logging.getLogger(__name__)
DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "template.yml"


class AmpelConfig(BaseModel):
    mongo_db_name: str
    template_path: Path = DEFAULT_TEMPLATE_PATH
    uri: str = "localhost:27017"

    @property
    def input_mongo_db_name(self) -> str:
        return self.mongo_db_name + "_input"

    def build_interface(self, original_id_key: str, input_csv: Path) -> AmpelInterface:
        return AmpelInterface(
            mongo_db_name=self.mongo_db_name,
            orig_id_key=original_id_key,
            input_csv=input_csv,
            input_mongo_db_name=self.input_mongo_db_name,
            template_path=self.template_path,
            uri=self.uri,
        )
