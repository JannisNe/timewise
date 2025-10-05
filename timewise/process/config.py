from pathlib import Path
import logging
from typing import Dict, Any

from pydantic import BaseModel, field_validator

from .interface import AmpelInterface


logger = logging.getLogger(__name__)
DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "template.yml"


class AmpelConfig(BaseModel):
    mongo_db_name: str
    template_path: Path = DEFAULT_TEMPLATE_PATH
    uri: str = "localhost:27017"
    input_mongo_db_name: str = ""

    @field_validator("input_mongo_db_name", mode="after")  # type: ignore
    @classmethod
    def default_input_db_name(
        cls, input_mongo_db_name: str, values: Dict[str, Any]
    ) -> str:
        if not input_mongo_db_name:
            return values["mongo_db_name"] + "_input"
        return input_mongo_db_name

    def build_interface(self, original_id_key: str, input_csv: Path) -> AmpelInterface:
        return AmpelInterface(
            mongo_db_name=self.mongo_db_name,
            orig_id_key=original_id_key,
            input_csv=input_csv,
            input_mongo_db_name=self.input_mongo_db_name,
            template_path=self.template_path,
            uri=self.uri,
        )
