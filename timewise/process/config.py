from pathlib import Path
import logging

from pydantic import BaseModel, model_validator

from .interface import AmpelInterface


logger = logging.getLogger(__name__)
DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "template.yml"


class AmpelConfig(BaseModel):
    mongo_db_name: str
    template_path: Path = DEFAULT_TEMPLATE_PATH
    uri: str = "localhost:27017"
    # will default to <mongo_db_name>_input
    input_mongo_db_name: str = ""

    @model_validator(mode="after")  # type: ignore
    def default_input_db_name(self) -> "AmpelConfig":
        if not self.input_mongo_db_name:
            self.input_mongo_db_name = self.mongo_db_name + "_input"
        return self

    def build_interface(self, original_id_key: str, input_csv: Path) -> AmpelInterface:
        return AmpelInterface(
            mongo_db_name=self.mongo_db_name,
            orig_id_key=original_id_key,
            input_csv=input_csv,
            input_mongo_db_name=self.input_mongo_db_name,
            template_path=self.template_path,
            uri=self.uri,
        )
