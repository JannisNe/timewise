from pathlib import Path
import logging

from pydantic import BaseModel


logger = logging.getLogger(__name__)
DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "template.yml"


class AmpelConfig(BaseModel):
    mongo_db_name: str
    template_path: Path = DEFAULT_TEMPLATE_PATH
    uri: str = "localhost:27017"

    @property
    def input_mongo_db_name(self) -> str:
        return self.mongo_db_name + "_input"
