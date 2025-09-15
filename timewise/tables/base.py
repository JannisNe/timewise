from typing import ClassVar, Dict
from pydantic import BaseModel


class TableConfig(BaseModel):
    name: ClassVar[str]
    columns_dtypes: ClassVar[Dict[str, str]]
    ra_column: ClassVar[str]
    dec_column: ClassVar[str]
