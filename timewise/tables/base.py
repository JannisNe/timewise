from typing import ClassVar, Dict, Type
from pydantic import BaseModel


class TableConfig(BaseModel):
    name: str
    columns_dtypes: ClassVar[Dict[str, Type]]
    ra_column: ClassVar[str]
    dec_column: ClassVar[str]
