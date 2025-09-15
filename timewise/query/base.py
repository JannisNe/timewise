import abc
from typing import ClassVar, Type, List
from pydantic import BaseModel, computed_field
from hashlib import sha256

from ..tables import TableType


class Query(abc.ABC, BaseModel):
    type: ClassVar[str]
    upload_name: ClassVar[str] = "mine"
    input_columns: ClassVar[dict[str, Type]]

    original_id_key: str = "orig_id"
    constraints: List[str] = [
        "nb < 2",
        "na < 1",
        "cc_flags like '00%'",
        "qi_fact >= 1",
        "saa_sep >= 5",
        "moon_masked like '00%'",
    ]
    columns: List[str]
    table: TableType

    @abc.abstractmethod
    def build(self) -> str: ...

    @computed_field
    @property
    def adql(self) -> str:
        """ADQL string computed once per instance."""
        return self.build()

    @computed_field
    @property
    def hash(self) -> str:
        return self.type + sha256(self.adql.encode()).hexdigest()
