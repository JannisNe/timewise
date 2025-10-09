import abc
from typing import ClassVar, List, Self
from pydantic import BaseModel, model_validator
from hashlib import sha256

from ..tables import TableType


class Query(abc.ABC, BaseModel):
    type: str
    upload_name: ClassVar[str] = "mine"

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

    @model_validator(mode="after")
    def check_columns(self) -> Self:
        for column in self.columns:
            if column not in self.table.columns_dtypes:
                raise KeyError(f"{column} not found in table {self.table.name}")
        return self

    @property
    @abc.abstractmethod
    def input_columns(self) -> dict[str, str]: ...

    @abc.abstractmethod
    def build(self) -> str: ...

    @property
    def adql(self) -> str:
        """ADQL string computed once per instance."""
        return self.build()

    @property
    def hash(self) -> str:
        return (
            self.type
            + "_"
            + self.table.name
            + "_"
            + sha256(self.adql.encode()).hexdigest()
        )
