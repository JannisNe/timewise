import abc
from typing import ClassVar
from pydantic import BaseModel


class Query(abc.ABC, BaseModel):
    constraints: list[str] = [
        "nb < 2",
        "na < 1",
        "cc_flags like '00%'",
        "qi_fact >= 1",
        "saa_sep >= 5",
        "moon_masked like '00%'"
    ]
    original_id_key: str = "orig_id"
    input_columns: ClassVar[dict[str, type]]

    @abc.abstractmethod
    def build(self) -> str:
        ...
