from typing import TypedDict

from timewise.query import QueryConfig


class TAPJobMeta(TypedDict):
    url: str
    status: str
    submitted: float
    last_checked: float
    input_length: int
    query: str
    query_config: QueryConfig | dict
    completed_at: float
