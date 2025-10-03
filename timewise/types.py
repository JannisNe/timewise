from typing import TypedDict, NamedTuple

from timewise.query import QueryType


class TAPJobMeta(TypedDict):
    url: str
    status: str
    submitted: str
    last_checked: str
    input_length: int
    query: str
    query_config: QueryType | dict
    completed_at: str


class TaskID(NamedTuple):
    """
    Generic identifier for a unit of work.
    Can be extended by Downloader/Processor as needed.
    """

    namespace: str  # e.g. "downloader", "processor"
    key: str  # unique string, e.g. "chunk_0001_q0" or "mask_2025-01-01"

    def __str__(self):
        return f"{self.namespace}_{self.key}"


TYPE_MAP = {"float": float, "str": str, "int": int}
