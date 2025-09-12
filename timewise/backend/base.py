import abc
from typing import Any, ClassVar
from pydantic import BaseModel
from astropy.table import Table
from ..types import TaskID


class Backend(abc.ABC, BaseModel):
    type: ClassVar[str]
    base_path: str | list[str]
    """
    Abstract persistence backend for jobs, results, and markers.
    Works with generic TaskIDs so it can be reused across Downloader/Processor.
    """

    # --- metadata ---
    @abc.abstractmethod
    def meta_exists(self, task: TaskID) -> bool: ...
    @abc.abstractmethod
    def save_meta(self, task: TaskID, meta: dict[str, Any]) -> None: ...
    @abc.abstractmethod
    def load_meta(self, task: TaskID) -> dict[str, Any] | None: ...

    # --- Markers ---
    @abc.abstractmethod
    def mark_done(self, task: TaskID) -> None: ...
    @abc.abstractmethod
    def is_done(self, task: TaskID) -> bool: ...

    # --- Data ---
    @abc.abstractmethod
    def save_data(self, task: TaskID, content: Table) -> None: ...
    @abc.abstractmethod
    def load_data(self, task: TaskID) -> Table: ...
    @abc.abstractmethod
    def data_exists(self, task: TaskID) -> bool: ...
