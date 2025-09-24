import json
import logging
from pathlib import Path
from typing import Any, Literal
from astropy.table import Table

from .base import Backend
from ..types import TaskID


logger = logging.getLogger(__name__)


class FileSystemBackend(Backend):
    type: Literal["filesystem"] = "filesystem"
    base_path: Path

    # ----------------------------
    # Helpers for paths
    # ----------------------------
    def _meta_path(self, task: TaskID) -> Path:
        return self.base_path / f"{task}.meta.json"

    def _marker_path(self, task: TaskID) -> Path:
        return self.base_path / f"{task}.ok"

    def _data_path(self, task: TaskID) -> Path:
        return self.base_path / f"{task}.fits"

    # ----------------------------
    # Metadata
    # ----------------------------
    def save_meta(self, task: TaskID, meta: dict[str, Any]) -> None:
        path = self._meta_path(task)
        tmp = path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"writing {path}")
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(path)

    def load_meta(self, task: TaskID) -> dict[str, Any] | None:
        path = self._meta_path(task)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def meta_exists(self, task: TaskID) -> bool:
        return self._meta_path(task).exists()

    # ----------------------------
    # Markers
    # ----------------------------
    def mark_done(self, task: TaskID) -> None:
        mp = self._marker_path(task)
        mp.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"writing {mp}")
        mp.write_text("done")

    def is_done(self, task: TaskID) -> bool:
        return self._marker_path(task).exists()

    # ----------------------------
    # Data
    # ----------------------------
    def save_data(self, task: TaskID, content: Table) -> None:
        path = self._data_path(task)
        tmp = path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"writing {path}")
        content.write(tmp, format="fits")
        tmp.replace(path)

    def load_data(self, task: TaskID) -> Table:
        path = self._data_path(task)
        if not path.exists():
            raise FileNotFoundError(path)
        return Table.read(path, format="fits")

    def data_exists(self, task: TaskID) -> bool:
        return self._data_path(task).exists()
