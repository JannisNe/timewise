from typing import Union

from .base import Backend
from .filesystem import FileSystemBackend

BackendType = Union[FileSystemBackend]
