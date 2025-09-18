from pydantic import BaseModel

from .io.config import DownloadConfig


class TimewiseConfig(BaseModel):
    download: DownloadConfig
