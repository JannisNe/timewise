from pathlib import Path
import yaml

from pydantic import BaseModel

from .io.config import DownloadConfig


class TimewiseConfig(BaseModel):
    download: DownloadConfig

    @classmethod
    def from_yaml(cls, path: str | Path):
        path = Path(path)
        assert path.exists(), f"{path} not found!"
        with path.open("r") as f:
            config_dict = yaml.safe_load(f)
        return cls.model_validate(config_dict)
