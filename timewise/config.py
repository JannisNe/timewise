from pathlib import Path
import yaml

import numpy as np
from pydantic import BaseModel, model_validator

from .io import DownloadConfig
from .process import AmpelConfig, AmpelInterface


class TimewiseConfig(BaseModel):
    download: DownloadConfig
    ampel: AmpelConfig

    @classmethod
    def from_yaml(cls, path: str | Path):
        path = Path(path)
        assert path.exists(), f"{path} not found!"
        with path.open("r") as f:
            config_dict = yaml.safe_load(f)
        return cls.model_validate(config_dict)

    @model_validator(mode="after")
    def validate_query_original_id_key(self) -> "TimewiseConfig":
        unique_keys = np.unique([q.original_id_key for q in self.download.queries])
        assert len(unique_keys) == 1, (
            "Can not use different 'original_id_key' in queries!"
        )
        return self

    def build_ampel_interface(self) -> AmpelInterface:
        return self.ampel.build_interface(
            self.download.queries[0].original_id_key, self.download.input_csv
        )
