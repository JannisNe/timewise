from typing import Any
from importlib.util import find_spec

from pydantic import BaseModel, model_validator

if find_spec("slack_sdk"):
    SLACK_EXISTS = True
else:
    SLACK_EXISTS = False


class SlackbotConfig(BaseModel):
    token: str

    @model_validator(mode="before")
    @classmethod
    def check_slack(cls, data: Any) -> Any:
        if not SLACK_EXISTS:
            raise ModuleNotFoundError(
                "slack_sdk is not installed! "
                "Please make sure to install timewise with the 'slack' extra!"
            )
        return data
