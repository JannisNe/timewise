import logging
import yaml
import json
import os
import inspect
from pydantic import BaseModel, validator
import pandas as pd
import importlib

from timewise.parent_sample_base import ParentSampleBase
from timewise.wise_data_base import WISEDataBase


logger = logging.getLogger(__name__)


class TimewiseConfig(BaseModel):

    wise_data: WISEDataBase
    timewise_instructions: list[dict] = list()

    class Config:
        arbitrary_types_allowed = True

    @validator("timewise_instructions")
    def validate_instructions(cls, v: list[dict], values: dict):
        # get the WiseData class
        wise_data = values["wise_data"]
        wise_data_class_name = type(wise_data).__name__
        # collect its members
        members = inspect.getmembers(wise_data)
        # loop through the methods and the corresponding arguments that wre given in the instructions
        for instructions in v:
            for method, arguments in instructions.items():
                found = False
                # check each member for a fit
                for member_name, member in members:
                    if member_name == method:
                        found = True
                        # get the call signature of the member and see if it fits the given arguments
                        signature = inspect.signature(member)
                        param_list = list(signature.parameters)
                        # check if the member is a normal method, i.e. if the first arguments is 'self'
                        is_method = param_list[0] == "self"
                        _arguments = arguments or dict()
                        try:
                            if is_method:
                                signature.bind(WISEDataBase, **_arguments)
                            else:
                                signature.bind(**_arguments)
                        except TypeError as e:
                            raise ValueError(f"{wise_data_class_name}.{method}: {e}!")

                if not found:
                    raise ValueError(f"{wise_data_class_name} does not have a method {method}!")

        return v

    def run_config(self):
        logger.info("running config")
        for instructions in self.timewise_instructions:
            for method, arguments in instructions.items():
                _arguments = arguments or dict()
                logger.info(f"running {method} with arguments {_arguments}")
                self.wise_data.__getattribute__(method)(**_arguments)
        logger.info("successfully ran config")


class TimewiseConfigLoader(BaseModel):

    base_name: str
    filename: str = None
    class_module: str = "timewise"
    class_name: str = "WiseDataByVisit"
    min_sep_arcsec: float = 6.
    n_chunks: int = 1
    default_keymap: dict = {k: k for k in ["ra", "dec", "id"]}
    timewise_instructions: list[dict] = list()

    @validator("filename")
    def validate_file(cls, v: str):
        if v is not None:
            if not os.path.isfile(v):
                raise ValueError(f"No file {v}!")
        return v

    @validator("class_module")
    def validate_class_module(cls, v: str):
        try:
            importlib.import_module(v)
        except ImportError:
            raise ValueError(f"Could not import module {v}!")
        return v

    @validator("class_name")
    def validate_class_name(cls, v: str, values: dict):
        class_module = values["class_module"]
        try:
            getattr(importlib.import_module(class_module), v)
        except ImportError:
            raise ValueError(f"Could not find {v} in {class_module}")
        return v

    @validator("default_keymap")
    def validate_keymap(cls, v: dict):
        for k in ["ra", "dec", "id"]:
            if k not in v:
                raise ValueError(f"Keymap is missing key {k}!")
        return v

    def parse_config(self):

        logger.info(f"Parsing config")
        logger.debug(json.dumps(self.dict(), indent=4))

        _default_keymap = self.default_keymap
        _base_name = self.base_name
        _filename = self.filename
        _class_name = self.class_name

        class DynamicParentSample(ParentSampleBase):
            default_keymap = _default_keymap

            def __init__(self):
                super().__init__(_base_name)
                self.df = pd.read_csv(_filename)

                for k, v in self.default_keymap.items():
                    if v not in self.df.columns:
                        raise KeyError(f"Can not map '{v}' to '{k}': '{v}' not in table columns! Adjust keymap")

        wise_data_config = {
            "base_name": _base_name,
            "parent_sample_class": DynamicParentSample,
            "n_chunks": self.n_chunks,
            "min_sep_arcsec": self.min_sep_arcsec
        }
        wise_data_class = getattr(importlib.import_module(self.class_module), self.class_name)
        wise_data = wise_data_class(**wise_data_config)

        return TimewiseConfig(wise_data=wise_data, timewise_instructions=self.timewise_instructions)

    @classmethod
    def read_yaml(cls, filename):
        logger.debug(f"reading {filename}")
        with open(filename, "r") as f:
            config_dict = yaml.safe_load(f)
        logger.debug(f"config: {json.dumps(config_dict, indent=4)}")
        return cls(**config_dict)

    @classmethod
    def run_yaml(cls, filename):
        logger.info(f"running {filename}")
        config = cls.read_yaml(filename).parse_config()
        config.run_config()
