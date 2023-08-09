import logging
import yaml
import json
import os
import inspect
from pydantic import BaseModel, validator
import pandas as pd

from timewise import WiseDataByVisit, WISEDataDESYCluster
from timewise.parent_sample_base import ParentSampleBase
from timewise.wise_data_base import WISEDataBase


logger = logging.getLogger(__name__)


wise_data_classes = {
    "WiseDataByVisit": WiseDataByVisit,
    "WISEDataDESYCluster": WISEDataDESYCluster
}


class TimewiseConfig(BaseModel):

    wise_data: WISEDataBase
    instructions: dict

    class Config:
        arbitrary_types_allowed = True

    @validator("instructions")
    def validate_instructions(cls, v: dict, values: dict):
        # get the WiseData class
        wise_data = values["wise_data"]
        wise_data_class_name = type(wise_data).__name__
        # collect its members
        members = inspect.getmembers(wise_data)
        # loop through the methods and the corresponding arguments that wre given in the instructions
        for method, arguments in v.items():
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
                            signature.bind(WiseDataByVisit, **_arguments)
                        else:
                            signature.bind(**_arguments)
                    except TypeError as e:
                        raise ValueError(f"{wise_data_class_name}.{method}: {e}!")

            if not found:
                raise ValueError(f"{wise_data_class_name} does not have a method {method}!")

        return v

    def run_config(self):
        logger.info("running config")
        for method, arguments in self.instructions.items():
            _arguments = arguments or dict()
            logger.debug(f"running {method} with arguments {_arguments}")
            self.wise_data.__getattribute__(method)(**_arguments)
        logger.info("successfully ran config")


class TimewiseConfigLoader(BaseModel):

    base_name: str
    filename: str
    class_name: str = "WiseDataByVisit"
    min_sep_arcsec: float = 6.
    n_chunks: int = 1
    default_keymap: dict = {k: k for k in ["ra", "dec", "id"]}
    instructions: dict

    @validator("filename")
    def validate_file(cls, v: str):
        if not os.path.isfile(v):
            raise ValueError(f"No file {v}!")
        return v

    @validator("class_name")
    def validate_class_name(cls, v: str):
        if v not in wise_data_classes:
            available_classes = ", ".join(list(wise_data_classes.keys()))
            ValueError(f"WiseData class {v} not implemented! (Only {available_classes} are available)")
        return v

    @validator("default_keymap")
    def validate_keymap(cls, v: dict):
        for k in ["ra", "dec", "id"]:
            if k not  in v:
                raise ValueError(f"Keymap is missing key {k}!")
        return v

    def parse_config(self):

        logger.info(f"Parsing config")

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
        wise_data = wise_data_classes[_class_name](**wise_data_config)

        return TimewiseConfig(wise_data=wise_data, instructions=self.instructions)

    @staticmethod
    def read_yaml(filename):
        logger.debug(f"reading {filename}")
        with open(filename, "r") as f:
            config_dict = yaml.safe_load(f)
        logger.debug(f"config: {json.dumps(config_dict, indent=4)}")
        return TimewiseConfigLoader(**config_dict).parse_config()

    @staticmethod
    def run_yaml(filename):
        logger.info(f"running {filename}")
        config = TimewiseConfigLoader.read_yaml(filename)
        config.run_config()
