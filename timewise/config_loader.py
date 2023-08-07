import logging
import yaml
import os
import inspect
from pydantic import BaseModel, field_validator, FieldValidationInfo
import pandas as pd

from timewise import WiseDataByVisit, WISEDataDESYCluster
from timewise.parent_sample_base import ParentSampleBase
from timewise.wise_data_base import WISEDataBase


logger = logging.getLogger(__name__)


wise_data_classes = {
    "WiseDataByVisit": WiseDataByVisit,
    "WISEDataDESYCluster": WISEDataDESYCluster
}


class TimewiseConfigLoader(BaseModel):

    base_name: str
    filename: str
    class_name: str = "WiseDataByVisit"
    min_sep_arcsec: float = 6.
    n_chunks: int = 1
    default_keymap: dict = {k: k for k in ["ra", "dec", "id"]}
    instructions: dict

    @field_validator("filename")
    @classmethod
    def validate_file(cls, v: str):
        if not os.path.isfile(v):
            raise ValueError(f"No file {v}!")
        return v

    @field_validator("class_name")
    @classmethod
    def validate_class_name(cls, v: str):
        if v not in wise_data_classes:
            available_classes = ", ".join(list(wise_data_classes.keys()))
            ValueError(f"WiseData class {v} not implemented! (Only {available_classes} are available)")
        return v

    @field_validator("default_keymap")
    @classmethod
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

        wise_data_config = {
            "base_name": _base_name,
            "parent_sample_class": DynamicParentSample,
            "n_chunks": self.n_chunks,
            "min_sep_arcsec": self.min_sep_arcsec
        }
        wise_data = wise_data_classes[_class_name](**wise_data_config)

        return wise_data, self.instructions

    @staticmethod
    def read_yaml(filename):
        logger.debug(f"reading {filename}")
        with open(filename, "r") as f:
            config_dict = yaml.safe_load(f)
        return TimewiseConfigLoader(**config_dict)


class TimewiseConfig(BaseModel):

    wise_data: WISEDataBase
    instructions: dict

    class Config:
        arbitrary_types_allowed = True

    @field_validator("instructions")
    @classmethod
    def validate_instructions(cls, v: dict, info: FieldValidationInfo):
        # get the WiseData class
        wise_data = wise_data_classes[info.data["wise_data"]]
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
                    try:
                        if is_method:
                            signature.bind(WiseDataByVisit, **arguments)
                        else:
                            signature.bind(**arguments)
                    except TypeError as e:
                        raise ValueError(f"{wise_data.__name__}.{method}: {e}!")

            if not found:
                raise ValueError(f"{wise_data.__name__} does not have a method {method}!")

        return v

    def run_config(self):
        logger.info("running config")
        for method, arguments in self.instructions.items():
            logger.debug(f"running {method} with arguments {arguments}")
            self.wise_data.__getattribute__(method)(**arguments)

    @staticmethod
    def read_yaml(filename):
        config_loader = TimewiseConfigLoader.read_yaml(filename)
        wise_data, instructions = config_loader.parse_config()
        return TimewiseConfig(
            wise_data=wise_data,
            instructions=instructions
        )

    @staticmethod
    def run_yaml(filename):
        logger.info(f"running {filename}")
        config = TimewiseConfig.read_yaml(filename)
        config.run_config()
