from pydantic import Field
from typing import Union, Annotated

from .allwise_p3as_mep import allwise_p3as_mep
from .neowiser_p1bs_psd import neowiser_p1bs_psd


TableType = Annotated[
    Union[allwise_p3as_mep, neowiser_p1bs_psd], Field(discriminator="name")
]
