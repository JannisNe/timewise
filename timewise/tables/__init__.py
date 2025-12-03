from typing import Annotated, Union

from pydantic import Field

from .allwise_p3as_mep import allwise_p3as_mep
from .allwise_p3as_psd import allwise_p3as_psd
from .neowiser_p1bs_psd import neowiser_p1bs_psd

TableType = Annotated[
    Union[allwise_p3as_mep, neowiser_p1bs_psd, allwise_p3as_psd], Field(discriminator="name")
]
