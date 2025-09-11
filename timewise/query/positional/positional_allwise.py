from typing import Literal
from .base import PositionalQuery


class PositionalAllWISEQuery(PositionalQuery):
    type: Literal["positional_allwise"] = "positional_allwise"
    table_name = "allwise_p3as_mep"
    id_key = "cntr_mf"
    magnitude_keys = ["w1mpro_ep", "w1sigmpro_ep", "w2mpro_ep", "w2sigmpro_ep"]
    flux_keys = ["w1flux_ep", "w1sigflux_ep", "w2flux_ep", "w2sigflux_ep"]
