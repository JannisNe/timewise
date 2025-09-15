from typing import Literal, List
from .base import PositionalQuery


class PositionalAllWISEQuery(PositionalQuery):
    type: Literal["positional_allwise_p3as_mep"] = "positional_allwise_p3as_mep"
    table_name = "allwise_p3as_mep"
    columns: List[str] = [
        "ra",
        "dec",
        "mjd",
        "cntr_mf",
        "w1mpro_ep",
        "w1sigmpro_ep",
        "w2mpro_ep",
        "w2sigmpro_ep",
        "w1flux_ep",
        "w1sigflux_ep",
        "w2flux_ep",
        "w2sigflux_ep",
    ]
