from typing import Literal
from .base import TableConfig


class allwise_p3as_mep(TableConfig):
    name: Literal["allwise_p3as_mep"] = "allwise_p3as_mep"
    columns_dtypes = {
        "ra": float,
        "dec": float,
        "mjd": float,
        "cntr_mf": str,
        "w1mpro_ep": float,
        "w1sigmpro_ep": float,
        "w2mpro_ep": float,
        "w2sigmpro_ep": float,
        "w1flux_ep": float,
        "w1sigflux_ep": float,
        "w2flux_ep": float,
        "w2sigflux_ep": float,
    }
    ra_column = "ra"
    dec_column = "dec"
