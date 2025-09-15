from typing import Literal
from .base import TableConfig


class neowiser_p1bs_psd(TableConfig):
    name: Literal["neowiser_p1bs_psd"] = "neowiser_p1bs_psd"
    columns_dtypes = {
        "ra": float,
        "dec": float,
        "mjd": float,
        "allwise_cntr": str,
        "w1mpro": float,
        "w1sigmpro": float,
        "w2mpro": float,
        "w2sigmpro": float,
        "w1flux": float,
        "w1sigflux": float,
        "w2flux": float,
        "w2sigflux": float,
    }
    ra_column = "ra"
    dec_column = "dec"
