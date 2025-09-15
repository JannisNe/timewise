from typing import Literal, List
from .base import PositionalQuery


class PositionalNEOWISEQuery(PositionalQuery):
    type: Literal["positional_neowiser_p1bs_psd"] = "positional_neowiser_p1bs_psd"
    table_name = "neowiser_p1bs_psd"
    columns: List[str] = [
        "ra",
        "dec",
        "mjd",
        "allwise_cntr",
        "w1mpro",
        "w1sigmpro",
        "w2mpro",
        "w2sigmpro",
        "w1flux",
        "w1sigflux",
        "w2flux",
        "w2sigflux",
    ]
