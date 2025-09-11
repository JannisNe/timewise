from typing import Literal
from .base import PositionalQuery


class PositionalNEOWISEQuery(PositionalQuery):
    type: Literal["positional_neowiser_p1bs_psd"] = "positional_neowiser_p1bs_psd"
    table_name = "neowiser_p1bs_psd"
    id_key = "allwise_cntr"
    magnitude_keys = ["w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro"]
    flux_keys = ["w1flux", "w1sigflux", "w2flux", "w2sigflux"]
