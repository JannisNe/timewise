from typing import Literal
from .base import PositionalQuery


class PositionalNEOWISEQuery(PositionalQuery):
    type: Literal["positional_neowise"] = "positional_neowise"
    table_name = "neowiser_p1bs_psd"
    id_key = "allwise_cntr"
    magnitude_keys = ["w1mpro_ep", "w1sigmpro_ep", "w2mpro_ep", "w3sigmpro_ep"]
    flux_keys = ['w1flux_ep', 'w1sigflux_ep', 'w2flux_ep', 'w2sigflux_ep']