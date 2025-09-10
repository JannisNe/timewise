from typing import Literal
from .base import PositionalQuery


class PositionalAllWISEQuery(PositionalQuery):
    type: Literal["positional_allwise"] = "positional_allwise"
    table_name = "allwise_p3as_mep"
    id_key = "cntr_mf"
    magnitude_keys = ["w1mpro", "w1sigmpro", "w2mpro", "w3sigmpro"]
    flux_keys = ['w1flux', 'w1sigflux', 'w2flux', 'w2sigflux']
