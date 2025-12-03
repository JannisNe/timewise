from typing import ClassVar, Dict, Literal, Type

from .base import TableConfig


class allwise_p3as_psd(TableConfig):
    name: Literal["allwise_p3as_psd"] = "allwise_p3as_psd"
    columns_dtypes: ClassVar[Dict[str, Type]] = {
        "ra": float,
        "dec": float,
        "mjd": float,
        "cntr": str,
        "w1mpro": float,
        "w1sigmpro": float,
        "w2mpro": float,
        "w2sigmpro": float,
        "w1flux": float,
        "w1sigflux": float,
        "w2flux": float,
        "w2sigflux": float,
    }
    ra_column: ClassVar[str] = "ra"
    dec_column: ClassVar[str] = "dec"
    allwise_cntr_column: ClassVar[str] = "cntr"
