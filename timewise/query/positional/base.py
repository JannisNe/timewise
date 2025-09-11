import logging
from typing import ClassVar

from ..base import Query

logger = logging.getLogger(__name__)


class PositionalQuery(Query):
    radius_arcsec: float
    magnitudes: bool = False
    fluxes: bool = True
    input_columns = {"ra": float, "dec": float, "orig_id": int}

    table_name: ClassVar[str]
    id_key: ClassVar[str]
    magnitude_keys: ClassVar[list[str]]
    flux_keys: ClassVar[list[str]]

    ra_key: ClassVar[str] = "ra"
    dec_key: ClassVar[str] = "dec"
    time_key: ClassVar[str] = "mjd"

    def build(self) -> str:
        logger.debug(f"constructing positional query for {self.table_name}")
        lum_keys = list()
        if self.magnitudes:
            lum_keys += self.magnitude_keys
        if self.fluxes:
            lum_keys += self.flux_keys
        keys = [self.ra_key, self.dec_key, self.time_key, self.id_key] + lum_keys

        q = "SELECT \n\t"
        for k in keys:
            q += f"{self.table_name}.{k}, "
        q += f"\n\tmine.{self.original_id_key} \n"
        q += f"FROM\n\tTAP_UPLOAD.{self.upload_name} AS mine \n"
        q += f"RIGHT JOIN\n\t{self.table_name} \n"
        q += "WHERE \n"
        q += (
            f"\tCONTAINS(POINT('J2000',{self.table_name}.{self.ra_key},{self.table_name}.{self.dec_key}),"
            f"CIRCLE('J2000',mine.ra,mine.dec,{self.radius_arcsec / 3600:.18f}))=1 "
        )

        if len(self.constraints) > 0:
            q += " AND (\n"
            for c in self.constraints:
                q += f"\t{self.table_name}.{c} AND \n"
            q = q.strip(" AND \n")
            q += "\t)"

        logger.debug(f"\n{q}")
        return q
