import logging
from typing import Dict, Literal

from .base import Query

logger = logging.getLogger(__name__)


class AllWISECntrQuery(Query):
    type: Literal["by_allwise_cntr_and_position"] = "by_allwise_cntr_and_position"
    radius_arcsec: float

    @property
    def input_columns(self) -> Dict[str, str]:
        return {
            "allwise_cntr": "int",
            "ra": "float",
            "dec": "float",
            self.original_id_key: "int",
        }

    def build(self) -> str:
        logger.debug(f"constructing query by AllWISE cntr for {self.table.name}")

        q = "SELECT \n\t"
        for k in self.columns:
            q += f"{self.table.name}.{k}, "
        q += f"\n\tmine.{self.original_id_key} \n"
        q += f"FROM\n\tTAP_UPLOAD.{self.upload_name} AS mine \n"
        q += f"RIGHT JOIN\n\t{self.table.name} \n"
        q += "WHERE \n"
        q += (
            f"\tCONTAINS(POINT('J2000',{self.table.name}.{self.table.ra_column},{self.table.name}.{self.table.dec_column}),"
            f"CIRCLE('J2000',mine.ra,mine.dec,{self.radius_arcsec / 3600:.18f}))=1 "
        )

        constraints = self.constraints + [
            f"{self.table.name}.{self.table.allwise_cntr_column} = {self.upload_name}.allwise_cntr"
        ]

        if len(constraints) > 0:
            q += " AND (\n"
            for c in constraints:
                q += f"\t{self.table.name}.{c} AND \n"
            q = q.strip(" AND \n")
            q += "\t)"

        logger.debug(f"\n{q}")
        return q
