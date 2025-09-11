from pydantic import BaseModel, Field
from typing import Union
from .positional.positional_allwise_p3as_mep import PositionalAllWISEQuery
from .positional.positional_neowiser_p1bs_psd import PositionalNEOWISEQuery

# Discriminated union of all query types
QueryType = Union[PositionalAllWISEQuery, PositionalNEOWISEQuery]


class QueryConfig(BaseModel):
    query: QueryType = Field(..., discriminator="type")
