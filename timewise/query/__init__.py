from pydantic import BaseModel, Field
from typing import Union
from .positional.positional_allwise import PositionalAllWISEQuery
from .positional.positional_neowise import PositionalNEOWISEQuery

# Discriminated union of all query types
QueryType = Union[PositionalAllWISEQuery, PositionalNEOWISEQuery]


class QueryConfig(BaseModel):
    query: QueryType = Field(..., discriminator="type")
