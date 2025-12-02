from typing import Annotated, TypeAlias, Union

from pydantic import Field

from .by_allwise_cntr import AllWISECntrQuery
from .positional import PositionalQuery

# Discriminated union of all query types
QueryType: TypeAlias = Annotated[Union[PositionalQuery, AllWISECntrQuery], Field(discriminator="type")]
