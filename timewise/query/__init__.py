from pydantic import Field
from typing import Union, Annotated, TypeAlias
from .positional import PositionalQuery

# Discriminated union of all query types
QueryType: TypeAlias = Annotated[Union[PositionalQuery], Field(discriminator="type")]
