#!/usr/bin/env python
# File:                timewise/ampel/timewise/ingest/TiDataPointShaper.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                14.12.2017
# Last Modified Date:  19.09.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>

from collections.abc import Iterable, Sequence
from typing import Any

from bson import encode

from ampel.abstract.AbsT0Unit import AbsT0Unit
from ampel.base.AmpelUnit import AmpelUnit
from ampel.content.DataPoint import DataPoint
from ampel.types import StockId, Tag
from ampel.util.hash import hash_payload

from ampel.timewise.ingest.tags import tags


class TiDataPointShaperBase(AmpelUnit):
    """
    This class 'shapes' datapoints in a format suitable
    to be saved into the ampel database
    """

    # JD2017 is used to define upper limits primary IDs
    JD2017: float = 2457754.5
    #: Byte width of datapoint ids
    digest_size: int = 8

    # Mandatory implementation
    def process(self, arg: Iterable[dict[str, Any]], stock: StockId) -> list[DataPoint]:
        """
        :param arg: sequence of unshaped pps
        IMPORTANT:
        1) This method *modifies* the input dicts (it removes 'candid' and programpi),
        even if the unshaped pps are ReadOnlyDict instances
        2) 'stock' is not set here on purpose since it will conflict with the $addToSet operation
        """

        ret_list: list[DataPoint] = []
        popitem = dict.pop

        for photo_dict in arg:
            # Photopoint
            assert photo_dict.get("candid"), "photometry points does not have 'candid'!"
            ret_list.append(
                {  # type: ignore[typeddict-item]
                    "id": photo_dict["candid"],
                    "stock": stock,
                    "tag": tags[photo_dict["fid"]],
                    "body": photo_dict,
                }
            )

            popitem(photo_dict, "candid", None)

        return ret_list

    def _create_datapoint(
        self, stock: StockId, tag: Sequence[Tag], body: dict[str, Any]
    ) -> DataPoint:
        """
        Create a Datapoint from stock, body, and tags, using the hash of the body as id
        """
        # ensure that keys are ordered
        sorted_body = dict(sorted(body.items()))
        # The following is a comment from the original ampel.ztf.ingest.ZiDataPointShaperBase:
        # This is not a complete DataPoint as (channel,meta) is missing, set later.
        # Should these be optional? or added default?
        return {  # type: ignore
            "id": hash_payload(encode(sorted_body), size=-self.digest_size * 8),
            "stock": stock,
            "tag": [*tags[body["fid"]], *tag],
            "body": sorted_body,
        }

    def ul_identity(self, uld: dict[str, Any]) -> int:
        """
        This should not happen
        """
        raise NotImplementedError


class TiDataPointShaper(TiDataPointShaperBase, AbsT0Unit):
    def process(self, arg: Any, stock: None | StockId = None) -> list[DataPoint]:
        assert stock is not None
        return super().process(arg, stock)
