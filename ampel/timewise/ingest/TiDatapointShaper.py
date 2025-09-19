#!/usr/bin/env python
# File:                Ampel-ZTF/ampel/ztf/ingest/ZiDataPointShaper.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                14.12.2017
# Last Modified Date:  10.05.2021
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from collections.abc import Iterable, Sequence
from typing import Any

from bson import encode

from ampel.abstract.AbsT0Unit import AbsT0Unit
from ampel.base.AmpelUnit import AmpelUnit
from ampel.content.DataPoint import DataPoint
from ampel.types import StockId, Tag
from ampel.util.hash import hash_payload
from ampel.ztf.ingest.tags import tags


class ZiDataPointShaperBase(AmpelUnit):
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
        setitem = dict.__setitem__
        popitem = dict.pop

        for photo_dict in arg:
            # Photopoint
            if photo_dict.get("candid"):
                # Cut path if present
                if photo_dict.get("pdiffimfilename"):
                    setitem(
                        photo_dict,
                        "pdiffimfilename",
                        photo_dict["pdiffimfilename"].split("/")[-1].replace(".fz", ""),
                    )

                ret_list.append(
                    {  # type: ignore[typeddict-item]
                        "id": photo_dict["candid"],
                        "stock": stock,
                        "tag": tags[photo_dict["programid"]][photo_dict["fid"]],
                        "body": photo_dict,
                    }
                )

                popitem(photo_dict, "candid", None)
                popitem(photo_dict, "programpi", None)
            elif "forcediffimflux" in photo_dict:
                ret_list.append(self._create_datapoint(stock, ["ZTF_FP"], photo_dict))
            elif "fcqfid" in photo_dict:
                ret_list.append(
                    self._create_datapoint(stock, ["ZTF_FP", "BTS_PHOT"], photo_dict)
                )
            else:
                ret_list.append(
                    {  # type: ignore[typeddict-item]
                        "id": self.ul_identity(photo_dict),
                        "tag": tags[photo_dict["programid"]][photo_dict["fid"]],
                        "stock": stock,
                        "body": {
                            "jd": photo_dict["jd"],
                            "diffmaglim": photo_dict["diffmaglim"],
                            "rcid": (
                                rcid
                                if (rcid := photo_dict.get("rcid")) is not None
                                else (photo_dict["pid"] % 10000) // 100
                            ),
                            "fid": photo_dict["fid"],
                            "programid": photo_dict["programid"],
                            #'pdiffimfilename': fname
                            #'pid': photo_dict['pid']
                        },
                    }
                )

        return ret_list

    def _create_datapoint(
        self, stock: StockId, tag: Sequence[Tag], body: dict[str, Any]
    ) -> DataPoint:
        """
        Create a Datapoint from stock, body, and tags, using the hash of the body as id
        """
        # ensure that keys are ordered
        sorted_body = dict(sorted(body.items()))
        # This is not a complete DataPoint as (channel,meta) is missing, set later. Should these be optional? or added default?
        return {  # type: ignore
            "id": hash_payload(encode(sorted_body), size=-self.digest_size * 8),
            "stock": stock,
            "tag": [*tags[body["programid"]][body["fid"]], *tag],
            "body": sorted_body,
        }

    def ul_identity(self, uld: dict[str, Any]) -> int:
        """
        Calculate a unique ID for an upper limit from:
          - jd, floored to the millisecond
          - readout quadrant number (extracted from pid)
          - diffmaglim, rounded to 1e-3
         Example::

                >>> ZiT0UpperLimitShaper().identity(
                        {
                          'diffmaglim': 19.024799346923828,
                          'fid': 2,
                          'jd': 2458089.7405324,
                          'pdiffimfilename': '/ztf/archive/sci/2017/1202/240532/ztf_20171202240532_000566_zr_c08_o_q1_scimrefdiffimg.fits.fz',
                          'pid': 335240532815,
                          'programid': 0
                        }
                )
                -3352405322819025
        """
        return (
            (int((self.JD2017 - uld["jd"]) * 1000000) * 10000000)
            - (
                (
                    rcid
                    if (rcid := uld.get("rcid")) is not None
                    else (uld["pid"] % 10000) // 100
                )
                * 100000
            )
            - round(uld["diffmaglim"] * 1000)
        )


class ZiDataPointShaper(ZiDataPointShaperBase, AbsT0Unit):
    def process(self, arg: Any, stock: None | StockId = None) -> list[DataPoint]:
        assert stock is not None
        return super().process(arg, stock)
