#!/usr/bin/env python
# File:                ampel/timewise/ingest/TiMongoMuxer.py
# License:             BSD-3-Clause
# Author:              Jannis Necker
# Date:                19.09.2025
# Last Modified Date:  27.09.2025
# Last Modified By:    Jannis Necker

from bisect import bisect_right
from contextlib import suppress
from typing import Any


from ampel.abstract.AbsT0Muxer import AbsT0Muxer
from ampel.content.DataPoint import DataPoint
from ampel.types import DataPointId, StockId
from ampel.util.mappings import unflatten_dict


class ConcurrentUpdateError(Exception):
    """
    Raised when the t0 collection was updated during ingestion
    """

    ...


class TiMongoMuxer(AbsT0Muxer):
    """
    This class compares info between alert and DB so that only the needed info is ingested.
    It checks for duplicate datapoints.
    """

    # Standard projection used when checking DB for existing PPS/ULS
    projection = {
        "_id": 0,
        "id": 1,
        "tag": 1,
        "channel": 1,
        "stock": 1,
        "body.mjd": 1,
        "body.w1_flux": 1,
        "body.w2_flux": 1,
        "body.ra": 1,
        "body.dec": 1,
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # used to check potentially already inserted pps
        self._photo_col = self.context.db.get_collection("t0")
        self._projection_spec = unflatten_dict(self.projection)

    def process(
        self, dps: list[DataPoint], stock_id: None | StockId = None
    ) -> tuple[None | list[DataPoint], None | list[DataPoint]]:
        """
        :param dps: datapoints from alert
        :param stock_id: stock id from alert
        Attempt to determine which pps/uls should be inserted into the t0 collection,
        and which one should be marked as superseded.
        """
        # IPAC occasionally issues multiple subtraction candidates for the same
        # exposure and source, and these may be received in parallel by two
        # AlertConsumers.
        for _ in range(10):
            with suppress(ConcurrentUpdateError):
                return self._process(dps, stock_id)
        raise ConcurrentUpdateError(
            f"More than 10 iterations ingesting alert {dps[0]['id']}"
        )

    # NB: this 1-liner is a separate method to provide a patch point for race condition testing
    def _get_dps(self, stock_id: None | StockId) -> list[DataPoint]:
        return list(self._photo_col.find({"stock": stock_id}, self.projection))

    def _process(
        self, dps: list[DataPoint], stock_id: None | StockId = None
    ) -> tuple[None | list[DataPoint], None | list[DataPoint]]:
        """
        :param dps: datapoints from alert
        :param stock_id: stock id from alert
        Attempt to determine which pps/uls should be inserted into the t0 collection,
        and which one should be marked as superseded.
        """

        # Part 1: gather info from DB and alert
        #######################################

        # New pps/uls lists for db loaded datapoints
        dps_db = self._get_dps(stock_id)

        # Create set with datapoint ids from alert
        ids_dps_alert = {el["id"] for el in dps}

        # python set of ids of datapoints from DB
        ids_dps_db = {el["id"] for el in dps_db}

        # uniquify photopoints by jd, fid. For duplicate points,
        # choose the one with the larger id
        # (jd, fid) -> ids
        unique_dps_ids: dict[tuple[float, float, float], list[DataPointId]] = {}

        for dp in dps_db + dps:
            # jd alone is not enough for matching pps because each time is associated with
            # two filters! Also, if there can be multiple sources within the same frame which
            # leads to duplicate MJD and FID. Check position in addition.
            key = (
                dp["body"]["mjd"],
                dp["body"]["ra"],
                dp["body"]["dec"],
            )

            if target := unique_dps_ids.get(key):
                # insert id in order
                idx = bisect_right(target, dp["id"])
                if idx == 0 or target[idx - 1] != dp["id"]:
                    target.insert(idx, dp["id"])
            else:
                unique_dps_ids[key] = [dp["id"]]

        # make sure no duplicate datapoints exist
        for key, simultaneous_dps in unique_dps_ids.items():
            dps_db_wrong = [dp for dp in dps_db if dp["id"] in simultaneous_dps]
            dps_wrong = [dp for dp in dps if dp["id"] in simultaneous_dps]
            msg = f"stockID {stock_id}: Duplicate photopoints at {key}!\nDPS from DB:\n{dps_db_wrong}\nNew DPS:\n{dps_wrong}"
            assert len(simultaneous_dps) == 1, msg

        # Part 2: Update new data points that are already superseded
        ############################################################

        # Difference between candids from the alert and candids present in DB
        ids_dps_to_insert = ids_dps_alert - ids_dps_db
        dps_to_insert = [dp for dp in dps if dp["id"] in ids_dps_to_insert]
        dps_to_combine = [
            dp for dp in dps + dps_db if dp["id"] in ids_dps_alert | ids_dps_db
        ]
        self.logger.debug(
            f"Got {len(ids_dps_alert)} datapoints from alerts, "
            f"found {len(dps_db)} in DB, "
            f"inserting {len(dps_to_insert)} datapoints, "
            f"combining {len(dps_to_combine)} datapoints"
        )

        return dps_to_insert, dps_to_combine

    def _project(self, doc, projection) -> DataPoint:
        out: dict[str, Any] = {}
        for key, spec in projection.items():
            if key not in doc:
                continue

            if isinstance(spec, dict):
                item = doc[key]
                if isinstance(item, list):
                    out[key] = [self._project(v, spec) for v in item]
                elif isinstance(item, dict):
                    out[key] = self._project(item, spec)
            else:
                out[key] = doc[key]

        return out  # type: ignore[return-value]
