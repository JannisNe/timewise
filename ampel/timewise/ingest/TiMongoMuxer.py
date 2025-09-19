#!/usr/bin/env python
# File:                Ampel-ZTF/ampel/ztf/ingest/ZiMongoMuxer.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                14.12.2017
# Last Modified Date:  25.05.2021
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from bisect import bisect_right
from contextlib import suppress
from typing import Any

from pymongo import UpdateOne

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

    # Be idempotent for the sake it (not required for prod)
    idempotent: bool = False

    # Standard projection used when checking DB for existing PPS/ULS
    projection = {
        "_id": 0,
        "id": 1,
        "tag": 1,
        "channel": 1,
        "stock": 1,
        "body.mjd": 1,
        "body.fid": 1,
        "body.flux": 1,
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # used to check potentially already inserted pps
        self._photo_col = self.context.db.get_collection("t0")
        self._projection_spec = unflatten_dict(self.projection)
        self._run_id = (
            self.updates_buffer.run_id[0]
            if isinstance(self.updates_buffer.run_id, list)
            else self.updates_buffer.run_id
        )

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

        ops: list[UpdateOne] = []
        add_update = ops.append

        # Create set with datapoint ids from alert
        ids_dps_alert = {el["id"] for el in dps}

        # python set of ids of datapoints from DB
        ids_dps_db = {el["id"] for el in dps_db}

        # uniquify photopoints by jd, fid. For duplicate points,
        # choose the one with the larger id
        # (jd, fid) -> ids
        unique_dps_ids: dict[tuple[float, int], list[DataPointId]] = {}
        # id -> superseding ids
        ids_dps_superseded: dict[DataPointId, list[DataPointId]] = {}
        # id -> final datapoint
        unique_dps: dict[DataPointId, DataPoint] = {}

        for dp in dps_db + dps:
            # jd alone is not enough for matching pps because each time is associated with
            # two filters!
            key = (dp["body"]["mjd"], dp["body"]["fid"])

            if target := unique_dps_ids.get(key):
                # insert id in order
                idx = bisect_right(target, dp["id"])
                if idx == 0 or target[idx - 1] != dp["id"]:
                    target.insert(idx, dp["id"])
            else:
                unique_dps_ids[key] = [dp["id"]]

        # make sure no duplicate datapoints exist
        for key, simultaneous_dps in unique_dps_ids.items():
            assert len(simultaneous_dps) == 1, f"Duplicate photopoints at {key}!"

        # Part 2: Update new data points that are already superseded
        ############################################################

        # Difference between candids from the alert and candids present in DB
        ids_dps_to_insert = ids_dps_alert - ids_dps_db

        # TODO: add combine_dps

        return [dp for dp in dps if dp["id"] in ids_dps_to_insert], None

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
