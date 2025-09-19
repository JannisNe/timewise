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
from ampel.content.MetaRecord import MetaRecord
from ampel.types import DataPointId, StockId
from ampel.util.mappings import unflatten_dict


class ConcurrentUpdateError(Exception):
    """
    Raised when the t0 collection was updated during ingestion
    """

    ...


class ZiMongoMuxer(AbsT0Muxer):
    """
    This class compares info between alert and DB so that only the needed info is ingested later.
    Also, it marks potentially reprocessed datapoints as superseded.

    :param check_reprocessing: whether the ingester should check if photopoints were reprocessed
    (costs an additional DB request per transient). Default is (and should be) True.

    :param alert_history_length: alerts must not contain all available info for a given transient.
    IPAC generated alerts for ZTF for example currently provide a photometric history of 30 days.
    Although this number is unlikely to change, there is no reason to use a constant in code.
    """

    check_reprocessing: bool = True
    alert_history_length: int = 30

    # Be idempotent for the sake it (not required for prod)
    idempotent: bool = False

    # True: Alert + DB dps will be combined into state
    # False: Only the alert dps will be combined into state
    db_complete: bool = True

    # Standard projection used when checking DB for existing PPS/ULS
    projection = {
        "_id": 0,
        "id": 1,
        "tag": 1,
        "channel": 1,
        "excl": 1,
        "stock": 1,
        "body.jd": 1,
        "body.programid": 1,
        "body.fid": 1,
        "body.rcid": 1,
        "body.magpsf": 1,
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
        if self.check_reprocessing:
            add_update = ops.append
        else:
            add_update = self.updates_buffer.add_t0_update

        # Create set with datapoint ids from alert
        ids_dps_alert = {el["id"] for el in dps}

        # python set of ids of datapoints from DB
        ids_dps_db = {el["id"] for el in dps_db}

        # uniquify photopoints by jd, rcid. For duplicate points,
        # choose the one with the larger id
        # (jd, rcid) -> ids
        unique_dps_ids: dict[tuple[float, int], list[DataPointId]] = {}
        # id -> superseding ids
        ids_dps_superseded: dict[DataPointId, list[DataPointId]] = {}
        # id -> final datapoint
        unique_dps: dict[DataPointId, DataPoint] = {}

        for dp in dps_db + dps:
            # jd alone is actually enough for matching pps reproc, but an upper limit can
            # be associated with multiple stocks at the same jd. here, match also by rcid
            key = (dp["body"]["jd"], dp["body"]["rcid"])

            # print(dp['id'], key, key in unique_dps)

            if target := unique_dps_ids.get(key):
                # insert id in order
                idx = bisect_right(target, dp["id"])
                if idx == 0 or target[idx - 1] != dp["id"]:
                    target.insert(idx, dp["id"])
            else:
                unique_dps_ids[key] = [dp["id"]]

        # build set of supersessions
        for simultaneous_dps in unique_dps_ids.values():
            for i in range(len(simultaneous_dps) - 1):
                ids_dps_superseded[simultaneous_dps[i]] = simultaneous_dps[i + 1 :]

        # build final set of datapoints, preferring entries loaded from the db
        final_dps_set = {v[-1] for v in unique_dps_ids.values()}
        for dp in dps_db + dps:
            if dp["id"] in final_dps_set and dp["id"] not in unique_dps:
                unique_dps[dp["id"]] = dp

        # Part 2: Update new data points that are already superseded
        ############################################################

        # Difference between candids from the alert and candids present in DB
        ids_dps_to_insert = ids_dps_alert - ids_dps_db

        for dp in dps:
            # If alerts were received out of order, this point may already be superseded.
            # Update it in place so that it can be inserted with the correct tags.
            if (
                self.check_reprocessing
                and dp["id"] in ids_dps_to_insert
                and dp["id"] in ids_dps_superseded
            ):
                self.logger.info(
                    f"Marking datapoint {dp['id']} "
                    f"as superseded by {ids_dps_superseded[dp['id']]}"
                )

                # point is newly superseded
                if "SUPERSEDED" not in dp["tag"]:
                    # mutate a copy, as the default tag list may be shared between all datapoints
                    dp["tag"] = list(dp["tag"])
                    dp["tag"].append("SUPERSEDED")  # type: ignore[attr-defined]

                # point may be superseded by more than one new datapoint
                meta: list[MetaRecord] = list(dp.get("meta", []))
                for newId in ids_dps_superseded[dp["id"]]:
                    if not any(
                        m.get("extra", {}).get("newId") == newId
                        for m in meta
                        if m.get("tag") == "SUPERSEDED"
                    ):
                        meta.append(
                            {
                                "run": self._run_id,
                                "traceid": {"muxer": self._trace_id},
                                "tag": "SUPERSEDED",
                                "extra": {"newId": newId},
                            }
                        )
                dp["meta"] = meta

        # Part 3: Update old data points that are superseded
        ####################################################

        if self.check_reprocessing:
            for dp in dps_db or []:
                if dp["id"] in ids_dps_superseded:
                    self.logger.info(
                        f"Marking datapoint {dp['id']} "
                        f"as superseded by {ids_dps_superseded[dp['id']]}"
                    )

                    # point is newly superseded
                    if "SUPERSEDED" not in dp["tag"]:
                        dp["tag"].append("SUPERSEDED")  # type: ignore[attr-defined]
                        add_update(
                            UpdateOne(
                                {
                                    "id": dp["id"],
                                },
                                {"$addToSet": {"tag": "SUPERSEDED"}},
                            )
                        )

                    # point may be superseded by more than one new datapoint
                    meta = list(dp.get("meta", []))
                    for newId in ids_dps_superseded[dp["id"]]:
                        if not any(
                            m.get("extra", {}).get("newId") == newId
                            for m in meta
                            if m.get("tag") == "SUPERSEDED"
                        ):
                            entry: MetaRecord = {
                                "run": self._run_id,
                                "traceid": {"muxer": self._trace_id},
                                "tag": "SUPERSEDED",
                                "extra": {"newId": newId},
                            }
                            meta.append(entry)
                            # issue idempotent update
                            add_update(
                                UpdateOne(
                                    {
                                        "id": dp["id"],
                                        "meta": {
                                            "$not": {
                                                "$elemMatch": {
                                                    "tag": "SUPERSEDED",
                                                    "extra.newId": newId,
                                                }
                                            }
                                        },
                                    },
                                    {"$push": {"meta": entry}},
                                )
                            )
                    dp["meta"] = meta

        # Part 4: commit ops and check for conflicts
        ############################################
        if self.check_reprocessing:
            # Commit ops, retrying on upsert races
            if ops:
                self.updates_buffer.call_bulk_write("t0", ops)
            # If another query returns docs not present in the first query, the
            # set of superseded photopoints may be incomplete.
            if concurrent_updates := (
                {
                    doc["id"]
                    for doc in self._photo_col.find({"stock": stock_id}, {"id": 1})
                }
                - (ids_dps_db | ids_dps_alert)
            ):
                raise ConcurrentUpdateError(
                    f"T0 collection contains {len(concurrent_updates)} "
                    f"extra photopoints: {concurrent_updates}"
                )

        # The union of the datapoints drawn from the db and
        # from the alert will be part of the t1 document
        if self.db_complete:
            # DB might contain datapoints newer than the newest alert dp
            # https://github.com/AmpelProject/Ampel-ZTF/issues/6
            latest_alert_jd = dps[0]["body"]["jd"]
            dps_combine = [
                dp for dp in unique_dps.values() if dp["body"]["jd"] <= latest_alert_jd
            ]

            # Project datapoint the same way whether they were drawn from the db or from the alert.
            if self.idempotent and self.projection:
                for i, el in enumerate(dps_combine):
                    if el in ids_dps_to_insert:
                        dps_combine[i] = self._project(el, self._projection_spec)
        else:
            dps_combine = dps

        return [dp for dp in dps if dp["id"] in ids_dps_to_insert], dps_combine

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
