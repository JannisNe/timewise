#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                timewise/ampel/timewise/alert/TimewiseAlertSupplier.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                16.09.2025
# Last Modified Date:  16.09.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>

import sys
from hashlib import blake2b
from typing import Literal, List, TYPE_CHECKING

import pandas as pd

from bson import encode
import numpy as np
from numpy.ma.core import MaskedConstant

from ampel.alert.AmpelAlert import AmpelAlert
from ampel.alert.BaseAlertSupplier import BaseAlertSupplier
from ampel.view.ReadOnlyDict import ReadOnlyDict


class TimewiseAlertSupplier(BaseAlertSupplier):
    """
    Iterable class that, for each transient name provided by the underlying alert_loader
    returns a PhotoAlert instance.
    """

    stat_pps: int = 0
    stat_uls: int = 0

    dpid: Literal["hash", "inc"] = "hash"
    #    external_directory: Optional[ str ]
    #    deserialize: None | Literal["avro", "json"]

    bands: List[str] = ["w1", "w2"]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.counter = 0 if self.dpid == "hash" else 1

    def __next__(self) -> AmpelAlert:
        """
        :returns: a dict with a structure that AlertProcessor understands
        :raises StopIteration: when alert_loader dries out.
        :raises AttributeError: if alert_loader was not set properly before this method is called
        """
        table: pd.DataFrame = self._deserialize(next(self.alert_loader))

        stock_ids = table["stock_id"].unique()
        assert len(stock_ids) == 1
        stock_id = stock_ids[0]

        # make the tables into a list of dictionaries for ampel to understand
        all_ids = b""
        pps = []

        # remove the _ep at the end of AllWISE MEP data
        columns_to_rename = [c for c in table.columns if c.endswith("_ep")]
        if len(columns_to_rename):
            rename = {
                c: (cnew := c.replace("_ep", ""))
                for c in columns_to_rename
                if cnew not in table.columns
            }
            if rename:
                # in this case only the allwise column eith the _ep extension exists
                # and we can simply rename the columns
                table.rename(columns=rename, inplace=True)

            move = {
                c: (cnew := c.replace("_ep", ""))
                for c in columns_to_rename
                if cnew in table.columns
            }
            if move:
                # In this case, the columns already exists because the neowise data is present
                # we have to insert the values form the columns with the _ep extension into the
                # respective neowise columns
                for c, nc in move.items():
                    na_mask = table[nc].isna()
                    table[nc][na_mask] = table[c][na_mask]
                table.drop(columns=[c for c in move], inplace=True)

        for i, row in table.iterrows():
            # convert table row to dict, convert data types from numpy to native python
            # Respect masked fields and convert to None
            pp = {k: None if pd.isna(v) else v for k, v in row.to_dict().items()}
            pp_hash = blake2b(encode(pp), digest_size=7).digest()
            if self.counter:
                pp["candid"] = self.counter
                self.counter += 1
            else:
                pp["candid"] = int.from_bytes(pp_hash, byteorder=sys.byteorder)

            all_ids += pp_hash
            pps.append(ReadOnlyDict(pp))

        if not pps:
            return self.__next__()

        # Update stats
        self.stat_pps += len(pps)

        return AmpelAlert(
            id=int.from_bytes(  # alert id
                blake2b(all_ids, digest_size=7).digest(), byteorder=sys.byteorder
            ),
            stock=int(stock_id),  # internal ampel id
            datapoints=tuple(pps),
        )
