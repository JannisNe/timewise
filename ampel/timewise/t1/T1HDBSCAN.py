#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                ampel/timewise/t1/T1HDBSCAN.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                24.09.2025
# Last Modified Date:  24.09.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>
from typing import Iterable, Sequence

import numpy as np
from numpy import typing as npt
from ampel.base.AuxUnitRegister import AuxUnitRegister
from astropy.coordinates.angle_utilities import angular_separation, position_angle
from sklearn.cluster import HDBSCAN
from pymongo import MongoClient

from ampel.content.DataPoint import DataPoint
from ampel.struct.T1CombineResult import T1CombineResult
from ampel.types import DataPointId
from ampel.abstract.AbsT1CombineUnit import AbsT1CombineUnit
from ampel.model.PlotProperties import PlotProperties

from ampel.model.UnitModel import UnitModel

from ampel.timewise.util.pdutil import datapoints_to_dataframe
from ampel.timewise.util.DiagnosticPlotter import DiagnosticPlotter


class T1HDBSCAN(AbsT1CombineUnit, DiagnosticPlotter):
    input_mongo_db_name: str
    original_id_key: str
    whitelist_region_arcsec: float = 1
    cluster_distance_arcsec: float = 0.5

    plot: bool = False
    plot_properties: PlotProperties | None = None
    plotter: UnitModel = {"unit": "DiagnosticPlotter"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._col = MongoClient()[self.input_mongo_db_name]["input"]
        self._plotter = AuxUnitRegister.new_unit(
            model=self.plotter, sub_type=DiagnosticPlotter
        )

    def combine(
        self, datapoints: Iterable[DataPoint]
    ) -> Sequence[DataPointId] | T1CombineResult:
        lightcurve, stock_ids = datapoints_to_dataframe(
            datapoints, ["ra", "dec", "mjd"], check_tables=["allwise_p3as_mep"]
        )

        # make sure that the is one stock id that fits all dps
        # this is a redundant check, the muxer should take care of it
        unique_stocks = np.unique(np.array(stock_ids).flatten())
        stock_in_all_dps = [
            all([s in sids for sids in stock_ids]) for s in unique_stocks
        ]
        # make sure only one stock is in all datapoints
        assert sum(stock_in_all_dps) == 1
        stock_id = unique_stocks[stock_in_all_dps][0].item()
        self.logger.debug(f"stock: {stock_id}")

        # query the database that holds the parent sample
        d = self._col.find_one({self.original_id_key: stock_id})
        source_ra = d["ra"]
        source_dec = d["dec"]

        lc_ra_rad = np.deg2rad(lightcurve.ra)
        lc_dec_rad = np.deg2rad(lightcurve.dec)
        source_ra_rad = np.deg2rad(source_ra)
        source_dec_rad = np.deg2rad(source_dec)

        # calculate separation and position angle
        _angular_separation = angular_separation(
            source_ra_rad, source_dec_rad, lc_ra_rad, lc_dec_rad
        )
        _position_angle = position_angle(
            source_ra_rad, source_dec_rad, lc_ra_rad, lc_dec_rad
        )

        # The AllWISE multiframe pipeline detects sources on the deep coadded atlas images and then measures the sources
        # for all available single-exposure images in all bands simultaneously, while the NEOWISE magnitudes are
        # obtained by PSF fit to individual exposures directly. Effect: all allwise data points that belong to the same
        # object have the same position. We take only the closest one and treat it as one datapoint in the clustering.
        allwise_time_mask = lightcurve.allwise_p3as_mep
        if any(allwise_time_mask):
            allwise_sep_min = np.min(_angular_separation[allwise_time_mask])
            closest_allwise_mask = (
                _angular_separation == allwise_sep_min
            ) & allwise_time_mask
            closest_allwise_mask_first_entry = (
                ~closest_allwise_mask.duplicated() & closest_allwise_mask
            )

            # the data we want to use is then the selected AllWISE datapoint and the NEOWISE-R data
            data_mask = closest_allwise_mask_first_entry | ~allwise_time_mask
        else:
            closest_allwise_mask_first_entry = closest_allwise_mask = None
            data_mask = np.ones_like(_angular_separation, dtype=bool)

        # no matter which cluster they belong to, we want to keep all datapoints within 1 arcsec
        one_arcsec_mask = _angular_separation < np.radians(
            self.whitelist_region_arcsec / 3600
        )
        selected_indices = set(lightcurve.index[data_mask & one_arcsec_mask])

        # if there are more than one datapoints, we use a clustering algorithm to potentially find a cluster with
        # its center within 1 arcsec
        labels = []
        if data_mask.sum() > 1:
            # instead of the polar coordinates separation and position angle we use cartesian coordinates because the
            # clustering algorithm works better with them
            cartesian_full = np.array(
                [
                    _angular_separation * np.cos(_position_angle),
                    _angular_separation * np.sin(_position_angle),
                ]
            ).T
            cartesian = cartesian_full[data_mask]

            # we are now ready to do the clustering
            cluster_res = HDBSCAN(
                store_centers="centroid",
                min_cluster_size=max(min(20, len(cartesian)), 2),
                allow_single_cluster=True,
                cluster_selection_epsilon=np.radians(
                    self.cluster_distance_arcsec / 3600
                ),
            ).fit(cartesian)
            centroids = cluster_res.__getattribute__("centroids_")  # type: npt.ArrayLike
            labels = cluster_res.__getattribute__("labels_")  # type: npt.ArrayLike

            # we select the closest cluster within 1 arcsec
            cluster_separations = np.sqrt(np.sum(centroids**2, axis=1))
            self.logger.debug(f"Found {len(cluster_separations)} clusters")

            # if there is no cluster or no cluster within 1 arcsec,
            # only the datapoints within 1 arcsec are selected as we did above
            if len(cluster_separations) == 0:
                self.logger.debug(
                    "No cluster found. Selecting all noise datapoints within 1 arcsec."
                )
            elif min(cluster_separations) > np.radians(
                self.whitelist_region_arcsec / 3600
            ):
                self.logger.debug(f"Closest cluster is at {cluster_separations} arcsec")

            # if there is a cluster within 1 arcsec, we select all datapoints belonging to that cluster
            # in addition to the datapoints within 1 arcsec
            else:
                closest_label = cluster_separations.argmin()
                selected_cluster_mask = labels == closest_label

                # now we have to trace back the selected datapoints to the original lightcurve
                selected_indices |= set(
                    lightcurve.index[data_mask][selected_cluster_mask]
                )
                self.logger.debug(f"Selected {len(selected_indices)} datapoints")

        # if the closest allwise source is selected, we also select all other detections belonging to that
        # source in the allwise period
        if (
            closest_allwise_mask_first_entry is not None
            and lightcurve.index[closest_allwise_mask_first_entry][0]
            in selected_indices
        ):
            closest_allwise_mask_not_first = (
                closest_allwise_mask & ~closest_allwise_mask_first_entry
            )
            closest_allwise_indices_not_first = lightcurve.index[
                closest_allwise_mask_not_first
            ]
            self.logger.debug(
                f"Adding remaining {len(closest_allwise_indices_not_first)} from AllWISE period"
            )
            selected_indices |= set(closest_allwise_indices_not_first)

        selected_indices = list(selected_indices)
        res = T1CombineResult(dps=selected_indices)

        if self.plot:
            all_labels = np.array([-1] * len(lightcurve))
            all_labels[data_mask] = labels
            svg_rec = self._plotter.make_plot(
                lightcurve, all_labels, source_ra, source_dec, selected_indices
            )
            res.add_meta("plot", svg_rec)

        return res


