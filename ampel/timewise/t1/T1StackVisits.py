#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                ampel/timewise/t1/T1CombineWISEVisits.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                24.09.2025
# Last Modified Date:  24.09.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>
from typing import Dict

import numpy as np
import pandas as pd
from numpy import typing as npt
from scipy import stats

from ampel.abstract.AbsT1ComputeUnit import AbsT1ComputeUnit
from ampel.content.DataPoint import DataPoint
from ampel.struct.UnitResult import UnitResult
from ampel.types import StockId, UBson


class T1StackVisits(AbsT1ComputeUnit):
    clean_outliers_when_stacking: bool = True

    mean_key: str = "_mean"
    median_key: str = "_median"
    rms_key: str = "_rms"
    upper_limit_key: str = "_ul"
    Npoints_key: str = "_Npoints"
    zeropoint_key_ext: str = "_zeropoint"
    flux_key_ext = "_flux"
    flux_density_key_ext = "_flux_density"
    mag_key_ext = "_mag"
    error_key_ext = "_error"

    # zero points come from https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#conv2flux
    # published in Jarret et al. (2011): https://ui.adsabs.harvard.edu/abs/2011ApJ...735..112J/abstract
    magnitude_zeropoints: Dict[str, float] = {"w1": 20.752, "w2": 19.596}
    # in Jy
    flux_zeropoints = {"w1": 309.54, "w2": 171.787}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def calculate_epochs(
        f: pd.Series,
        e: pd.Series,
        visit_mask: npt.ArrayLike,
        counts: npt.ArrayLike,
        remove_outliers: bool,
        outlier_mask: npt.ArrayLike | None = None,
    ) -> tuple[
        npt.ArrayLike,
        npt.ArrayLike,
        npt.ArrayLike,
        npt.ArrayLike,
        npt.ArrayLike,
        npt.ArrayLike,
    ]:
        """
        Calculates the visits within a raw lightcurve.

        :param f: the fluxes
        :type f: np.array
        :param e: the flux errors
        :type e: np.array
        :param visit_mask: the visit mask
        :type visit_mask: np.array
        :param counts: the counts
        :type counts: np.array
        :param remove_outliers: whether to remove outliers
        :type remove_outliers: bool
        :param outlier_mask: the outlier mask
        :type outlier_mask: np.array
        :return: the epoch
        :rtype: float
        """

        if len(f) == 0:
            return [], [], [], [], [], []

        u_lims = pd.isna(e)
        nan_mask = pd.isna(f)

        # ---------------------   remove outliers in the bins   ---------------------- #

        # if we do not want to clean outliers just set the threshold to infinity
        outlier_thresh = np.inf if not remove_outliers else 20

        # set up empty masks
        outlier_mask = (
            np.array([False] * len(f)) if outlier_mask is None else outlier_mask
        )
        median = np.nan
        u = np.nan
        use_mask = None
        n_points = counts

        # set up dummy values for number of remaining outliers
        n_remaining_outlier = np.inf

        # ---------------------   flag upper limits   ---------------------- #
        bin_n_ulims = np.bincount(visit_mask, weights=u_lims, minlength=len(counts))
        bin_ulim_bool = (counts - bin_n_ulims) == 0  # type: npt.ArrayLike
        use_mask_ul = ~u_lims | (u_lims & bin_ulim_bool[visit_mask])

        n_loops = 0

        # recalculate uncertainty and median as long as no outliers left
        while n_remaining_outlier > 0:
            # make a mask of values to use
            use_mask = ~outlier_mask & use_mask_ul & ~nan_mask
            n_points = np.bincount(visit_mask, weights=use_mask)
            zero_points_mask = n_points == 0  # type: npt.ArrayLike

            # -------------------------   calculate median   ------------------------- #
            median = np.zeros_like(counts, dtype=float)
            visits_at_least_one_point = np.unique(
                visit_mask[~zero_points_mask[visit_mask]]
            )
            visits_zero_points = np.unique(visit_mask[zero_points_mask[visit_mask]])
            median[visits_at_least_one_point] = np.array(
                [
                    np.median(f[(visit_mask == i) & use_mask])
                    for i in visits_at_least_one_point
                ]
            )
            median[visits_zero_points] = np.nan

            # median is NaN for visits with 0 detections, (i.e. detections in one band and not the other)
            # if median is NaN for other visits raise Error
            if np.any(np.isnan(median[n_points > 0])):
                nan_indices = np.where(np.isnan(median))[0]
                msg = ""
                for inan_index in nan_indices:
                    nanf = f[visit_mask == inan_index]
                    msg += f"median is nan for {inan_index}th bin\n{nanf}\n\n"
                raise ValueError(msg)

            # ---------------------   calculate uncertainty   ---------------------- #
            mean_deviation = np.bincount(
                visit_mask[use_mask],
                weights=(f[use_mask] - median[visit_mask[use_mask]]) ** 2,
                minlength=len(counts),
            )
            one_points_mask = n_points <= 1
            # calculate standard deviation
            std = np.zeros_like(counts, dtype=float)
            std[~one_points_mask] = (
                np.sqrt(mean_deviation[~one_points_mask])
                / (n_points[~one_points_mask] - 1)
                * stats.t.interval(0.68, df=n_points[~one_points_mask] - 1)[1]
                # for visits with small number of detections we have to correct according to the t distribution
            )
            std[one_points_mask] = -np.inf

            # calculate the propagated errors of the single exposure measurements
            single_exp_measurement_errors = np.sqrt(
                np.bincount(
                    visit_mask[use_mask],
                    weights=e[use_mask] ** 2,
                    minlength=len(counts),
                )
            )
            e_meas = np.zeros_like(std, dtype=float)
            e_meas[~zero_points_mask] = (
                single_exp_measurement_errors[n_points > 0] / n_points[n_points > 0]
            )
            e_meas[zero_points_mask] = np.nan
            # take the maximum value of the measured single exposure errors and the standard deviation
            u = np.maximum(std, e_meas)

            # calculate 90% confidence interval
            u70 = np.zeros_like(counts, dtype=float)
            u70[one_points_mask] = 1e-10
            visits_at_least_two_point = np.unique(
                visit_mask[~one_points_mask[visit_mask]]
            )
            u70[visits_at_least_two_point] = np.array(
                [
                    np.quantile(
                        abs(f[(visit_mask == i) & use_mask] - median[i]),
                        0.7,
                        method="interpolated_inverted_cdf",
                    )
                    for i in visits_at_least_two_point
                ]
            )

            # ---------------------   remove outliers in the bins   ---------------------- #
            remaining_outliers = (
                abs(median[visit_mask] - f) > outlier_thresh * u70[visit_mask]
            ) & ~outlier_mask
            outlier_mask |= remaining_outliers
            n_remaining_outlier = sum(remaining_outliers) if remove_outliers else 0
            # setting remaining_outliers to 0 will exit the while loop

            n_loops += 1

            if n_loops > 20:
                raise Exception(f"{n_loops}!")

        return median, u, bin_ulim_bool, outlier_mask, use_mask, n_points

    @staticmethod
    def get_visit_map(lightcurve: pd.DataFrame) -> npt.ArrayLike:
        """
        Create a map datapoint to visit

        :param lightcurve: the raw lightcurve
        :type lightcurve: pd.DataFrame
        :returns: visit map
        :rtype: npt.ArrayLike
        """
        # -------------------------   find epoch intervals   -------------------------- #
        sorted_mjds = np.sort(lightcurve.mjd)
        epoch_bounds_mask = (sorted_mjds[1:] - sorted_mjds[:-1]) > 100
        epoch_bins = np.array(
            [
                lightcurve.mjd.min() * 0.99
            ]  # this makes sure that the first datapoint gets selected
            + list(
                ((sorted_mjds[1:] + sorted_mjds[:-1]) / 2)[epoch_bounds_mask]
            )  # finding the middle between
            +
            # two visits
            [
                lightcurve.mjd.max() * 1.01
            ]  # this just makes sure that the last datapoint gets selected as well
        )

        visit_mask = np.digitize(lightcurve.mjd, epoch_bins) - 1
        return visit_mask

    def stack_visits(self, lightcurve: pd.DataFrame):
        """
        Combine the data by visits of the satellite of one region in the sky.
        The visits typically consist of some tens of observations. The individual visits are separated by about
        six months.
        The mean flux for one visit is calculated by the weighted mean of the data.
        The error on that mean is calculated by the root-mean-squared and corrected by the t-value.
        Outliers per visit are identified if they are more than 100 times the rms away from the mean. These outliers
        are removed from the calculation of the mean and the error if self.clean_outliers_when_stacking is True.

        :param lightcurve: the raw lightcurve
        :type lightcurve: pandas.DataFrame
        :return: the stacked lightcurve
        :rtype: pandas.DataFrame
        """

        # -------------------------   create visit mask   -------------------------- #
        visit_map = self.get_visit_map(lightcurve)
        counts = np.bincount(visit_map)

        stacked_data = dict()

        # -------------------------   calculate mean mjd   -------------------------- #
        stacked_data["mean_mjd"] = (
            np.bincount(visit_map, weights=lightcurve.mjd) / counts
        )

        # -------------------------   loop through bands   -------------------------- #
        for b in ["w1", "w2"]:
            # loop through magnitude and flux and save the respective datapoints

            outlier_masks = dict()
            use_masks = dict()
            bin_ulim_bools = dict()

            for lum_ext in [self.flux_key_ext, self.mag_key_ext]:
                f = lightcurve[f"{b}{lum_ext}"]
                e = lightcurve[f"{b}{lum_ext}{self.error_key_ext}"]

                # we will flag outliers based on the flux only
                remove_outliers = (
                    lum_ext == self.flux_key_ext and self.clean_outliers_when_stacking
                )
                outlier_mask = outlier_masks.get(self.flux_key_ext, None)

                mean, u, bin_ulim_bool, outlier_mask, use_mask, n_points = (
                    self.calculate_epochs(
                        f,
                        e,
                        visit_map,
                        counts,
                        remove_outliers=remove_outliers,
                        outlier_mask=outlier_mask,
                    )
                )
                n_outliers = np.sum(outlier_mask)

                if n_outliers > 0:
                    self.logger.debug(
                        f"removed {n_outliers} outliers by brightness for {b} {lum_ext}"
                    )

                stacked_data[f"{b}{self.mean_key}{lum_ext}"] = mean
                stacked_data[f"{b}{lum_ext}{self.rms_key}"] = u
                stacked_data[f"{b}{lum_ext}{self.upper_limit_key}"] = bin_ulim_bool
                stacked_data[f"{b}{lum_ext}{self.Npoints_key}"] = n_points

                outlier_masks[lum_ext] = outlier_mask
                use_masks[lum_ext] = use_mask
                bin_ulim_bools[lum_ext] = bin_ulim_bool

            # -------  calculate the zeropoints per exposure ------- #
            # this might look wrong since we use the flux mask on the magnitudes but it s right
            # for each flux measurement we need the corresponding magnitude to get the zeropoint
            mags = lightcurve[f"{b}{self.mag_key_ext}"]
            inst_fluxes = lightcurve[f"{b}{self.flux_key_ext}"]
            pos_m = inst_fluxes > 0  # select only positive fluxes, i.e. detections
            zp_mask = pos_m & use_masks[self.flux_key_ext]

            # calculate zero points
            zps = np.zeros_like(inst_fluxes)
            zps[zp_mask] = mags[zp_mask] + 2.5 * np.log10(inst_fluxes[zp_mask])
            # find visits with no zeropoints
            n_valid_zps = np.bincount(visit_map, weights=zp_mask)
            at_least_one_valid_zp = n_valid_zps > 0
            # calculate the median zeropoint for each visit
            zps_median = np.zeros_like(n_valid_zps, dtype=float)
            zps_median[n_valid_zps > 0] = np.array(
                [
                    np.median(zps[(visit_map == i) & zp_mask])
                    for i in np.unique(visit_map[at_least_one_valid_zp[visit_map]])
                ]
            )
            # if there are only non-detections then fall back to default zeropoint
            zps_median[n_valid_zps == 0] = self.magnitude_zeropoints[b]
            # if the visit only has upper limits then use the fall-back zeropoint
            zps_median[bin_ulim_bools[self.flux_key_ext]] = self.magnitude_zeropoints[b]

            # ---------------   calculate flux density from instrument flux   ---------------- #
            # get the instrument flux [digital numbers], i.e. source count
            inst_fluxes_e = lightcurve[f"{b}{self.flux_key_ext}{self.error_key_ext}"]

            # calculate the proportionality constant between flux density and source count
            mag_zp = self.flux_zeropoints[b] * 1e3  # in mJy
            flux_dens_const = mag_zp * 10 ** (-zps_median / 2.5)

            # calculate flux densities from instrument counts
            flux_densities = inst_fluxes * flux_dens_const[visit_map]
            flux_densities_e = inst_fluxes_e * flux_dens_const[visit_map]

            # bin flux densities
            mean_fd, u_fd, ul_fd, outlier_mask_fd, use_mask_fd, n_points_fd = (
                self.calculate_epochs(
                    flux_densities,
                    flux_densities_e,
                    visit_map,
                    counts,
                    remove_outliers=False,
                    outlier_mask=outlier_masks[self.flux_key_ext],
                )
            )
            stacked_data[f"{b}{self.mean_key}{self.flux_density_key_ext}"] = mean_fd
            stacked_data[f"{b}{self.flux_density_key_ext}{self.rms_key}"] = u_fd
            stacked_data[f"{b}{self.flux_density_key_ext}{self.upper_limit_key}"] = (
                ul_fd
            )
            stacked_data[f"{b}{self.flux_density_key_ext}{self.Npoints_key}"] = (
                n_points_fd
            )

        return pd.DataFrame(stacked_data)

    def compute(
        self, datapoints: list[DataPoint]
    ) -> tuple[UBson | UnitResult, StockId]:
        """
        :param datapoints: list of datapoints to combine
        :return: tuple of UBson or UnitResult and StockId
        """
        ra = []
        dec = []
        mjd = []
        stock_ids = []
        dp_ids = []
        allwise = []
        for dp in datapoints:
            ra.append(dp["body"]["ra"])
            dec.append(dp["body"]["dec"])
            mjd.append(dp["body"]["mjd"])
            stock_ids.append(np.atleast_1d(dp["stock"]))
            dp_ids.append(dp["id"])
            allwise.append(any(["allwise" in t for t in dp["tag"]]))

        lightcurve = pd.DataFrame(
            {
                "ra": ra,
                "dec": dec,
                "mjd": mjd,
                "allwise": allwise,
            },
            index=dp_ids,
        )
