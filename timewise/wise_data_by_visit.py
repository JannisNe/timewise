import tqdm
import pandas as pd
import numpy as np
import logging
from scipy import stats

from timewise.wise_data_base import WISEDataBase
from timewise.utils import get_excess_variance

logger = logging.getLogger(__name__)


class WiseDataByVisit(WISEDataBase):
    """
    WISEData class to bin lightcurve by visit
    """

    mean_key = '_mean'
    median_key = '_median'
    rms_key = '_rms'
    upper_limit_key = '_ul'
    Npoints_key = '_Npoints'
    zeropoint_key_ext = '_zeropoint'

    flux_error_factor = {
        "W1": 1.6,
        "W2": 1.2
    }

    def __init__(
            self,
            base_name,
            parent_sample_class,
            min_sep_arcsec,
            n_chunks,
            clean_outliers_when_binning=True,
            multiply_flux_error=True
    ):
        # TODO: add doc
        super().__init__(base_name, parent_sample_class, min_sep_arcsec, n_chunks)
        self.clean_outliers_when_binning = clean_outliers_when_binning
        self.multiply_flux_error = multiply_flux_error

    def bin_lightcurve(self, lightcurve):
        """
        Combine the data by visits of the satellite of one region in the sky.
        The visits typically consist of some tens of observations. The individual visits are separated by about
        six months.
        The mean flux for one visit is calculated by the weighted mean of the data.
        The error on that mean is calculated by the root-mean-squared.
        # TODO: add doc about clean when binning and error factor

        :param lightcurve: the unbinned lightcurve
        :type lightcurve: pandas.DataFrame
        :return: the binned lightcurve
        :rtype: pandas.DataFrame
        """

        # -------------------------   find epoch intervals   -------------------------- #
        sorted_mjds = np.sort(lightcurve.mjd)
        epoch_bounds_mask = (sorted_mjds[1:] - sorted_mjds[:-1]) > 100
        epoch_bounds = np.array(
            [lightcurve.mjd.min()] +
            list(sorted_mjds[1:][epoch_bounds_mask]) +
            [lightcurve.mjd.max() * 1.01]  # this just makes sure that the last datapoint gets selected as well
        )
        epoch_intervals = np.array([epoch_bounds[:-1], epoch_bounds[1:]]).T

        # -------------------------   loop through epoch intervals   -------------------------- #
        binned_lc = pd.DataFrame()
        for ei in epoch_intervals:
            r = dict()
            epoch_mask = (lightcurve.mjd >= ei[0]) & (lightcurve.mjd < ei[1])
            r['mean_mjd'] = np.median(lightcurve.mjd[epoch_mask])

            epoch = dict()
        # -------------------------   loop through bands   -------------------------- #
            for b in self.bands:
                # loop through magnitude and flux and save the respective datapoints
                for lum_ext in [self.flux_key_ext, self.mag_key_ext]:
                    f = lightcurve[f"{b}{lum_ext}"][epoch_mask]
                    e = lightcurve[f"{b}{lum_ext}{self.error_key_ext}"][epoch_mask]
                    ulims = pd.isna(e)

                    epoch[f"{b}{lum_ext}"] = f
                    epoch[f"{b}{lum_ext}{self.error_key_ext}"] = e
                    epoch[f"{b}{lum_ext}{self.upper_limit_key}"] = ulims

                # -------  calculate the zeropoints per exposure ------- #
                mags = epoch[f'{b}{self.mag_key_ext}']
                inst_fluxes = epoch[f'{b}{self.flux_key_ext}']
                pos_m = inst_fluxes > 0  # select only positive fluxes, i.e. detections

                if sum(pos_m) == 0:
                    # if there are only non-detections then fall back to default zeropoint
                    zp_med = self.magnitude_zeropoints['Mag'][b]
                else:
                    # calculate zeropoints and save median
                    zps = mags[pos_m] + 2.5 * np.log10(inst_fluxes[pos_m])
                    zp_med = np.median(zps)

                r[f'{b}{self.median_key}{self.zeropoint_key_ext}'] = zp_med

        # ---------------------   calculate flux density from instrument flux   ----------------------- #
                # get the instrument flux [digital numbers], i.e. source count
                fl = epoch[f'{b}{self.flux_key_ext}']
                fl_err = epoch[f'{b}{self.flux_key_ext}{self.error_key_ext}']
                fl_ul = epoch[f"{b}{self.flux_key_ext}{self.upper_limit_key}"]

                # calculate the proportionality constant between flux density and source count
                mag_zp = self.magnitude_zeropoints['F_nu'][b].to('mJy').value
                flux_dens_const = mag_zp * 10 ** (-zp_med / 2.5)

                # save values in epoch dictionary
                epoch[f"{b}{self.flux_density_key_ext}"] = fl * flux_dens_const
                epoch[f"{b}{self.flux_density_key_ext}{self.error_key_ext}"] = fl_err * flux_dens_const
                epoch[f"{b}{self.flux_density_key_ext}{self.upper_limit_key}"] = fl_ul

        # ---------------------   loop through different brightness units   ---------------------- #
                for lum_ext in [self.flux_key_ext, self.mag_key_ext, self.flux_density_key_ext]:

                    if ('flux' in lum_ext) and self.multiply_flux_error:
                        error_factor = self.flux_error_factor[b]
                    else:
                        error_factor = 1

                    f = epoch[f"{b}{lum_ext}"]
                    e = epoch[f"{b}{lum_ext}{self.error_key_ext}"] * error_factor
                    ulims = epoch[f"{b}{lum_ext}{self.upper_limit_key}"]
                    ul = np.all(pd.isna(e))
                    nans = f.isna()

                    if ul:
                        f = f[~nans]
                        e = e[~nans]

                    else:
                        f = f[~ulims & ~nans]
                        e = e[~ulims & ~nans]

        # ---------------------   if no data is there then enter nans   ---------------------- #

                    if len(f) == 0:
                        r[f"{b}{lum_ext}_outlier_indices"] = np.nan
                        r[f'{b}{self.mean_key}{lum_ext}'] = np.nan
                        r[f'{b}{lum_ext}{self.rms_key}'] = np.nan
                        r[f'{b}{lum_ext}{self.upper_limit_key}'] = np.nan
                        r[f'{b}{lum_ext}{self.Npoints_key}'] = np.nan
                        continue

        # ---------------------   remove outliers in the bins   ---------------------- #

                    # if we do not want to clean outliers just set the threshold to infinity
                    outlier_thresh = np.inf if not self.clean_outliers_when_binning else 100

                    # set up empty masks
                    remaining_outlier_mask = np.array([False] * len(f))
                    outlier_mask = np.copy(remaining_outlier_mask)

                    # set up dummy values for number of remaining and total outliers
                    N_remaining_outlier = 1
                    N_outlier = 0

                    # recalculate uncertainty and median as long as no outliers left
                    while N_remaining_outlier > 0:
                        f = f[~remaining_outlier_mask]
                        e = e[~remaining_outlier_mask]
                        mean = np.median(f)
                        t_value = stats.t.interval(0.68, df=len(f) - 1)[1]
                        rms = np.sqrt(sum((f - mean) ** 2)) / (len(f) - 1) * t_value
                        u_mes = 0 if ul else np.sqrt(sum(e[~outlier_mask] ** 2)) / len(e[~outlier_mask])
                        u = max(rms, u_mes)

                        remaining_outlier_mask = abs(mean - f) > outlier_thresh * u
                        outlier_mask = outlier_mask | remaining_outlier_mask
                        N_remaining_outlier = sum(remaining_outlier_mask)
                        N_outlier += N_remaining_outlier

                    if N_outlier > 0:
                        logger.debug(f"{b}{lum_ext}, MJD {ei}: removed {N_outlier}")
                        r[f"{b}{lum_ext}_outlier_indices"] = [list(outlier_mask.index[outlier_mask])]
                    else:
                        r[f"{b}{lum_ext}_outlier_indices"] = np.nan

        # ---------------------   assemble final result   ---------------------- #

                    r[f'{b}{self.mean_key}{lum_ext}'] = mean
                    r[f'{b}{lum_ext}{self.rms_key}'] = u
                    r[f'{b}{lum_ext}{self.upper_limit_key}'] = bool(ul)
                    r[f'{b}{lum_ext}{self.Npoints_key}'] = len(f)

            binned_lc = pd.concat([binned_lc, pd.DataFrame(r, index=[0])], ignore_index=True)

        return binned_lc

    def calculate_metadata_single(self, lc):
        """
        Calculates some metadata, describing the variability of the lightcurves.

        - `max_dif`: maximum difference in magnitude between any two datapoints
        - `min_rms`: the minimum errorbar of all datapoints
        - `N_datapoints`: The number of datapoints
        - `max_deltat`: the maximum time difference between any two datapoints
        - `mean_weighted_ppb`: the weighted average brightness where the weights are the points per bin

        :param lc: the lightcurves
        :type lc: dict
        :return: the metadata
        :rtype: dict
        """

        metadata = dict()

        for band in self.bands:
            for lum_key in [self.mag_key_ext, self.flux_key_ext, self.flux_density_key_ext]:
                llumkey = f"{band}{self.mean_key}{lum_key}"
                errkey = f"{band}{lum_key}{self.rms_key}"
                ul_key = f'{band}{lum_key}{self.upper_limit_key}'
                ppb_key = f'{band}{lum_key}{self.Npoints_key}'

                difk = f"{band}_max_dif{lum_key}"
                rmsk = f"{band}_min_rms{lum_key}"
                Nk = f"{band}_N_datapoints{lum_key}"
                dtk = f"{band}_max_deltat{lum_key}"
                medk = f"{band}_median{lum_key}"
                chi2tmk = f"{band}_chi2_to_med{lum_key}"
                mean_weighted_ppb_key = f"{band}_mean_weighted_ppb{lum_key}"
                excess_variance_key = f"{band}_excess_variance_{lum_key}"
                excess_variance_err_key = f"{band}_excess_variance_err_{lum_key}"

                ilc = lc[~np.array(lc[ul_key]).astype(bool)] if ul_key in lc else dict()
                metadata[Nk] = len(ilc)

                if len(ilc) > 0:
                    metadata[mean_weighted_ppb_key] = np.average(ilc[llumkey], weights=ilc[ppb_key])
                    metadata[excess_variance_key], metadata[excess_variance_err_key] = get_excess_variance(
                        np.array(ilc[llumkey]),
                        np.array(ilc[errkey]),
                        np.array(metadata[mean_weighted_ppb_key])
                    )

                    imin = ilc[llumkey].min()
                    imax = ilc[llumkey].max()
                    imin_rms_ind = ilc[errkey].argmin()
                    imin_rms = ilc[errkey].iloc[imin_rms_ind]

                    imed = np.median(ilc[llumkey])
                    ichi2_to_med = sum(((ilc[llumkey] - imed) / ilc[errkey]) ** 2)

                    metadata[difk] = imax - imin
                    metadata[rmsk] = imin_rms
                    metadata[medk] = imed
                    metadata[chi2tmk] = ichi2_to_med

                    if len(ilc) == 1:
                        metadata[dtk] = 0
                    else:
                        mjds = np.array(ilc.mean_mjd).astype(float)
                        dt = mjds[1:] - mjds[:-1]
                        metadata[dtk] = max(dt)

                else:
                    for k in [
                        difk,
                        dtk,
                        mean_weighted_ppb_key,
                        excess_variance_key,
                        rmsk,
                        medk,
                        chi2tmk

                    ]:
                        metadata[k] = np.nan

        return metadata
