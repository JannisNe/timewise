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

    def calculate_epoch(self, f, e, visit_mask, counts, remove_outliers):
        # TODO: add doc
        u_lims = pd.isna(e)

        # ---------------------   remove outliers in the bins   ---------------------- #

        # if we do not want to clean outliers just set the threshold to infinity
        outlier_thresh = np.inf if not self.clean_outliers_when_binning else 100

        # set up empty masks
        outlier_mask = np.array([False] * len(f))
        mean = np.nan
        u = np.nan
        use_mask = None

        # set up dummy values for number of remaining outliers
        n_remaining_outlier = np.inf

        # ---------------------   flag upper limits   ---------------------- #
        bin_n_ulims = np.bincount(visit_mask, weights=u_lims, minlength=len(counts))
        bin_ulim_bool = (counts - bin_n_ulims) == 0
        use_mask_ul = ~u_lims | (u_lims & bin_ulim_bool[visit_mask])

        # recalculate uncertainty and median as long as no outliers left
        while (n_remaining_outlier > 0) and remove_outliers:

            # make a mask of values to use
            use_mask = ~outlier_mask & use_mask_ul

            # -------------------------   calculate mean   ------------------------- #
            sums = np.bincount(visit_mask[use_mask], weights=f[use_mask], minlength=len(counts))
            mean = sums / counts
            mean_deviation = np.bincount(
                visit_mask[use_mask],
                weights=(f[use_mask] - mean[visit_mask[use_mask]]) ** 2,
                minlength=len(counts)
            )

            # ---------------------   calculate uncertainty   ---------------------- #
            std = np.sqrt(mean_deviation) / (counts - 1)
            ecomb = np.sqrt(np.bincount(
                visit_mask[use_mask],
                weights=e[use_mask] ** 2,
                minlength=len(counts)
            )) / counts
            t_value = stats.t.interval(0.68, df=counts - 1)
            u = np.maximum(std, ecomb) * t_value

            # ---------------------   remove outliers in the bins   ---------------------- #
            remaining_outliers = abs(mean[visit_mask] - f) > outlier_thresh * u[visit_mask]
            outlier_mask |= remaining_outliers
            n_remaining_outlier = sum(remaining_outliers)

        return mean, u, bin_ulim_bool, outlier_mask, use_mask

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
        epoch_bins = np.array(
            [lightcurve.mjd.min() * 0.99] +  # this makes sure that the first datapoint gets selected
            list(((sorted_mjds[1:] + sorted_mjds[:-1]) / 2)[epoch_bounds_mask]) +  # finding the middle between
                                                                                   # two visits
            [lightcurve.mjd.max() * 1.01]    # this just makes sure that the last datapoint gets selected as well
        )

        # -------------------------   create visit mask   -------------------------- #
        visit_mask = np.digitize(lightcurve.mjd, epoch_bins) - 1
        counts = np.bincount(visit_mask)

        binned_data = dict()

        # -------------------------   calculate mean mjd   -------------------------- #
        binned_data["mean_mjd"] = np.bincount(visit_mask, weights=lightcurve.mjd) / counts

        # -------------------------   loop through bands   -------------------------- #
        for b in self.bands:
            # loop through magnitude and flux and save the respective datapoints

            outlier_masks = dict()
            use_masks = dict()

            for lum_ext in [self.flux_key_ext, self.mag_key_ext]:
                f = lightcurve[f"{b}{lum_ext}"]
                e = lightcurve[f"{b}{lum_ext}{self.error_key_ext}"]

                mean, u, bin_ulim_bool, outlier_mask, use_mask = self.calculate_epoch(
                    f, e, visit_mask, counts, remove_outliers=True
                )
                n_outliers = np.bincount(visit_mask, weights=outlier_mask)
                n_points = np.bincount(visit_mask, weights=use_mask)

                if n_outliers > 0:
                    logger.info(f"{lum_ext}: removed {n_outliers} outliers")

                binned_data[f'{b}{self.mean_key}{lum_ext}'] = mean
                binned_data[f'{b}{lum_ext}{self.rms_key}'] = u
                binned_data[f'{b}{lum_ext}{self.upper_limit_key}'] = bin_ulim_bool
                binned_data[f'{b}{lum_ext}{self.Npoints_key}'] = n_points

                outlier_masks[lum_ext] = outlier_mask
                use_masks[lum_ext] = use_mask

            # -------  calculate the zeropoints per exposure ------- #
            # this might look wrong since we use the flux mask on the magnitudes but it s right
            # for each flux measurement we need the corresponding magnitude to get the zeropoint
            mags = lightcurve[f'{b}{self.mag_key_ext}']
            inst_fluxes = lightcurve[f'{b}{self.flux_key_ext}']
            pos_m = inst_fluxes > 0  # select only positive fluxes, i.e. detections
            zp_mask = pos_m & use_masks[self.flux_key_ext]

            # calculate zero points
            zps = np.zeros_like(inst_fluxes)
            zps[zp_mask] = mags[zp_mask] + 2.5 * np.log10(inst_fluxes[zp_mask])
            zps_sum = np.bincount(visit_mask, weights=zps)

            zps_mean = zps_sum / counts
            # if there are only non-detections then fall back to default zeropoint
            zps_mean[zps_mean == 0] = self.magnitude_zeropoints['Mag'][b]

            # ---------------   calculate flux density from instrument flux   ---------------- #
            # get the instrument flux [digital numbers], i.e. source count
            inst_fluxes_e = lightcurve[f'{b}{self.flux_key_ext}{self.error_key_ext}']

            # calculate the proportionality constant between flux density and source count
            mag_zp = self.magnitude_zeropoints['F_nu'][b].to('mJy').value
            flux_dens_const = mag_zp * 10 ** (-zps_mean / 2.5)

            # calculate flux densities from instrument counts
            flux_densities = inst_fluxes * flux_dens_const
            flux_densities_e = inst_fluxes_e * flux_dens_const

            # bin flux densities
            mean_fd, u_fd, ul_fd, outlier_mask_fd, use_mask_fd = self.calculate_epoch(
                flux_densities, flux_densities_e, visit_mask, counts,
                remove_outliers=False  # we do not remove outliers here because they have already been removed in
                                       # the inst flux calculation
            )
            n_points_fd = np.bincount(visit_mask, weights=use_mask)
            binned_data[f'{b}{self.mean_key}{self.flux_density_key_ext}'] = mean_fd
            binned_data[f'{b}{self.flux_density_key_ext}{self.rms_key}'] = u_fd
            binned_data[f'{b}{self.flux_density_key_ext}{self.upper_limit_key}'] = ul_fd
            binned_data[f'{b}{self.flux_density_key_ext}{self.Npoints_key}'] = n_points_fd

        return pd.DataFrame(binned_data)

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
