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

    def calculate_epoch(self, f, e, visit_mask, counts, remove_outliers, outlier_mask=None):
        """
        Calculates the epoch of a lightcurve.

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
        # TODO: add doc
        u_lims = pd.isna(e)
        nan_mask = pd.isna(f)

        # ---------------------   remove outliers in the bins   ---------------------- #

        # if we do not want to clean outliers just set the threshold to infinity
        outlier_thresh = np.inf if not self.clean_outliers_when_binning else 100

        # set up empty masks
        outlier_mask = np.array([False] * len(f)) if outlier_mask is None else outlier_mask
        median = np.nan
        u = np.nan
        use_mask = None
        n_points = counts

        # set up dummy values for number of remaining outliers
        n_remaining_outlier = np.inf

        # ---------------------   flag upper limits   ---------------------- #
        bin_n_ulims = np.bincount(visit_mask, weights=u_lims, minlength=len(counts))
        bin_ulim_bool = (counts - bin_n_ulims) == 0
        use_mask_ul = ~u_lims | (u_lims & bin_ulim_bool[visit_mask])

        n_loops = 0

        # recalculate uncertainty and median as long as no outliers left
        while n_remaining_outlier > 0:

            # make a mask of values to use
            use_mask = ~outlier_mask & use_mask_ul & ~nan_mask
            n_points = np.bincount(visit_mask, weights=use_mask)
            zero_points_mask = n_points == 0

            # -------------------------   calculate median   ------------------------- #
            median = np.zeros_like(counts, dtype=float)
            visits_at_least_one_point = np.unique(visit_mask[~zero_points_mask[visit_mask]])
            visits_zero_points = np.unique(visit_mask[zero_points_mask[visit_mask]])
            median[visits_at_least_one_point] = np.array([
                np.median(f[(visit_mask == i) & use_mask]) for i in visits_at_least_one_point
            ])
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
                minlength=len(counts)
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
            single_exp_measurement_errors = np.sqrt(np.bincount(
                visit_mask[use_mask],
                weights=e[use_mask] ** 2,
                minlength=len(counts)
            ))
            e_meas = np.zeros_like(std, dtype=float)
            e_meas[~zero_points_mask] = single_exp_measurement_errors[n_points > 0] / n_points[n_points > 0]
            e_meas[zero_points_mask] = np.nan
            # take the maximum value of the measured single exposure errors and the standard deviation
            u = np.maximum(std, e_meas)

            # if np.any(np.isnan(u)):
            #     ids = np.where(np.isnan(u))[0]
            #     msg = ""
            #     for inan_index in ids:
            #         nanf = f[visit_mask == inan_index]
            #         nane = e[visit_mask == inan_index]
            #         msg += f"u is nan for {inan_index}th bin\n{nanf}\n{nane}\n\n"
            #     # raise ValueError(msg)

            # ---------------------   remove outliers in the bins   ---------------------- #
            remaining_outliers = (abs(median[visit_mask] - f) > outlier_thresh * u[visit_mask]) & ~outlier_mask
            outlier_mask |= remaining_outliers
            n_remaining_outlier = sum(remaining_outliers) if remove_outliers else 0
            # setting remaining_outliers to 0 will exit the while loop

            n_loops += 1

            if n_loops > 20:
                raise Exception(f"{n_loops}!")

        return median, u, bin_ulim_bool, outlier_mask, use_mask, n_points

    def bin_lightcurve(self, lightcurve):
        """
        Combine the data by visits of the satellite of one region in the sky.
        The visits typically consist of some tens of observations. The individual visits are separated by about
        six months.
        The mean flux for one visit is calculated by the weighted mean of the data.
        The error on that mean is calculated by the root-mean-squared and corrected by the t-value.
        Outliers per visit are identified if they are more than 100 times the rms away from the mean. These outliers
        are removed from the calculation of the mean and the error if self.clean_outliers_when_binning is True.

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
            bin_ulim_bools = dict()

            for lum_ext in [self.flux_key_ext, self.mag_key_ext]:
                f = lightcurve[f"{b}{lum_ext}"]
                e = lightcurve[f"{b}{lum_ext}{self.error_key_ext}"]

                mean, u, bin_ulim_bool, outlier_mask, use_mask, n_points = self.calculate_epoch(
                    f, e, visit_mask, counts, remove_outliers=True
                )
                n_outliers = np.sum(outlier_mask)

                if n_outliers > 0:
                    logger.info(f"{lum_ext}: removed {n_outliers} outliers")

                binned_data[f'{b}{self.mean_key}{lum_ext}'] = mean
                binned_data[f'{b}{lum_ext}{self.rms_key}'] = u
                binned_data[f'{b}{lum_ext}{self.upper_limit_key}'] = bin_ulim_bool
                binned_data[f'{b}{lum_ext}{self.Npoints_key}'] = n_points

                outlier_masks[lum_ext] = outlier_mask
                use_masks[lum_ext] = use_mask
                bin_ulim_bools[lum_ext] = bin_ulim_bool

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
            # find visits with no zeropoints
            n_valid_zps = np.bincount(visit_mask, weights=zp_mask)
            at_least_one_valid_zp = n_valid_zps > 0
            # calculate the median zeropoint for each visit
            zps_median = np.zeros_like(n_valid_zps, dtype=float)
            zps_median[n_valid_zps > 0] = np.array([
                np.median(zps[(visit_mask == i) & zp_mask])
                for i in np.unique(visit_mask[at_least_one_valid_zp[visit_mask]])
            ])
            # if there are only non-detections then fall back to default zeropoint
            zps_median[n_valid_zps == 0] = self.magnitude_zeropoints['Mag'][b]
            # if the visit only has upper limits then use the fall-back zeropoint
            zps_median[bin_ulim_bools[self.flux_key_ext]] = self.magnitude_zeropoints['Mag'][b]

            # ---------------   calculate flux density from instrument flux   ---------------- #
            # get the instrument flux [digital numbers], i.e. source count
            inst_fluxes_e = lightcurve[f'{b}{self.flux_key_ext}{self.error_key_ext}']

            # calculate the proportionality constant between flux density and source count
            mag_zp = self.magnitude_zeropoints['F_nu'][b].to('mJy').value
            flux_dens_const = mag_zp * 10 ** (-zps_median / 2.5)

            # calculate flux densities from instrument counts
            flux_densities = inst_fluxes * flux_dens_const[visit_mask]
            flux_densities_e = inst_fluxes_e * flux_dens_const[visit_mask]

            # bin flux densities
            mean_fd, u_fd, ul_fd, outlier_mask_fd, use_mask_fd, n_points_fd = self.calculate_epoch(
                flux_densities, flux_densities_e, visit_mask, counts,
                remove_outliers=False, outlier_mask=outlier_masks[self.flux_key_ext]
                # we do not remove outliers here because they have already been removed in the inst flux calculation
            )
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
                mck = f"{band}_coverage_of_median{lum_key}"

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
                    imc = np.sum(abs(ilc[llumkey] - imed) < ilc[errkey]) / len(ilc[llumkey])

                    metadata[difk] = imax - imin
                    metadata[rmsk] = imin_rms
                    metadata[medk] = imed
                    metadata[chi2tmk] = ichi2_to_med
                    metadata[mck] = imc

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
