import logging
from typing import cast, Dict, Any, Literal

from scipy import stats
from scipy.special import gamma
import numpy as np
from numpy import typing as npt
import pandas as pd

from ..util.visits import get_visit_map
from timewise.process import keys


logger = logging.getLogger(__name__)


# zero points come from https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#conv2flux
# published in Jarret et al. (2011): https://ui.adsabs.harvard.edu/abs/2011ApJ...735..112J/abstract
MAGNITUDE_ZEROPOINTS: Dict[str, float] = {"w1": 20.752, "w2": 19.596}
# in Jy
FLUX_ZEROPOINTS = {"w1": 309.54, "w2": 171.787}


def std_debias_factor(n: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    # Cureton The American Statistician 22, 22–22 (1968). https://www.jstor.org/stable/2681876
    return gamma((n - 1) / 2) / np.sqrt(2 / (n - 1)) / gamma(n / 2)


def t_distribution_correction(n: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    return stats.t.interval(0.68, df=n - 1)[1]


def no_correction(n: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    return np.ones_like(n, dtype=float)


CORRECTION_FUNCTIONS = {
    "debias": std_debias_factor,
    "tdist": t_distribution_correction,
    "none": no_correction,
}


def calculate_epochs(
    f: pd.Series,
    e: pd.Series,
    visit_mask: npt.NDArray[np.int64],
    counts: npt.NDArray[np.int64],
    remove_outliers: bool,
    outlier_threshold: float,
    outlier_quantile: float,
    outlier_mask: npt.NDArray[np.bool_] | None = None,
    mean_name: Literal["mean", "median"] = "median",
    std_name: Literal["std", "sdom", "sdom-1"] = "sdom-1",
    correction_name: Literal["tdist", "debias", "none"] = "tdist",
    calculate_pvalues: bool = False,
    use_single_exposure_errors: bool = False
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
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
    :param outlier_threshold: threshold to identify outliers
    :type outlier_threshold: float
    :param outlier_quantile: quantile that the outlier_threshold is multiplied with
    :type outlier_quantile: float
    :param outlier_mask: the outlier mask
    :type outlier_mask: np.array, optional
    :param mean_name: name of the numpy function to calculate the mean, defaults to "mean"
    :type mean_name: str, optional
    :param std_name: name of the function to calculate the stacked error, defaults to "std"
    :type std_name: str, optional
    :param correction_name: name of the correction function to apply to the standard deviation, defaults to "debias"
    :type correction_name: str, optional
    :param calculate_pvalues: if true, calculate ks-test p-values to check consistency with normal distribution per visit
    :type calculate_pvalues: bool
    :param use_single_exposure_errors:
        if true, use the maximum of the RMS and the combined single exposure measurements as the final uncertainty
    :type use_single_exposure_errors: bool
    :return: the epoch
    :rtype: float
    """

    if len(f) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    u_lims = pd.isna(e)
    nan_mask = pd.isna(f)

    # ---------------------   remove outliers in the bins   ---------------------- #

    # if we do not want to clean outliers just set the threshold to infinity
    _outlier_threshold = np.inf if not remove_outliers else outlier_threshold

    # set up empty masks
    outlier_mask = cast(
        npt.NDArray[np.bool_],
        (np.array([False] * len(f)) if outlier_mask is None else outlier_mask),
    )
    median = np.full_like(counts, np.nan, dtype=float)
    u = np.full_like(counts, np.nan, dtype=float)
    use_mask = np.full_like(counts, False, dtype=bool)
    n_points = counts

    # set up dummy values for number of remaining outliers
    n_remaining_outlier = np.inf

    # ---------------------   flag upper limits   ---------------------- #
    bin_n_ulims: npt.NDArray[np.int64] = np.bincount(
        visit_mask, weights=u_lims, minlength=len(counts)
    )
    bin_ulim_bool = cast(npt.NDArray[np.bool_], (counts - bin_n_ulims) == 0)
    use_mask_ul = ~u_lims | (u_lims & bin_ulim_bool[visit_mask])

    n_loops = 0

    # recalculate uncertainty and median as long as no outliers left
    mean_function = np.mean if mean_name == "mean" else np.median

    # select function to use to correct standard deviation with
    bias_correction_function = CORRECTION_FUNCTIONS[correction_name]

    one_points_mask = None
    visits_at_least_two_point = None

    while n_remaining_outlier > 0:
        # make a mask of values to use
        use_mask = ~outlier_mask & use_mask_ul & ~nan_mask  # type: ignore[operator]
        n_points = np.bincount(visit_mask, weights=use_mask)
        zero_points_mask = cast(npt.NDArray[np.bool_], n_points == 0)

        # -------------------------   calculate median   ------------------------- #
        median = np.zeros_like(counts, dtype=float)
        visits_at_least_one_point = np.unique(visit_mask[~zero_points_mask[visit_mask]])
        visits_zero_points = np.unique(visit_mask[zero_points_mask[visit_mask]])
        median[visits_at_least_one_point] = np.array(
            [
                mean_function(f[(visit_mask == i) & use_mask])
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
        extra_factor = (
            1
            if std_name == "std"
            else 1 / n_points[~one_points_mask]
            if std_name == "sdom"
            else 1 / (n_points[~one_points_mask] - 1)
        )
        std[~one_points_mask] = (
            np.sqrt(
                mean_deviation[~one_points_mask]
                / (n_points[~one_points_mask] - 1)
                * extra_factor
            )
            * bias_correction_function(n_points[~one_points_mask])
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

        if use_single_exposure_errors:
            # take the maximum value of the measured single exposure errors and the standard deviation
            u = np.maximum(std, e_meas)
        else:
            # take the rms as the measurement uncertainty except if there is only one single exposure
            u = std
            u[one_points_mask] = e_meas[one_points_mask]

        # Estimate the spread of the flux.
        # To be robust against outliers, do that with quantiles instead of std
        qs = np.zeros_like(counts, dtype=float)
        qs[one_points_mask] = 1e-10
        visits_at_least_two_point = np.unique(visit_mask[~one_points_mask[visit_mask]])
        qs[visits_at_least_two_point] = np.array(
            [
                np.quantile(
                    abs(f[(visit_mask == i) & use_mask] - median[i]),
                    outlier_quantile,
                    method="interpolated_inverted_cdf",
                )
                for i in visits_at_least_two_point
            ]
        )

        # ---------------------   remove outliers in the bins   ---------------------- #
        remaining_outliers = (
            abs(median[visit_mask] - f) > _outlier_threshold * qs[visit_mask]
        ) & ~outlier_mask
        outlier_mask |= remaining_outliers
        n_remaining_outlier = sum(remaining_outliers) if remove_outliers else 0
        # setting remaining_outliers to 0 will exit the while loop

        n_loops += 1

        if n_loops > 20:
            raise Exception(f"{n_loops}!")

    # --------------------   calculate std for crosscheck   -------------------- #
    if calculate_pvalues:
        npstd = np.zeros_like(counts, dtype=float)
        npstd[~one_points_mask] = np.array(
            [np.std(f[(visit_mask == i) & use_mask]) for i in visits_at_least_two_point]
        )
        npstd[one_points_mask] = np.nan

    # ----------------   calculate compatibility with gaussian   ---------------- #
        pvalues = np.ones_like(counts, dtype=float)
        pvalues[~one_points_mask] = np.array(
            [
                stats.kstest(
                    f[(visit_mask == i) & use_mask], stats.norm(median[i], npstd[i]).cdf
                ).pvalue
                for i in visits_at_least_two_point
            ]
        )
    else:
        pvalues = np.full_like(counts, -999)

    return median, u, bin_ulim_bool, outlier_mask, use_mask, n_points, pvalues


def stack_visits(
    lightcurve: pd.DataFrame,
    outlier_threshold: float,
    outlier_quantile: float,
    clean_outliers: bool = True,
    mean_name: Literal["mean", "median"] = "median",
    std_name: Literal["std", "sdom", "sdom-1"] = "sdom-1",
    correction_name: Literal["tdist", "debias", "none"] = "tdist",
    calculate_pvalues: bool = False,
    use_single_exposure_errors: bool = False
):
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
    :param outlier_threshold: threshold to identify outliers
    :type outlier_threshold: float
    :param outlier_quantile: quantile that the outlier_threshold is multiplied with
    :type outlier_quantile: float
    :param clean_outliers:
        if True, remove outliers that are outlier_threshold x outlier_quantile far away from the mean per visit,
        default is True
    :type clean_outliers: bool
    :param mean_name: name of the numpy function to calculate the mean, defaults to "mean"
    :type mean_name: str, optional
    :param std_name: name of the function to calculate the stacked error, defaults to "std"
    :type std_name: str, optional
    :param correction_name: name of the correction function to apply to the standard deviation, defaults to "debias"
    :type correction_name: str, optional
    :param calculate_pvalues: if true, calculate ks-test p-values to check consistency with normal distribution per visit
    :type calculate_pvalues: bool
    :param use_single_exposure_errors:
        if true, use the maximum of the RMS and the combined single exposure measurements as the final uncertainty
    :type use_single_exposure_errors: bool
    :return: the stacked lightcurve
    :rtype: pandas.DataFrame
    """

    # -------------------------   create visit mask   -------------------------- #
    visit_map = get_visit_map(lightcurve.mjd)
    counts = np.bincount(visit_map)

    stacked_data: Dict[str, Any] = dict()

    # -------------------------   calculate mean mjd   -------------------------- #
    stacked_data["mean_mjd"] = np.bincount(visit_map, weights=lightcurve.mjd) / counts

    # -------------------------   loop through bands   -------------------------- #
    for b in ["w1", "w2"]:
        # loop through magnitude and flux and save the respective datapoints

        outlier_masks: Dict[str, Any] = dict()
        use_masks = dict()
        bin_ulim_bools = dict()

        for lum_ext in [keys.FLUX_EXT, keys.MAG_EXT]:
            f = lightcurve[f"{b}{lum_ext}"]
            e = lightcurve[f"{b}{keys.ERROR_EXT}{lum_ext}"]

            # we will flag outliers based on the flux only
            remove_outliers = lum_ext == keys.FLUX_EXT and clean_outliers
            outlier_mask = outlier_masks.get(keys.FLUX_EXT, None)

            mean, u, bin_ulim_bool, outlier_mask, use_mask, n_points, p_values = calculate_epochs(
                f,
                e,
                visit_map,
                counts,
                remove_outliers=remove_outliers,
                outlier_mask=outlier_mask,
                outlier_quantile=outlier_quantile,
                outlier_threshold=outlier_threshold,
                mean_name=mean_name,
                std_name=std_name,
                correction_name=correction_name,
                calculate_pvalues=calculate_pvalues,
                use_single_exposure_errors=use_single_exposure_errors
            )
            n_outliers = np.sum(outlier_mask)

            if n_outliers > 0:
                logger.debug(
                    f"removed {n_outliers} outliers by brightness for {b} {lum_ext}"
                )

            stacked_data[f"{b}{keys.MEAN}{lum_ext}"] = mean
            stacked_data[f"{b}{lum_ext}{keys.RMS}"] = u
            stacked_data[f"{b}{lum_ext}{keys.UPPER_LIMIT}"] = bin_ulim_bool
            stacked_data[f"{b}{lum_ext}{keys.NPOINTS}"] = n_points
            stacked_data[f"{b}{lum_ext}{keys.KSTEST_NORM_EXT}"] = p_values

            outlier_masks[lum_ext] = outlier_mask
            use_masks[lum_ext] = use_mask
            bin_ulim_bools[lum_ext] = bin_ulim_bool

        # -------  calculate the zeropoints per exposure ------- #
        # this might look wrong since we use the flux mask on the magnitudes but it s right
        # for each flux measurement we need the corresponding magnitude to get the zeropoint
        mags = lightcurve[f"{b}{keys.MAG_EXT}"]
        inst_fluxes = lightcurve[f"{b}{keys.FLUX_EXT}"]
        pos_m = inst_fluxes > 0  # select only positive fluxes, i.e. detections
        zp_mask = pos_m & use_masks[keys.FLUX_EXT]

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
        zps_median[n_valid_zps == 0] = MAGNITUDE_ZEROPOINTS[b]
        # if the visit only has upper limits then use the fall-back zeropoint
        zps_median[bin_ulim_bools[keys.FLUX_EXT]] = MAGNITUDE_ZEROPOINTS[b]

        # ---------------   calculate flux density from instrument flux   ---------------- #
        # get the instrument flux [digital numbers], i.e. source count
        inst_fluxes_e = lightcurve[f"{b}{keys.ERROR_EXT}{keys.FLUX_EXT}"]

        # calculate the proportionality constant between flux density and source count
        mag_zp = FLUX_ZEROPOINTS[b] * 1e3  # in mJy
        flux_dens_const = mag_zp * 10 ** (-zps_median / 2.5)

        # calculate flux densities from instrument counts
        flux_densities = inst_fluxes * flux_dens_const[visit_map]
        flux_densities_e = inst_fluxes_e * flux_dens_const[visit_map]

        # bin flux densities
        mean_fd, u_fd, ul_fd, outlier_mask_fd, use_mask_fd, n_points_fd, p_values_fd = (
            calculate_epochs(
                flux_densities,
                flux_densities_e,
                visit_map,
                counts,
                remove_outliers=False,
                outlier_mask=outlier_masks[keys.FLUX_EXT],
                outlier_threshold=outlier_threshold,
                outlier_quantile=outlier_quantile,
                mean_name=mean_name,
                std_name=std_name,
                correction_name=correction_name,
                calculate_pvalues=calculate_pvalues,
                use_single_exposure_errors=use_single_exposure_errors
            )
        )
        stacked_data[f"{b}{keys.MEAN}{keys.FLUX_DENSITY_EXT}"] = mean_fd
        stacked_data[f"{b}{keys.FLUX_DENSITY_EXT}{keys.RMS}"] = u_fd
        stacked_data[f"{b}{keys.FLUX_DENSITY_EXT}{keys.UPPER_LIMIT}"] = ul_fd
        stacked_data[f"{b}{keys.FLUX_DENSITY_EXT}{keys.NPOINTS}"] = n_points_fd
        stacked_data[f"{b}{keys.FLUX_DENSITY_EXT}{keys.KSTEST_NORM_EXT}"] = p_values_fd

    return pd.DataFrame(stacked_data)
