import tqdm
import pandas as pd
import numpy as np

from timewise.general import main_logger
from timewise.wise_data_base import WISEDataBase


logger = main_logger.getChild(__name__)


class WiseDataByVisit(WISEDataBase):
    """
    WISEData class to bin lightcurve by visit
    """

    mean_key = '_mean'
    rms_key = '_rms'
    upper_limit_key = '_ul'
    Npoints_key = '_Npoints'

    def bin_lightcurve(self, lightcurve):
        """
        Combine the data by visits of the satellite of one region in the sky.
        The visits typically consist of some tens of observations. The individual visits are separated by about
        six months.
        The mean flux for one visit is calculated by the weighted mean of the data.
        The rror on that mean is calculated bu the root-mean-squared.

        :param lightcurve: the unbnned lightcurve
        :type lightcurve: pandas.DataFrame
        :return: the binned lightcurve
        :rtype: pandas.DataFrame
        """

        # bin lightcurves in time intervals where observations are closer than 100 days together
        sorted_mjds = np.sort(lightcurve.mjd)
        epoch_bounds_mask = (sorted_mjds[1:] - sorted_mjds[:-1]) > 100
        epoch_bounds = np.array(
            [lightcurve.mjd.min()] +
            list(sorted_mjds[1:][epoch_bounds_mask]) +
            [lightcurve.mjd.max() * 1.01]  # this just makes sure that the last datapoint gets selected as well
        )
        epoch_intervals = np.array([epoch_bounds[:-1], epoch_bounds[1:]]).T

        binned_lc = pd.DataFrame()
        for ei in epoch_intervals:
            r = dict()
            epoch_mask = (lightcurve.mjd >= ei[0]) & (lightcurve.mjd < ei[1])
            r['mean_mjd'] = np.median(lightcurve.mjd[epoch_mask])

            for b in self.bands:
                for lum_ext in [self.flux_key_ext, self.mag_key_ext, self.flux_density_key_ext]:
                    try:
                        f = lightcurve[f"{b}{lum_ext}"][epoch_mask]
                        e = lightcurve[f"{b}{lum_ext}{self.error_key_ext}"][epoch_mask]
                        ulims = pd.isna(e)
                        ul = np.all(pd.isna(e))

                        if ul:
                            mean = np.mean(f)
                            u_mes = 0
                        else:
                            f = f[~ulims]
                            e = e[~ulims]
                            w = e / sum(e)
                            mean = np.average(f, weights=w)
                            u_mes = np.sqrt(sum(e ** 2 / len(e)))

                        u_rms = np.sqrt(sum((f - mean) ** 2) / len(f))
                        r[f'{b}{self.mean_key}{lum_ext}'] = mean
                        r[f'{b}{lum_ext}{self.rms_key}'] = max(u_rms, u_mes)
                        r[f'{b}{lum_ext}{self.upper_limit_key}'] = bool(ul)
                        r[f'{b}{lum_ext}{self.Npoints_key}'] = len(f)
                    except KeyError:
                        pass

            binned_lc = binned_lc.append(r, ignore_index=True)

        return binned_lc

    def calculate_metadata_single(self, lcs):
        """
        Calculates some metadata, describing the variability of the lightcurves.

        - `max_dif`: maximum difference in magnitude between any two datapoints
        - `min_rms`: the minimum errorbar of all datapoints
        - `N_datapoints`: The number of datapoints
        - `max_deltat`: the maximum time difference between any two datapoints
        - `mean_weighted_ppb`: the weighted average brightness where the weights are the points per bin

        :param lcs: the lightcurves
        :type lcs: dict
        :return: the metadata
        :rtype: dict
        """
        metadata = dict()

        for ID, lc_dict in tqdm.tqdm(lcs.items(), desc='calculating metadata', total=len(lcs)):
            imetadata = dict()
            lc = pd.DataFrame.from_dict(lc_dict)
            for band in self.bands:
                for lum_key in [self.mag_key_ext, self.flux_key_ext]:
                    llumkey = f"{band}{self.mean_key}{lum_key}"
                    errkey = f"{band}{lum_key}{self.rms_key}"
                    ul_key = f'{band}{lum_key}{self.upper_limit_key}'
                    ppb_key = f'{band}{lum_key}{self.Npoints_key}'

                    difk = f"{band}_max_dif{lum_key}"
                    rmsk = f"{band}_min_rms{lum_key}"
                    Nk = f"{band}_N_datapoints{lum_key}"
                    dtk = f"{band}_max_deltat{lum_key}"
                    mean_weighted_ppb_key = f"{band}_mean_weighted_ppb{lum_key}"

                    try:
                        ilc = lc[~np.array(lc[ul_key]).astype(bool)]
                        imetadata[Nk] = len(ilc)

                        if len(ilc) > 0:
                            imetadata[mean_weighted_ppb_key] = np.average(ilc[llumkey], weights=ilc[ppb_key])

                            imin = ilc[llumkey].min()
                            imax = ilc[llumkey].max()
                            imin_rms_ind = ilc[errkey].argmin()
                            imin_rms = ilc[errkey].iloc[imin_rms_ind]

                            if lum_key == self.mag_key_ext:
                                imetadata[difk] = imax - imin
                                imetadata[rmsk] = imin_rms
                            else:
                                imetadata[difk] = imax / imin
                                imetadata[rmsk] = imin_rms / ilc[llumkey].iloc[imin_rms_ind]

                            if len(ilc) == 1:
                                imetadata[dtk] = 0
                            else:
                                mjds = np.array(ilc.mean_mjd).astype(float)
                                dt = mjds[1:] - mjds[:-1]
                                imetadata[dtk] = max(dt)

                        else:
                            imetadata[difk] = np.nan
                            imetadata[dtk] = np.nan
                    except KeyError as e:
                        pass

            metadata[ID] = imetadata
        return metadata