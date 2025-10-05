import numpy as np
import pandas as pd
import numpy.typing as npt


def get_visit_map(
    mjd: npt.NDArray[np.float64] | pd.Series[float],
) -> npt.NDArray[np.int64]:
    """
    Create a map datapoint to visit

    :param mjd: the MJDs of the observations
    :type mjd: npt.NDArray[float]
    :returns: visit map
    :rtype: npt.ArrayLike
    """
    # -------------------------   find epoch intervals   -------------------------- #
    sorted_mjds = np.sort(mjd)
    epoch_bounds_mask = (sorted_mjds[1:] - sorted_mjds[:-1]) > 100
    epoch_bins = np.array(
        [mjd.min() * 0.99]  # this makes sure that the first datapoint gets selected
        + list(
            ((sorted_mjds[1:] + sorted_mjds[:-1]) / 2)[epoch_bounds_mask]
        )  # finding the middle between
        +
        # two visits
        [
            mjd.max() * 1.01
        ]  # this just makes sure that the last datapoint gets selected as well
    )

    visit_mask = np.digitize(mjd, epoch_bins) - 1
    return visit_mask
