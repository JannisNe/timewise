import abc, os
import pandas as pd
import numpy as np
import logging

from timewise.general import main_logger, cache_dir, plots_dir
from timewise.utils import plot_sdss_cutout, plot_panstarrs_cutout


logger = logging.getLogger(__name__)


class ParentSampleBase(abc.ABC):
    """
    Base class for parent sample.
    Any subclass must implement

    - `ParentSample.df`: A `pandas.DataFrame` consisting of minimum three columns: two columns holding the sky positions of each object in the form of right ascension and declination and one row with a unique identifier.
    - `ParentSample.default_keymap`: a dictionary, mapping the column in `ParentSample.df` to 'ra', 'dec' and 'id'

    :param base_name: determining the location of any data in the `timewise` data directory.
    """

    df = pd.DataFrame()
    default_keymap = dict()

    def __init__(self, base_name):
        # set up directories
        self.cache_dir = os.path.join(cache_dir, base_name)
        self.plots_dir = os.path.join(plots_dir, base_name)

        for d in [self.cache_dir, self.plots_dir]:
            if not os.path.isdir(d):
                os.makedirs(d)

        self.local_sample_copy = os.path.join(self.cache_dir, 'sample.csv')

    def plot_cutout(self, ind, arcsec=20, interactive=False, **kwargs):
        """
        Plot the coutout images in all filters around the position of object with index i

        :param ind: the index in the sample
        :type ind: int or list-like
        :param arcsec: the radius of the cutout
        :type arcsec: float
        :param interactive: interactive mode
        :type interactive: bool
        :param kwargs: any additional kwargs will be passed to `matplotlib.pyplot.subplots()`
        :return: figure and axes if `interactive=True`
        """
        sel = self.df.loc[np.atleast_1d(ind)]
        ra, dec = sel[self.default_keymap["ra"]], sel[self.default_keymap["dec"]]
        title = [r[self.default_keymap["id"]] for i, r in sel.iterrows()]

        fn = kwargs.pop(
            "fn",
            [os.path.join(self.plots_dir, f"{i}_{r[self.default_keymap['id']]}.pdf")
             for i, r in sel.iterrows()]
        )

        logger.debug(f"\nRA: {ra}\nDEC: {dec}\nTITLE: {title}\nFN: {fn}")
        ou = list()

        ras = np.atleast_1d(ra)
        decs = np.atleast_1d(dec)
        title = np.atleast_1d(title) if title else [None] * len(ras)
        fn = np.atleast_1d(fn) if fn else [None] * len(ras)

        for _ra, _dec, _title, _fn in zip(ras, decs, title, fn):
            ou.append(self._plot_cutout(_ra, _dec, arcsec, interactive, title=_title, fn=_fn, **kwargs))

        if len(ou) == 1:
            ou = ou[0]
        return ou

    @staticmethod
    def _plot_cutout(ra, dec, arcsec, interactive, which="sdss", **kwargs):
        print(kwargs)
        if which == "sdss":
            return plot_sdss_cutout(ra, dec, arcsec=arcsec, interactive=interactive, **kwargs)
        elif which == "panstarrs":
            return plot_panstarrs_cutout(ra, dec, arcsec=arcsec, interactive=interactive, **kwargs)
        else:
            raise ValueError(f"{which} not an implemented survey! Choose one of 'sdss' or 'panstarrs'.")

    def save_local(self):
        logger.debug(f"saving under {self.local_sample_copy}")
        self.df.to_csv(self.local_sample_copy)