from typing import Literal, Dict
from functools import partial

import pandas as pd
import matplotlib.pyplot as plt

from ampel.base.AmpelBaseModel import AmpelBaseModel

from timewise.plot import plot_lightcurve, plot_panstarrs_cutout, plot_sdss_cutout
from timewise.plot.lightcurve import BAND_PLOT_COLORS


class DiagnosticPlotter(AmpelBaseModel):
    cutout: Literal["sdss", "panstarrs"] = "panstarrs"
    band_colors: Dict[str, str] = BAND_PLOT_COLORS

    def plot_lightcurve(
        self,
        lum_key: str,
        stacked_lightcurve: pd.DataFrame | None = None,
        raw_lightcurve: pd.DataFrame | None = None,
        ax: plt.Axes | None = None,
        **kwargs,
    ):
        return plot_lightcurve(
            lum_key=lum_key,
            stacked_lightcurve=stacked_lightcurve,
            raw_lightcurve=raw_lightcurve,
            ax=ax,
            colors=self.band_colors,
            **kwargs,
        )

    def plot_cutout(self, ra: float, dec: float, radius_arcsec: float, ax: plt.Axes):
        if self.cutout == "sdss":
            plot_cutout = plot_sdss_cutout
        elif self.cutout == "panstarrs":
            plot_cutout = partial(plot_panstarrs_cutout, plot_color_image=True)
        else:
            raise NotImplementedError  # should never happen
        return plot_cutout(ra=ra, dec=dec, arcsec=radius_arcsec, ax=ax)
