from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from timewise.process import keys


BAND_PLOT_COLORS = {"w1": "r", "w2": "b"}


def plot_lightcurve(
    lum_key: str,
    stacked_lightcurve: pd.DataFrame | None = None,
    raw_lightcurve: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
    colors: Dict[str, str] | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    assert (stacked_lightcurve is not None) or (raw_lightcurve is not None)

    if not colors:
        colors = BAND_PLOT_COLORS

    if not ax:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = plt.gcf()

    for b in ["w1", "w2"]:
        try:
            if isinstance(stacked_lightcurve, pd.DataFrame):
                ul_mask = np.array(
                    stacked_lightcurve[f"{b}{lum_key}{keys.UPPER_LIMIT}"]
                ).astype(bool)
                ax.errorbar(
                    stacked_lightcurve[keys.MEAN + "_mjd"][~ul_mask],
                    stacked_lightcurve[f"{b}{keys.MEAN}{lum_key}"][~ul_mask],
                    yerr=stacked_lightcurve[f"{b}{lum_key}{keys.RMS}"][~ul_mask],
                    label=b,
                    ls="",
                    marker="s",
                    c=colors[b],
                    markersize=4,
                    markeredgecolor="k",
                    ecolor="k",
                    capsize=2,
                )
                ax.scatter(
                    stacked_lightcurve[keys.MEAN + "_mjd"][ul_mask],
                    stacked_lightcurve[f"{b}{keys.MEAN}{lum_key}"][ul_mask],
                    marker="v",
                    c=colors[b],
                    alpha=0.7,
                    s=2,
                )

            if isinstance(raw_lightcurve, pd.DataFrame):
                m = ~raw_lightcurve[f"{b}{lum_key}"].isna()
                ul_mask = raw_lightcurve[f"{b}{keys.ERROR_EXT}{lum_key}"].isna()

                tot_m = m & ~ul_mask
                if np.any(tot_m):
                    ax.errorbar(
                        raw_lightcurve.mjd[tot_m],
                        raw_lightcurve[f"{b}{lum_key}"][tot_m],
                        yerr=raw_lightcurve[f"{b}{keys.ERROR_EXT}{lum_key}"][tot_m],
                        label=f"{b} unbinned",
                        ls="",
                        marker="o",
                        c=colors[b],
                        markersize=4,
                        alpha=0.3,
                    )

                single_ul_m = m & ul_mask
                if np.any(single_ul_m):
                    label = f"{b} unbinned upper limits" if not np.any(tot_m) else ""
                    ax.scatter(
                        raw_lightcurve.mjd[single_ul_m],
                        raw_lightcurve[f"{b}{lum_key}"][single_ul_m],
                        marker="d",
                        c=colors[b],
                        alpha=0.3,
                        s=1,
                        label=label,
                    )

        except KeyError as e:
            raise KeyError(f"Could not find brightness key {e}!")

    if lum_key == keys.MAG_EXT:
        ylim = ax.get_ylim()
        ax.set_ylim(max(ylim), min(ylim))

    ax.set_xlabel("MJD")
    ax.set_ylabel(lum_key)
    ax.legend()

    return fig, ax
