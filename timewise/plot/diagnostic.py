from typing import Literal, Dict, Any, Sequence, List, cast
from functools import partial
from pathlib import Path
import logging

import pandas as pd
import numpy as np
from numpy import typing as npt
from pydantic import BaseModel
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

from timewise.plot import plot_lightcurve, plot_panstarrs_cutout, plot_sdss_cutout
from timewise.plot.lightcurve import BAND_PLOT_COLORS
from timewise.process import keys
from timewise.util.visits import get_visit_map
from timewise.config import TimewiseConfig


logger = logging.getLogger(__name__)


class DiagnosticPlotter(BaseModel):
    cutout: Literal["sdss", "panstarrs"] = "panstarrs"
    band_colors: Dict[str, str] = BAND_PLOT_COLORS
    lum_key: str = keys.FLUX_EXT

    def plot_lightcurve(
        self,
        stacked_lightcurve: pd.DataFrame | None = None,
        raw_lightcurve: pd.DataFrame | None = None,
        ax: plt.Axes | None = None,
        **kwargs,
    ):
        return plot_lightcurve(
            lum_key=self.lum_key,
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

    def make_plot(
        self,
        stacked_lightcurve: pd.DataFrame | None,
        raw_lightcurve: pd.DataFrame,
        labels: npt.ArrayLike,
        source_ra: float,
        source_dec: float,
        selected_indices: list[Any],
        highlight_radius: float | None = None,
    ) -> tuple[plt.Figure, Sequence[plt.Axes]]:
        fig, axs = plt.subplots(
            nrows=2, gridspec_kw={"height_ratios": [3, 2]}, figsize=(5, 8)
        )

        self.plot_cutout(ra=source_ra, dec=source_dec, ax=axs[0], radius_arcsec=20)

        selected_mask = raw_lightcurve.index.isin(selected_indices)
        plot_lightcurve(
            raw_lightcurve=raw_lightcurve[~selected_mask],
            lum_key=self.lum_key,
            ax=axs[-1],
            save=False,
            colors={"w1": "gray", "w2": "lightgray"},
            add_to_label=" ignored",
        )
        self.plot_lightcurve(
            stacked_lightcurve=stacked_lightcurve,
            raw_lightcurve=raw_lightcurve[selected_mask],
            ax=axs[-1],
            save=False,
        )

        # set markers for clusters
        markers_strings = list(Line2D.filled_markers) + [
            "$1$",
            "$2$",
            "$3$",
            "$4$",
            "$5$",
            "$6$",
            "$7$",
            "$8$",
            "$9$",
        ]
        markers_straight = [MarkerStyle(im) for im in markers_strings]
        rot = Affine2D().rotate_deg(180)
        markers_rotated = [MarkerStyle(im, transform=rot) for im in markers_strings]
        markers = markers_straight + markers_rotated

        # calculate ra and dec relative to center of cutout
        ra = (raw_lightcurve.ra - source_ra) * 3600
        dec = (raw_lightcurve.dec - source_dec) * 3600

        # get visit map
        visit_map = get_visit_map(raw_lightcurve.mjd)

        # for each visit plot the datapoints on the cutout
        # for each visit plot the datapoints on the cutout
        for visit in np.unique(visit_map):
            m = visit_map == visit
            label = str(visit)
            axs[0].plot(
                [],
                [],
                marker=markers[visit],
                label=label,
                mec="k",
                mew=1,
                mfc="none",
                ls="",
            )

            for im, mec, zorder in zip(
                [selected_mask, ~selected_mask], ["k", "none"], [1, 0]
            ):
                mask = m & im

                for i_label in np.unique(labels):
                    label_mask = labels == i_label
                    final_mask = mask & label_mask
                    datapoints_label = raw_lightcurve[final_mask]
                    color = f"C{i_label}" if i_label != -1 else "grey"

                    if ("sigra" in datapoints_label.columns) and (
                        "sigdec" in datapoints_label.columns
                    ):
                        has_sig = (
                            ~datapoints_label.sigra.isna()
                            & ~datapoints_label.sigdec.isna()
                        )
                        _ra = ra[final_mask]
                        _dec = dec[final_mask]

                        axs[0].errorbar(
                            _ra[has_sig],
                            _dec[has_sig],
                            xerr=datapoints_label.sigra[has_sig] / 3600,
                            yerr=datapoints_label.sigdec[has_sig] / 3600,
                            marker=markers[visit],
                            ls="",
                            color=color,
                            zorder=zorder,
                            ms=10,
                            mec=mec,
                            mew=0.1,
                        )
                        axs[0].scatter(
                            _ra[~has_sig],
                            _dec[~has_sig],
                            marker=markers[visit],
                            color=color,
                            zorder=zorder,
                            edgecolors=mec,
                            linewidths=0.1,
                        )
                    else:
                        axs[0].scatter(
                            ra[final_mask],
                            dec[final_mask],
                            marker=markers[visit],
                            color=color,
                            zorder=zorder,
                            edgecolors=mec,
                            linewidths=0.1,
                        )

        if highlight_radius:
            circle = plt.Circle(
                (0, 0),
                highlight_radius,
                color="g",
                fill=False,
                ls="-",
                lw=3,
                zorder=0,
            )
            axs[0].add_artist(circle)

        # formatting
        title = axs[0].get_title()
        axs[-1].set_ylabel("Apparent Vega Magnitude")
        axs[-1].grid(ls=":", alpha=0.5)
        axs[0].set_title("")
        axs[0].legend(
            ncol=5,
            bbox_to_anchor=(0, 1, 1, 0),
            loc="lower left",
            mode="expand",
            title=title,
        )
        axs[0].set_aspect(1, adjustable="box")

        return fig, axs


def make_plot(
    config_path: Path,
    cutout: Literal["sdss", "panstarrs"],
    indices: List[int],
    output_directory: Path,
):
    cfg = TimewiseConfig.from_yaml(config_path)
    ampel_interface = cfg.build_ampel_interface()
    input_data = pd.read_csv(cfg.download.input_csv).set_index(
        ampel_interface.orig_id_key
    )
    plotter = DiagnosticPlotter(cutout=cutout)
    for index in indices:
        stacked_lightcurve = ampel_interface.extract_stacked_lightcurve(stock_id=index)
        raw_lightcurve = ampel_interface.extract_datapoints(stock_id=index)
        selected_dp_ids = ampel_interface.extract_selected_datapoint_ids(stock_id=index)
        labels = [0] * len(raw_lightcurve)
        source = input_data.loc[index]
        ra: float = cast(float, source.ra)
        dec: float = cast(float, source.dec)

        fig, axs = plotter.make_plot(
            stacked_lightcurve=stacked_lightcurve,
            raw_lightcurve=raw_lightcurve,
            labels=labels,
            source_ra=ra,
            source_dec=dec,
            selected_indices=selected_dp_ids,
        )
        fn = output_directory / f"{index}.pdf"
        logger.info(f"Saving plot to {fn}")
        fig.savefig(fn)
        plt.close()
