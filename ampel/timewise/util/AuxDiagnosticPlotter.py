from typing import Literal, Dict, Any

import pandas as pd
from numpy import typing as npt

from ampel.plot.create import create_plot_record
from ampel.base.AmpelBaseModel import AmpelBaseModel
from ampel.model.PlotProperties import PlotProperties
from ampel.content.NewSVGRecord import NewSVGRecord

from timewise.plot.diagnostic import DiagnosticPlotter


class AuxDiagnosticPlotter(AmpelBaseModel):
    plot_properties: PlotProperties
    cutout: Literal["sdss", "panstarrs"] = DiagnosticPlotter.model_fields[
        "cutout"
    ].default
    band_colors: Dict[str, str] = DiagnosticPlotter.model_fields["band_colors"].default
    lum_key: str = DiagnosticPlotter.model_fields["lum_key"].default

    def __init__(self, /, **data: Any):
        super().__init__(**data)
        self._plotter = DiagnosticPlotter.model_validate(
            {k: self.__getattribute__(k) for k in ["cutout", "band_colors", "lum_key"]}
        )

    def make_plot(
        self,
        lightcurve: pd.DataFrame,
        labels: npt.ArrayLike,
        source_ra: float,
        source_dec: float,
        selected_indices: list[Any],
        highlight_radius: float | None = None,
    ) -> NewSVGRecord:
        fig, axs = self._plotter.make_plot(
            raw_lightcurve=lightcurve,
            labels=labels,
            source_ra=source_ra,
            source_dec=source_dec,
            selected_indices=selected_indices,
            highlight_radius=highlight_radius,
        )
        return create_plot_record(fig, self.plot_properties)
