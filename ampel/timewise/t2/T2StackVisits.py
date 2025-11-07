#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                ampel/timewise/t2/T2StackVisits.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                24.09.2025
# Last Modified Date:  24.09.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>
from typing import Literal
from scipy import stats

from ampel.abstract.AbsLightCurveT2Unit import AbsLightCurveT2Unit
from ampel.struct.UnitResult import UnitResult
from ampel.types import UBson
from ampel.view.LightCurve import LightCurve

from ampel.timewise.util.pdutil import datapoints_to_dataframe

from timewise.process import keys
from timewise.process.stacking import stack_visits


SIGMA = float(stats.chi2.cdf(1, 1))


class T2StackVisits(AbsLightCurveT2Unit):
    clean_outliers: bool = True

    # spread quantile of each visit to estimate, defaults to
    # ~68% which is the equivalent of 1 sigma for a normal distribution
    outlier_quantile: float = SIGMA

    # threshold above which to exclude outliers
    outlier_threshold: float = 5

    # methods to calculate mean and std
    mean_name: Literal["mean", "median"] = "median"
    std_name: Literal["std", "sdom"] = "sdom"

    def process(self, light_curve: LightCurve) -> UBson | UnitResult:
        columns = [
            "ra",
            "dec",
            "mjd",
        ]
        for i in range(1, 3):
            for key in [keys.MAG_EXT, keys.FLUX_EXT]:
                columns.extend([f"w{i}{key}", f"w{i}{keys.ERROR_EXT}{key}"])

        photopoints = light_curve.get_photopoints()
        if photopoints is None:
            return {}
        data, _ = datapoints_to_dataframe(photopoints, columns=columns)
        if len(data) == 0:
            return {}
        return stack_visits(
            data,
            outlier_threshold=self.outlier_threshold,
            outlier_quantile=self.outlier_quantile,
            clean_outliers=self.clean_outliers,
            mean_name=self.mean_name,
            std_name=self.std_name
        ).to_dict(orient="records")
