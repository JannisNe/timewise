#!/usr/bin/env python
# File:                Ampel-ZTF/ampel/ztf/ingest/ZiCompilerOptions.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                14.05.2021
# Last Modified Date:  14.05.2021
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from typing import Any

from ampel.model.ingest.CompilerOptions import CompilerOptions


class TiCompilerOptions(CompilerOptions):
    stock: dict[str, Any] = {"tag": "TIMEWISE"}
    t0: dict[str, Any] = {"tag": "TIMEWISE"}
    t1: dict[str, Any] = {"tag": "TIMEWISE"}
    state_t2: dict[str, Any] = {"tag": "TIMEWISE"}
    point_t2: dict[str, Any] = {"tag": "TIMEWISE"}
    stock_t2: dict[str, Any] = {"tag": "TIMEWISE"}
