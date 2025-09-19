#!/usr/bin/env python
# File:                Ampel-ZTF/ampel/ztf/ingest/ZiCompilerOptions.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                14.05.2021
# Last Modified Date:  14.05.2021
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from typing import Any

from ampel.model.ingest.CompilerOptions import CompilerOptions


class ZiCompilerOptions(CompilerOptions):
    stock: dict[str, Any] = {"id_mapper": "ZTFIdMapper", "tag": "ZTF"}
    t0: dict[str, Any] = {"tag": "ZTF"}
    t1: dict[str, Any] = {"tag": "ZTF"}
    state_t2: dict[str, Any] = {"tag": "ZTF"}
    point_t2: dict[str, Any] = {"tag": "ZTF"}
    stock_t2: dict[str, Any] = {"tag": "ZTF"}
