#!/usr/bin/env python
# File:                timewise/ampel/timewise/ingest/tags.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                19.09.2025
# Last Modified Date:  19.09.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>


# tags is used by TiT0PhotoPointShaper and ZiT0UpperLimitShaper
# key: filter id
tags: dict[int, list[str]] = {
    1: ["WISE", "TIMEWISE", "WISE_W1"],
    2: ["WISE", "TIMEWISE", "WISE_W2"],
}
