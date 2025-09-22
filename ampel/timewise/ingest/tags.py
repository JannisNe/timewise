#!/usr/bin/env python
# File:                timewise/ampel/timewise/ingest/tags.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                19.09.2025
# Last Modified Date:  19.09.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>


# tags is used by TiT0PhotoPointShaper and ZiT0UpperLimitShaper
# key: filter id
tags: dict[str, list[str]] = {
    "allwise_p3as_mep": ["WISE", "TIMEWISE", "allwise_p3as_mep"],
    "neowiser_p1bs_psd": ["WISE", "TIMEWISE", "neowiser_p1bs_psd"],
}
