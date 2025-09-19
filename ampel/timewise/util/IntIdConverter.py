#!/usr/bin/env python
# File:                Ampel-ZTF/ampel/ztf/util/ZTFIdMapper.py
# License:             BSD-3-Clause
# Author:              valery brinnel <firstname.lastname@gmail.com>
# Date:                07.06.2018
# Last Modified Date:  12.02.2021
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

from collections.abc import Iterable
from typing import cast, overload

from ampel.abstract.AbsIdMapper import AbsIdMapper
from ampel.types import StockId, StrictIterable

# Optimization variables
alphabet = "abcdefghijklmnopqrstuvwxyz"
ab = {alphabet[i]: i for i in range(26)}
rg = (6, 5, 4, 3, 2, 1, 0)
powers = tuple(26**i for i in (6, 5, 4, 3, 2, 1, 0))
enc_ztf_years = {str(i + 17): i for i in range(16)}
dec_ztf_years = {i: str(i + 17) for i in range(16)}


class ZTFIdMapper(AbsIdMapper):
    @overload
    @classmethod
    def to_ampel_id(cls, ztf_id: str) -> int: ...

    @overload
    @classmethod
    def to_ampel_id(cls, ztf_id: StrictIterable[str]) -> list[int]: ...

    @classmethod
    def to_ampel_id(cls, ztf_id: str | StrictIterable[str]) -> int | list[int]:
        """
        :returns: ampel id (positive integer).

        ====== First 4 bits encode the ZTF year (until max 2032) =====

        In []: to_ampel_id('ZTF17aaaaaaa')
        Out[]: 0

        In []: to_ampel_id('ZTF18aaaaaaa')
        Out[]: 1

        In []: to_ampel_id('ZTF19aaaaaaa')
        Out[]: 2

        ====== Bits onwards encode the ZTF name converted from base 26 into base 10 =====

        In []: to_ampel_id('ZTF17aaaaaaa')
        Out[]: 0

        In []: to_ampel_id('ZTF17aaaaaab')
        Out[]: 16

        In []: to_ampel_id('ZTF17aaaaaac')
        Out[]: 32

        ====== Biggest numerical value is < 2**37 =====

        Out[]: In []: to_ampel_id('ZTF32zzzzzzz')
        Out[]: 128508962815

        ========================================================================
        This encoding allows to save most of ZTF transients (up until ~akzzzzzz)
        with a signed int32. Note: MongoDB imposes signed integers and chooses
        automatically the right _id type (int32/int64/...) per document
        =====================================================================

        In []: to_ampel_id('ZTF20akzzzzzz') < 2**31
        Out[]: True

        In []: to_ampel_id('ZTF20alzzzzzz') < 2**31
        Out[]: False

        =====================================================================

        Note: slightly slower than the legacy method

        -> Legacy
        %timeit to_ampel_id("ZTF19abcdfef")
        949 ns ± 9.08 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

        -> This method
        %timeit to_ampel_id('ZTF19abcdfef')
        1.51 µs ± 35.5 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

        but encodes ZTF ids with 37 bits instead of 52 bits

        In []: 2**36 < to_ampel_id('ZTF32zzzzzzz') < 2**37
        Out[]: True

        from ampel.ztf.legacy_utils import to_ampel_id as to_legacy_ampel_id
        In []:  2**51 < to_legacy_ampel_id("ZTF32zzzzzz") < 2**52
        Out []: True
        """

        if isinstance(ztf_id, str):
            num = 0
            s2 = ztf_id[5:]
            for i in rg:
                num += ab[s2[i]] * powers[i]
            return (num << 4) + enc_ztf_years[ztf_id[3:5]]
        return [cast(int, cls.to_ampel_id(name)) for name in ztf_id]

    @overload
    @classmethod
    def to_ext_id(cls, ampel_id: StockId) -> str: ...

    @overload
    @classmethod
    def to_ext_id(cls, ampel_id: StrictIterable[StockId]) -> list[str]: ...

    @classmethod
    def to_ext_id(cls, ampel_id: StockId | StrictIterable[StockId]) -> str | list[str]:
        """
        %timeit to_ext_id(274878346346)
        1.54 µs ± 77.9 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
        """
        # Handle sequences
        if isinstance(ampel_id, Iterable) and not isinstance(ampel_id, str):
            return [cast(str, cls.to_ext_id(l)) for l in ampel_id]
        if isinstance(ampel_id, int):
            # int('00001111', 2) bitmask equals 15
            year = dec_ztf_years[ampel_id & 15]

            # Shift base10 encoded value 4 bits to the right
            ampel_id = ampel_id >> 4

            # Convert back to base26
            l = ["a", "a", "a", "a", "a", "a", "a"]
            for i in rg:
                l[i] = alphabet[ampel_id % 26]
                ampel_id //= 26
                if not ampel_id:
                    break

            return f"ZTF{year}{''.join(l)}"
        raise TypeError(
            f"Ampel ids for ZTF transients should be ints (got {type(ampel_id)} {ampel_id})"
        )


# backward compatibility shortcuts
to_ampel_id = ZTFIdMapper.to_ampel_id
to_ztf_id = ZTFIdMapper.to_ext_id
