# This script creates a file documenting the vaule of the t distribution intervals
# for scipy <= 1.17.0. These changed, probably due to changes in the integration in `quantile`
# (see https://docs.scipy.org/doc/scipy/release/1.17.0-notes.html#scipy-stats-improvements)

import scipy
import logging
from packaging.version import Version
from scipy import stats
import numpy as np
from pathlib import Path

OUTPUT_FILE = Path(__file__).parent / "scipy_tdist_quantiles.npy"
logger = logging.getLogger(__name__)


def create_scipy_correction_file():
    n = np.arange(0, 100)
    vals = stats.t.interval(0.68, n)[1]
    out_arr = np.array([n, vals]).T
    logger.info(f"Writing {OUTPUT_FILE}")
    np.save(OUTPUT_FILE, out_arr)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    assert Version(scipy.__version__) <= Version("1.17.0"), (
        "This script needs scipy versions <= 1.17.0!"
    )
    create_scipy_correction_file()
