from pathlib import Path
import numpy as np
from timewise.process import keys


DATA_DIR = Path(__file__).parent / "data"
INPUT_CSV_PATH = DATA_DIR / "test_sample.csv"
AMPEL_CONFIG_PATH = Path(__file__).parent.parent / "ampel_config.yml"


V0_KEYS = dict(
    mean_key="_mean",
    median_key="_median",
    rms_key="_rms",
    upper_limit_key="_ul",
    Npoints_key="_Npoints",
    zeropoint_key_ext="_zeropoint",
    mag="_mag",
    flux="_flux",
    flux_density="_flux_density",
    error="_error",
)

V0_KEYMAP = []
for i in range(1, 3):
    for nl, ol in zip([keys.MAG_EXT, keys.FLUX_EXT], [V0_KEYS["mag"], V0_KEYS["flux"]]):
        V0_KEYMAP.extend(
            [
                (f"W{i}{ol}", f"w{i}{nl}"),
                (f"W{i}{ol}{V0_KEYS['error']}", f"w{i}{keys.ERROR_EXT}{nl}"),
            ]
        )

    for nl, ol in zip(
        [keys.MAG_EXT, keys.FLUX_EXT, keys.FLUX_DENSITY_EXT],
        [V0_KEYS["mag"], V0_KEYS["flux"], V0_KEYS["flux_density"]],
    ):
        V0_KEYMAP.extend(
            [
                (f"W{i}{V0_KEYS['mean_key']}{ol}", f"w{i}{nl}"),
                (f"W{i}{ol}{V0_KEYS['rms_key']}", f"w{i}{keys.RMS}{nl}"),
                (
                    f"W{i}{ol}{V0_KEYS['upper_limit_key']}",
                    f"w{i}{keys.UPPER_LIMIT}{nl}",
                ),
                (f"W{i}{ol}{V0_KEYS['Npoints_key']}", f"w{i}{keys.NPOINTS}{nl}"),
            ]
        )
V0_KEYMAP = np.array(V0_KEYMAP)
