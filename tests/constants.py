from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
INPUT_CSV_PATH = DATA_DIR / "test_sample.csv"
AMPEL_CONFIG_PATH = Path(__file__).parent.parent / "ampel_config.yml"


V0_KEYMAP = {
    "AllWISE Multiepoch Photometry Table": {
        "flux": {
            "w1flux_ep": f"W1{flux_key_ext}",
            "w1sigflux_ep": f"W1{flux_key_ext}{error_key_ext}",
            "w2flux_ep": f"W2{flux_key_ext}",
            "w2sigflux_ep": f"W2{flux_key_ext}{error_key_ext}",
        },
        "mag": {
            "w1mpro_ep": f"W1{mag_key_ext}",
            "w1sigmpro_ep": f"W1{mag_key_ext}{error_key_ext}",
            "w2mpro_ep": f"W2{mag_key_ext}",
            "w2sigmpro_ep": f"W2{mag_key_ext}{error_key_ext}",
        },
    },
    "NEOWISE-R Single Exposure (L1b) Source Table": {
        "flux": {
            "w1flux": f"W1{flux_key_ext}",
            "w1sigflux": f"W1{flux_key_ext}{error_key_ext}",
            "w2flux": f"W2{flux_key_ext}",
            "w2sigflux": f"W2{flux_key_ext}{error_key_ext}",
        },
        "mag": {
            "w1mpro": f"W1{mag_key_ext}",
            "w1sigmpro": f"W1{mag_key_ext}{error_key_ext}",
            "w2mpro": f"W2{mag_key_ext}",
            "w2sigmpro": f"W2{mag_key_ext}{error_key_ext}",
        },
    },
}
