import pytest
from pathlib import Path
from pydantic import TypeAdapter
from timewise.query import QueryType

DATA_DIR = Path(__file__).parent / "data" / "queries"

query_inputs = [
    (
        {
            "type": "positional_allwise_p3as_mep",
            "radius_arcsec": 6,
            "magnitudes": True,
            "fluxes": False,
        },
        "positional_allwise_p3as_mep_mag.txt",
    ),
    (
        {
            "type": "positional_allwise_p3as_mep",
            "radius_arcsec": 6,
            "magnitudes": False,
            "fluxes": True,
        },
        "positional_allwise_p3as_mep_fluxes.txt",
    ),
    (
        {
            "type": "positional_allwise_p3as_mep",
            "radius_arcsec": 6,
            "magnitudes": True,
            "fluxes": True,
        },
        "positional_allwise_p3as_mep_mag_fluxes.txt",
    ),
    (
        {
            "type": "positional_neowiser_p1bs_psd",
            "radius_arcsec": 6,
            "magnitudes": True,
            "fluxes": False,
        },
        "positional_neowiser_p1bs_psd_mag.txt",
    ),
    (
        {
            "type": "positional_neowiser_p1bs_psd",
            "radius_arcsec": 6,
            "magnitudes": False,
            "fluxes": True,
        },
        "positional_neowiser_p1bs_psd_fluxes.txt",
    ),
    (
        {
            "type": "positional_neowiser_p1bs_psd",
            "radius_arcsec": 6,
            "magnitudes": True,
            "fluxes": True,
        },
        "positional_neowiser_p1bs_psd_mag_fluxes.txt",
    ),
]


def normalize_sql(s: str) -> str:
    return " ".join(s.split())


@pytest.mark.parametrize("config_dict,ref_path", query_inputs)
def test_query_build_matches_reference(config_dict, ref_path):
    # Load reference text
    ref_file = DATA_DIR / ref_path
    expected = normalize_sql(ref_file.read_text())

    # Instantiate query from config
    q = TypeAdapter(QueryType).validate_python(config_dict)

    # Build query and compare
    built = normalize_sql(q.build())

    print(expected)
    print(built)

    assert built == expected, "query mismatch"
