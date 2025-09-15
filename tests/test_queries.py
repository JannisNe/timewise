import pytest
from pathlib import Path
from pydantic import TypeAdapter
from timewise.query import QueryType

DATA_DIR = Path(__file__).parent / "data" / "queries"

query_inputs = [
    (
        {
            "type": "positional",
            "radius_arcsec": 6,
            "columns": [
                "ra",
                "dec",
                "mjd",
                "cntr_mf",
                "w1mpro_ep",
                "w1sigmpro_ep",
                "w2mpro_ep",
                "w2sigmpro_ep",
            ],
            "table": {"name": "allwise_p3as_mep"},
        },
        "positional_allwise_p3as_mep_mag.txt",
    ),
    (
        {
            "type": "positional",
            "radius_arcsec": 6,
            "columns": [
                "ra",
                "dec",
                "mjd",
                "cntr_mf",
                "w1flux_ep",
                "w1sigflux_ep",
                "w2flux_ep",
                "w2sigflux_ep",
            ],
            "table": {"name": "allwise_p3as_mep"},
        },
        "positional_allwise_p3as_mep_fluxes.txt",
    ),
    (
        {
            "type": "positional",
            "radius_arcsec": 6,
            "columns": [
                "ra",
                "dec",
                "mjd",
                "cntr_mf",
                "w1mpro_ep",
                "w1sigmpro_ep",
                "w2mpro_ep",
                "w2sigmpro_ep",
                "w1flux_ep",
                "w1sigflux_ep",
                "w2flux_ep",
                "w2sigflux_ep",
            ],
            "table": {"name": "allwise_p3as_mep"},
        },
        "positional_allwise_p3as_mep_mag_fluxes.txt",
    ),
    (
        {
            "type": "positional",
            "radius_arcsec": 6,
            "columns": [
                "ra",
                "dec",
                "mjd",
                "allwise_cntr",
                "w1mpro",
                "w1sigmpro",
                "w2mpro",
                "w2sigmpro",
            ],
            "table": {"name": "neowiser_p1bs_psd"},
        },
        "positional_neowiser_p1bs_psd_mag.txt",
    ),
    (
        {
            "type": "positional",
            "radius_arcsec": 6,
            "columns": [
                "ra",
                "dec",
                "mjd",
                "allwise_cntr",
                "w1flux",
                "w1sigflux",
                "w2flux",
                "w2sigflux",
            ],
            "table": {"name": "neowiser_p1bs_psd"},
        },
        "positional_neowiser_p1bs_psd_fluxes.txt",
    ),
    (
        {
            "type": "positional",
            "radius_arcsec": 6,
            "columns": [
                "ra",
                "dec",
                "mjd",
                "allwise_cntr",
                "w1mpro",
                "w1sigmpro",
                "w2mpro",
                "w2sigmpro",
                "w1flux",
                "w1sigflux",
                "w2flux",
                "w2sigflux",
            ],
            "table": {"name": "neowiser_p1bs_psd"},
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
