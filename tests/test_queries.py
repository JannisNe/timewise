import pytest
from pathlib import Path
from timewise.query import QueryConfig

DATA_DIR = Path(__file__).parent / "data" / "queries"

# Map query "type" → config dict to instantiate
query_inputs = [
    (
        {"type": "positional_allwise",
         "radius_arcsec": 6,
         "magnitudes": True,
         "fluxes": False},
        "positional_allwise_mag.txt"
    )
]


def normalize_sql(s: str) -> str:
    return " ".join(s.split())


@pytest.mark.parametrize("config_dict,ref_path", query_inputs)
def test_query_build_matches_reference(config_dict, ref_path):
    # Load reference text
    ref_file = DATA_DIR / ref_path
    expected = normalize_sql(ref_file.read_text())

    # Instantiate query from config
    cfg = QueryConfig.model_validate({"query": config_dict})

    # Build query and compare
    built = normalize_sql(cfg.query.build())

    print(expected)
    print(built)

    assert built == expected, f"query mismatch"
