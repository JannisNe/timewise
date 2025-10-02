import pytest
from timewise.plot.diagnostic import make_plot
from tests.constants import DATA_DIR
from tests.util import restore_from_bson_dir


@pytest.mark.parametrize("cutout", ["sdss", "panstarrs"])
def test_make_plot(timewise_config_path, tmp_db_name, tmp_path, cutout):
    mongoexport_path = DATA_DIR / "mongodump"
    restore_from_bson_dir(mongoexport_path, tmp_db_name)
    indices = [0, 1]
    make_plot(timewise_config_path, cutout, indices, tmp_path)
    fns = [tmp_path / f"{index}.pdf" for index in indices]
    assert all([fn.exists() for fn in fns])
