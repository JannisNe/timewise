import subprocess

import pytest
from timewise.plot.diagnostic import make_plot
from tests.constants import DATA_DIR


@pytest.mark.parametrize("cutout", ["sdss", "panstarrs"])
def test_make_plot(timewise_config_path, tmp_db_name, tmp_path, cutout):
    mongoexport_path = DATA_DIR / "mongodump.agz"
    print(tmp_db_name)
    cmd = [
        "mongorestore",
        "--nsFrom=jannisnecker_pytest-14_test_run_ampel1.*",
        f"--nsTo={tmp_db_name}.*",
        f"--archive={mongoexport_path}",
        "--gzip",
    ]
    subprocess.run(cmd)

    indices = [0, 1]
    make_plot(timewise_config_path, cutout, indices, tmp_path)
    fns = [tmp_path / f"{index}.pdf" for index in indices]
    assert all([fn.exists() for fn in fns])
