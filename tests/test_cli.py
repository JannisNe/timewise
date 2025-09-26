from pathlib import Path
import pytest
from typer.testing import CliRunner
from timewise.cli import app

from tests.constants import AMPEL_CONFIG_PATH


def test_download():
    runner = CliRunner()
    res = runner.invoke(app, ["download", "conf.yml"])
    assert res.exit_code == 1
    with pytest.raises(AssertionError, match="conf.yml not found"):
        raise res.exc_info[1]


def test_make_ampel_job(timewise_config_path):
    runner = CliRunner()
    res = runner.invoke(app, ["prepare-ampel", str(timewise_config_path)])
    assert res.exit_code == 0
    assert Path(res.output.split(" file: ")[-1].strip()).exists()


def test_run_ampel(timewise_config_path, ampel_prepper):
    runner = CliRunner()
    res = runner.invoke(
        app, ["process", str(timewise_config_path), str(AMPEL_CONFIG_PATH)]
    )
    assert res.exit_code == 0
