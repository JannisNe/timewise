from pathlib import Path
import pytest
from typer.testing import CliRunner
from timewise.cli import app


def test_download():
    runner = CliRunner()
    res = runner.invoke(app, ["download", "conf.yml"])
    assert res.exit_code == 1
    with pytest.raises(AssertionError, match="conf.yml not found"):
        raise res.exc_info[1]


def test_make_ampel_job(timewise_config_path):
    runner = CliRunner()
    res = runner.invoke(
        app, ["make-ampel-job", str(timewise_config_path), "test_ampel"]
    )
    assert res.exit_code == 0
    assert Path(res.output.split(" file: ")[-1].strip()).exists()
