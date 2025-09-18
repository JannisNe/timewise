import logging
from pathlib import Path
import yaml

import typer

from .io.download import Downloader
from .config import TimewiseConfig

from rich.logging import RichHandler


app = typer.Typer(help="Timewsie CLI")


def load_config(path: Path) -> TimewiseConfig:
    assert path.exists(), f"{path} not found!"
    with path.open("r") as f:
        config_dict = yaml.safe_load(f)
    return TimewiseConfig.model_validate(config_dict)


# --- Global callback (runs before every command) ---
@app.callback()
def main(
    ctx: typer.Context,
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        case_sensitive=False,
    ),
):
    """Global options for all Timewise commands."""
    # Normalize log level
    level = getattr(logging, log_level.upper(), None)
    if not isinstance(level, int):
        raise typer.BadParameter(f"Invalid log level: {log_level}")

    # Rich logging
    logging.basicConfig(
        level=level,
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )

    # Store log level in context for subcommands
    ctx.obj = {"log_level": level}


@app.command()
def download(
    config_path: Path = typer.Option(
        ..., "--config", "-c", exists=True, help="Pipeline config file (YAML/JSON)"
    ),
):
    Downloader(load_config(config_path).download).run()
