import logging
from pathlib import Path

import typer

from .io import Downloader
from .config import TimewiseConfig

from rich.logging import RichHandler


app = typer.Typer(help="Timewsie CLI")


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


@app.command(help="Download WISE photometry from IRSA")
def download(
    config_path: Path = typer.Argument(help="Pipeline config file (YAML/JSON)"),
):
    Downloader(TimewiseConfig.from_yaml(config_path).download).run()


@app.command(
    help="Reads the timewise config and replaces TIMEWISE_CONFIG_PATH and ORIGINAL_ID_KEY in the ampel job template"
)
def prepare_ampel(
    config_path: Path = typer.Argument(help="Pipeline config file (YAML/JSON)"),
):
    cfg = TimewiseConfig.from_yaml(config_path)
    ampel_prepper = cfg.build_ampel_prepper()
    p = ampel_prepper.prepare(config_path)
    typer.echo(f"AMPEL job file: {p}")
