import logging
from typing import Annotated, Literal, List
from pathlib import Path

import typer

from .config import TimewiseConfig
from .plot.diagnostic import make_plot

from rich.logging import RichHandler


app = typer.Typer(help="Timewsie CLI")

config_path_type = Annotated[
    Path, typer.Argument(help="Pipeline config file (YAML/JSON)")
]
ampel_config_path_type = Annotated[Path, typer.Argument(help="AMPEL config YAML")]


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
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
    logging.getLogger("timewise").setLevel(level)

    # Store log level in context for subcommands
    ctx.obj = {"log_level": level}


@app.command(help="Download WISE photometry from IRSA")
def download(
    config_path: config_path_type,
):
    TimewiseConfig.from_yaml(config_path).download.build_downloader().run()


@app.command(help="Prepares the AMPEL job file so AMPEL can be run manually")
def prepare_ampel(
    config_path: config_path_type,
):
    cfg = TimewiseConfig.from_yaml(config_path)
    ampel_interface = cfg.build_ampel_interface()
    p = ampel_interface.prepare(config_path)
    typer.echo(f"AMPEL job file: {p}")


@app.command(help="Processes the lightcurves using AMPEL")
def process(
    config_path: config_path_type,
    ampel_config_path: ampel_config_path_type,
):
    cfg = TimewiseConfig.from_yaml(config_path)
    ampel_interface = cfg.build_ampel_interface()
    ampel_interface.run(config_path, ampel_config_path)


@app.command(help="Write stacked lightcurves to disk")
def export(
    config_path: config_path_type,
    output_directory: Annotated[Path, typer.Argument(help="output directory")],
    indices: Annotated[
        list[int] | None,
        typer.Option(
            "-i", "--indices", help="Indices to export, defaults to all indices"
        ),
    ] = None,
):
    TimewiseConfig.from_yaml(config_path).build_ampel_interface().export_many(
        output_directory, indices
    )


@app.command(help="Run download, process and export")
def run_chain(
    config_path: config_path_type,
    ampel_config_path: ampel_config_path_type,
    output_directory: Annotated[Path, typer.Argument(help="output directory")],
    indices: Annotated[
        list[int] | None,
        typer.Option(
            "-i", "--indices", help="Indices to export, defaults to all indices"
        ),
    ] = None,
):
    download(config_path)
    process(config_path, ampel_config_path)
    export(config_path, output_directory, indices)


@app.command(help="Make diagnostic plots")
def plot(
    config_path: config_path_type,
    indices: Annotated[
        List[int],
        typer.Argument(help="Identifiers of the objects for which to create plots"),
    ],
    output_directory: Annotated[Path, typer.Argument(help="Output directory")],
    cutout: Annotated[
        Literal["sdss", "panstarrs"],
        typer.Option("-c", "--cutout", help="Which survey to use for cutouts"),
    ] = "panstarrs",
):
    make_plot(
        config_path, indices=indices, cutout=cutout, output_directory=output_directory
    )
