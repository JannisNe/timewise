import logging
import os
from pathlib import Path


# Setting up the Logger
main_logger = logging.getLogger('timewise')
logger_format = logging.Formatter(
    '%(levelname)s:%(name)s:%(funcName)s - [%(threadName)s] - %(asctime)s: \n\t%(message)s', "%H:%M:%S"
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logger_format)
main_logger.addHandler(stream_handler)
main_logger.propagate = False  # do not propagate to root logger

logger = logging.getLogger(__name__)


def get_directories() -> dict[str, Path | None]:
    # Setting up data directory
    DATA_DIR_KEY = 'TIMEWISE_DATA'
    if DATA_DIR_KEY in os.environ:
        data_dir = Path(os.environ[DATA_DIR_KEY]).expanduser()
    else:
        logger.warning(f'{DATA_DIR_KEY} not set! Using home directory.')
        data_dir = Path('~/').expanduser()

    BIGDATA_DIR_KEY = 'TIMEWISE_BIGDATA'
    if BIGDATA_DIR_KEY in os.environ:
        bigdata_dir = Path(os.environ[BIGDATA_DIR_KEY]).expanduser()
        logger.info(f"Using bigdata directory {bigdata_dir}")
    else:
        bigdata_dir = None
        logger.info(f"No bigdata directory set.")

    output_dir = data_dir / 'output'
    plots_dir = output_dir / 'plots'
    cache_dir = data_dir / 'cache'

    return {
        'data_dir': data_dir,
        'bigdata_dir': bigdata_dir,
        'output_dir': output_dir,
        'plots_dir': plots_dir,
        'cache_dir': cache_dir
    }


def backoff_hndlr(details):
    logger.info("Backing off {wait:0.1f} seconds after {tries} tries "
                "calling function {target} with args {args} and kwargs "
                "{kwargs}".format(**details))
