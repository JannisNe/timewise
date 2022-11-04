import logging, os


# Setting up the Logger
main_logger = logging.getLogger('timewise')
logger_format = logging.Formatter('%(levelname)s:%(threadName)s %(name)s - %(asctime)s: %(message)s', "%H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logger_format)
main_logger.addHandler(stream_handler)

logger = logging.getLogger(__name__)

# Setting up data directory
DATA_DIR_KEY = 'TIMEWISE_DATA'
if DATA_DIR_KEY in os.environ:
    data_dir = os.environ[DATA_DIR_KEY]
else:
    logger.warning(f'{DATA_DIR_KEY} not set! Using home directory.')
    data_dir = os.path.expanduser('~/')

BIGDATA_DIR_KEY = 'TIMEWISE_BIGDATA'
if BIGDATA_DIR_KEY in os.environ:
    bigdata_dir = os.environ[BIGDATA_DIR_KEY]
    logger.info(f"Using bigdata directory {bigdata_dir}")
else:
    bigdata_dir = None
    logger.info(f"No bigdata directory set.")

output_dir = os.path.join(data_dir, 'output')
plots_dir = os.path.join(output_dir, 'plots')
cache_dir = os.path.join(data_dir, 'cache')

for d in [data_dir, output_dir, plots_dir, cache_dir]:
    if not os.path.isdir(d):
        os.mkdir(d)


def backoff_hndlr(details):
    logger.info("Backing off {wait:0.1f} seconds after {tries} tries "
                "calling function {target} with args {args} and kwargs "
                "{kwargs}".format(**details))
