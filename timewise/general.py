import logging, os


# Setting up the Logger
main_logger = logging.getLogger('timewise')
logger_format = logging.Formatter('%(levelname)s - %(name)s - %(asctime)s: %(message)s', "%H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logger_format)
main_logger.addHandler(stream_handler)
logger = main_logger.getChild(__name__)

# Setting up data directory
DATA_DIR_KEY = 'TIMEWISE_DATA'
if DATA_DIR_KEY in os.environ:
    data_dir = os.environ[DATA_DIR_KEY]
else:
    logger.warning(f'{DATA_DIR_KEY} not set! Using home directory.')
    data_dir = os.path.expanduser('~/')

output_dir = os.path.join(data_dir, 'output')
plots_dir = os.path.join(output_dir, 'plots')
cache_dir = os.path.join(data_dir, 'cache')

for d in [data_dir, output_dir, plots_dir, cache_dir]:
    if not os.path.isdir(d):
        os.mkdir(d)