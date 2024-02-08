import yaml, logging, socket, os


class Rank(logging.Filter):
    def filter(self, record):
        record.global_rank = os.environ.get("GLOBAL_RANK", 0)
        record.local_rank = os.environ.get("LOCAL_RANK", 0)
        return True

def get_logger(name):
    logger = logging.Logger(name)
    # logger.addFilter(Rank())
    level = {
        'crititical': 50,
        'error': 40,
        'warning': 30,
        'info': 20,
        'debug': 10
    }[os.environ.get('LOGGER_LEVEL', 'info')]
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    os.makedirs(os.environ.get('MODEL_DIR', './workdir/default'),exist_ok=True)
    fh = logging.FileHandler(os.path.join(os.environ.get('MODEL_DIR', './workdir/default'), 'log.out'))
    fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(f'%(asctime)s [{socket.gethostname()}:%(process)d:%(global_rank)s:%(local_rank)s]
    # [%(levelname)s] %(message)s') #  (%(name)s)
    formatter = logging.Formatter(f'%(asctime)s [{socket.gethostname()}:%(process)d] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
