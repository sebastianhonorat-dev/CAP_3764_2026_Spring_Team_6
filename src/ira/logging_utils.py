import logging


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    logger = logging.getLogger(name)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return logger
