import logging
import pathlib


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    log_file_path = pathlib.Path(log_file)

    if not log_file_path.parent.exists():
        log_file_path.parent.mkdir(parents=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console)

    return logger
