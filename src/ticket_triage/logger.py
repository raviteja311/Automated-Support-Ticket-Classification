import logging
import sys

from distro import name

def get_logger(name: str) -> logging.Logger:
    """Return a configured, non-duplicating stdout logger."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured, avoid duplicate handlers
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger