import logging
import os


def setup(level: str | None = None) -> logging.Logger:
    lvl = getattr(logging, (level or os.environ.get("LOG_LEVEL", "INFO")).upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger("ts-anomaly")
