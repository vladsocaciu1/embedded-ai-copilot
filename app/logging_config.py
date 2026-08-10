import logging

from logging.config import dictConfig
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def setup_logging(level=logging.INFO):
    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "standard": {
                "format": (
                    "%(asctime)s | %(levelname)-8s | "
                    "%(name)s | %(filename)s:%(lineno)d | %(message)s"
                )
            }
        },

        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": level,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(LOG_DIR / "app.log"),
                "maxBytes": 10 * 1024 * 1024,  # 10 MB
                "backupCount": 5,
                "formatter": "standard",
                "level": level,
                "encoding": "utf-8",
            },
        },

        "root": {
            "handlers": ["console", "file"],
            "level": level,
        },
    })