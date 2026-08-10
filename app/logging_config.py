import logging

from logging.config import dictConfig

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
                "class": "logging.FileHandler",
                "filename": "app/logs/app.log",
                "formatter": "standard",
                "level": level,
            },
        },

        "root": {
            "handlers": ["console", "file"],
            "level": level,
        },
    })