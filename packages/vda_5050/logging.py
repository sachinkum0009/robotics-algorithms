import logging
from pathlib import Path


def setup_logging(
    level: str = "INFO", log_file: Path | None = None
) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file)
            if log_file
            else logging.NullHandler(),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)
