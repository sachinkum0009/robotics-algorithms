from simpy import Environment, Event
from dataclasses import dataclass
from typing import Optional
import os
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
import logging
from diffusers.schedulers import DanceDiffusionPipeline


LOG_LVL = os.getenv("LOGLEVEL", "INFO")
logging.basicConfig(level=LOG_LVL.upper())
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dit:
    name: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None


def main():
    dit = Dit("john", "0.0.1", "this is diffusion transformer")
    logger.info(
        f"dit: {dit.name}, version: {dit.version}, description: {dit.description}"
    )
    logger.debug("this is debug message")


if __name__ == "__main__":
    main()
