"""
Implementation of Monte Carlo Localization
and Adaptive Monte Carlo Localization
"""

import random
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Particle:
    x: float
    y: float
    theta: float
    weight: float = 1.0


class MCL:
    def __init__(self):
        self.a = 1
        self.b = (1, 2, 2)


class AMCL:
    def __init__(self):
        self.data = np.zeros((3, 1))
