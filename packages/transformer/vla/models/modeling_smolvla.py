"""
Modeling SmolVLA

"""

from os import PathLike

from torch import Tensor
from transformers.dynamic_module_utils import create_dynamic_module


class System2:
    def __init__(self):
        pass


class System1:
    def __init__(self):
        pass


class SmolVLA:
    def __init__(self):
        self.system1 = System1()
        self.system2 = System2()
        pass

    def forward(self, x: Tensor):
        return x
