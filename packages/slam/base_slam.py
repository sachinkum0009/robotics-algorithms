"""
SLAM
"""

from abc import ABC, abstractmethod


class BaseSlam(ABC):
    @abstractmethod
    def localize(self):
        pass

    @abstractmethod
    def mapping(self):
        pass
