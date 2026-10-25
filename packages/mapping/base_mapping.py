"""
MIT
"""

from abc import ABC, abstractmethod

from numpy.typing import NDArray

from .utils import Pose


class BaseMapping(ABC):
    """Base Mapping Abstract Class"""

    @abstractmethod
    def update(self, lidar_data: NDArray, pose: Pose):
        """Updates the grid map from sensor data"""
        ...

    @abstractmethod
    def get_grid(self) -> NDArray:
        """Returns the current grid of the mapping algorithm"""
        ...

    @abstractmethod
    def get_pose(self) -> Pose:
        """Returns the current pose of the robot"""
        ...
