"""
Gmapping algorithm
"""

import numpy as np
from nptyping import NDArray

from .base_mapping import BaseMapping
from .utils import Pose


class GMapping(BaseMapping):
    """GMapping algorithm to create a grid map from the sensor readings"""

    def __init__(self, grid_size: int, resolution: float):
        self.grid_size = grid_size
        self.resolution = resolution
        self.grid = np.zeros([grid_size, grid_size])
        self.robot_pose = Pose(0.0, 0.0, 0.0)

    def update(
        self, lidar_data: NDArray[[np.float32, np.float32]], pose: Pose
    ):
        # Here I assume that lidar_data is being passed in Cartesian format
        self.robot_pose += pose
        for pt in lidar_data:
            self.grid[self.robot_pose.x, self.robot_pose.y] = pt

    def get_grid(self) -> NDArray:
        return self.grid

    def get_pose(self) -> Pose:
        return self.robot_pose
