# Copyright 2026 Sachin Kumar.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import numpy as np


@dataclass
class Pose:
    x: float
    y: float
    theta: float

    def __add__(self, other: "Pose") -> "Pose":
        """Overloads the + operator (returns a new Pose instance)."""
        if not isinstance(other, Pose):
            return NotImplemented

        return Pose(
            x=self.x + other.x,
            y=self.y + other.y,
            theta=self._normalize_angle(self.theta + other.theta),
        )

    def __iadd__(self, other: "Pose") -> "Pose":
        """Overloads the += operator (modifies self in place)."""
        if not isinstance(other, Pose):
            return NotImplemented

        self.x += other.x
        self.y += other.y
        self.theta = self._normalize_angle(self.theta + other.theta)
        return self

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Keeps theta wrapped within [-pi, pi]."""
        return np.atan2(np.sin(angle), np.cos(angle))
