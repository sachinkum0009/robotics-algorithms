from dataclasses import dataclass

import numpy as np


@dataclass
class Gps:
    x_gps: float
    y_gps: float
    z_gps: float

    def to_array(self):
        return np.array([self.x_gps, self.y_gps, self.z_gps])


@dataclass
class Imu:
    ax: float
    ay: float
    az: float
    ox: float
    oy: float
    oz: float

    def to_array(self):
        return np.array([self.ax, self.ay, self.az, self.ox, self.oy, self.oz])

    def to_vel_rot_array(self, dt: float):
        return np.array(
            [
                self.ax * dt,
                self.ay * dt,
                self.az * dt,
                self.ox,
                self.oy,
                self.oz,
            ]
        )


@dataclass
class Position:
    x: float
    y: float
    z: float


@dataclass
class Orientation:
    x: float
    y: float
    z: float
    w: float


@dataclass
class Linear:
    position: Position
    orientation: Orientation


@dataclass
class Twist:
    vx: float
    vy: float
    vz: float


@dataclass
class Odom:
    linear: Linear
    twist: Twist
