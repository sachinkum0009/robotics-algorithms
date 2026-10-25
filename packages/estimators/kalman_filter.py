from typing import Any

import numpy as np
from transformers.models.

from .filter import BaseFilter
from .utils import Gps, Imu, Odom


class LinearKf(BaseFilter):
    """Kalman Filter estimator for linear dynamic systems."""

    def __init__(self):
        self.x = np.zeros((2, 1))  # state

        self.P_ = np.eye(2, 2) * 0.05  # Initial covariance

        self.Q = np.eye(2, 2)  # Process noise
        self.R = np.eye(2, 2)  # Measurement noise

    def update(self):
        """Calculate the Covariance

        Args:

        Returns:
            None

        Raises:

        """
        return super().update()

    def predict(self, x: Any):
        """Estimates the fusion of the sensor

        Args:
            x: Any

        Return:
            Tuple

        Raises:
            ValueError:
        """
        return super().predict(x)


class ExtendedKf(BaseFilter):
    """Extended Kalman Filter estimator for non linear dynamic systems."""

    def __init__(self):
        self._state = np.zeros((9, 1)) * 1.0  # state
        # x, y, z, vx, vy, vz, roll, pitch, yaw

        self._P = np.eye(9) * 0.1

        self.Q = np.eye(9) * 0.01  # Process noise
        self.R_odometry = np.eye(3) * 0.05  # Measurement noise
        self.R_imu = np.eye(6) * 0.02
        self.R_gps = np.eye(3) * 0.1

    def update_imu(self, imu_data: Imu, dt: float):
        """
        Update step using angular velocity and acceleration
        """
        imu_arr = imu_data.to_vel_rot_array(dt)

        x, y, z, vx_prev, vy_prev, vz_prev, roll, pitch, yaw = self._state

        h = np.array([vx_prev, vy_prev, vz_prev, roll, pitch, yaw])

        H = np.zeros((6, 9))
        H[3, 6] = 1  # omega_x depends on roll
        H[4, 7] = 1  # omega_y depends on pitch
        H[5, 8] = 1  # omega_z depends on yaw

        # kalman gain
        S = H @ self._P @ H.T + self.R_imu
        K = self._P @ H.T @ np.linalg.inv(S)

        # update state and covariance
        residual = imu_arr - h
        self._state = self._state + K @ residual
        self._P = (np.eye(9) - K @ H) @ self._P

        self._state[6:8] += imu_arr[3:5]

        # self._state[6] += ox * dt
        # self._state[7] += oy * dt
        # self._state[8] += oz * dt

    def update_gps(self, gps_data: Gps, dt: float):
        """Update step using GPS position"""
        gps_arr = gps_data.to_array()
        x, y, z, vx, vy, vz, roll, pitch, yaw = self._state

        # Predicted GPS measurement (h)
        h = np.array([x, y, z])

        # Jacobian of the measurement model (H)
        H = np.zeros((3, 9))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # z

        # Kalman gain
        S = H @ self.covariance @ H.T + self.R_gps
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        residual = gps_arr - h
        self.state = self.state + K @ residual
        self.covariance = (np.eye(9) - K @ H) @ self.covariance

    def predict(self, odom_data: Odom, dt: float):
        "Predict step using Odometry"
        vx = odom_data.twist.vx
        vy = odom_data.twist.vy
        vz = odom_data.twist.vz

        x, y, z, vx_prev, vy_prev, vz_prev, roll, pitch, yaw = self._state

        # State transition model (non-linear)
        new_x = x + (vx * dt)
        new_y = y + (vy * dt)
        new_z = z + (vz * dt)
        new_vx = vx
        new_vy = vy
        new_vz = vz
        new_roll = roll
        new_pitch = pitch
        new_yaw = yaw

        # Jacobian of the state transition (F)
        F = np.eye(9)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        self._state = np.array(
            [
                new_x,
                new_y,
                new_z,
                new_vx,
                new_vy,
                new_vz,
                new_roll,
                new_pitch,
                new_yaw,
            ]
        )

        self._P = F @ self._P @ F.T + self.Q

    def get_state(self):
        return self._state

    def get_covariance(self):
        return self._P


class UnscentedKf(BaseFilter):
    """Unscented Kalman Filter estimator for linear dynamic systems."""

    def __init__(self):
        self.x = np.zeros((3, 1))  # state

        self.P_ = np.eye(3, 3) * 0.05  # Initial covariance

        self.Q = np.eye(3, 3)  # Process noise
        self.R = np.eye(3, 3)  # Measurement noise

    def update(self):
        return super().update()

    def predict(self, x: Any):
        return super().predict(x)
