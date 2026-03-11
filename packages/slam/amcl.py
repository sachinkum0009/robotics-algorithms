"""
Adaptive Monte Carlo Localization is a Probabilistic Robot Localization Algorithm.

This algorithm uses particle filter to match the scan data of robot
with the map to estimate the position of the robot
"""

import time
from dataclasses import dataclass
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from jax.typing import ArrayLike
from matplotlib import pyplot as plt
from numba import njit, prange
from numpy.typing import NDArray

# from .base_slam import BaseSlam


@njit(parallel=True)
def _predict_particles(particles, dx, dy, dtheta):
    for i in prange(particles.shape[0]):
        particles[i, 0] += dx + np.random.normal(0, 0.1)
        particles[i, 1] += dy + np.random.normal(0, 0.1)
        particles[i, 2] += dtheta + np.random.normal(0, 0.05)


@dataclass
class Particle:
    x: float
    y: float
    theta: float
    weight: float = 1.0


class Amcl:
    def __init__(
        self,
        num_particles: int,
        map_size: tuple,
        initial_pose: tuple = (0, 0, 0),
    ):
        # key = jax.random.PRNGKey(0)
        # self.particles = init_particles(num_particles, initial_pose, key)
        self.num_particles = num_particles
        self.map_size = map_size
        self.particles = np.zeros((num_particles, 4))
        self.particles[:, 0] = initial_pose[0]
        self.particles[:, 1] = initial_pose[1]
        self.particles[:, 2] = initial_pose[2]
        self.particles[:, 3] = 1.0 / num_particles
        # self.particles = [
        #     Particle(*initial_pose) for _ in range(num_particles)
        # ]
        # self.normalize_weights()

    # @staticmethod
    # def init_particles(
    #     num_particles: int, initial_pose: tuple, key: ArrayLike
    # ):
    #     keys = jax.random.split(key, num_particles)
    #     noise = jax.random.normal(keys, (num_particles, 2)) * 0.1
    #     theta_noise = jax.random.normal(keys, (num_particles,)) * 0.05
    #     particles = jnp.zeros((num_particles, 3))
    #     particles = particles.at[:, 0].set(initial_pose[0] + noise[:, 0])
    #     particles = particles.at[:, 1].set(initial_pose[1] + noise[:, 1])
    #     particles = particles.at[:, 2].set(initial_pose[2] + theta_noise)

    # def normalize_weights(self):
    #     weights = np.array([p.weight for p in self.particles])
    #     weights /= np.sum(weights)
    #     for i, p in enumerate(self.particles):
    #         p.weight = weights[i]

    # @partial(jax.jit, static_argnums=(2,))
    def predict(self, odom: tuple):
        """Predict the new particle states based on odometry."""
        # _predict_particles(self.particles, odom[0], odom[1], odom[2])
        # return self.particles
        self.particles[:, 0] += odom[0] + np.random.normal(
            0, 0.1, size=self.num_particles
        )
        self.particles[:, 1] += odom[1] + np.random.normal(
            0, 0.1, size=self.num_particles
        )
        self.particles[:, 2] += odom[2] + np.random.normal(
            0, 0.05, size=self.num_particles
        )
        # for p in self.particles:

    def update(self, measurements: tuple):
        # for p in self.particles:
        #     p[3] = 1.0 / (1.0 + abs(np.random.normal(0, 0.5)))
        self.particles[:, 3] = 1.0 / (
            1.0 + abs(np.random.normal(0, 0.05, size=len(self.particles)))
        )

    def resample(self):
        weights = np.array([p[3] for p in self.particles])
        indices = np.random.choice(
            range(self.num_particles),
            size=self.num_particles,
            p=weights / np.sum(weights),
        )
        self.particles = self.particles[indices]
        # self.normalize_weights()

    def get_estimate(self) -> tuple:
        # weights = self.particles[:, 3] # np.array([p[3] for p in self.particles])
        # # print(weights)
        # poses =  np.array([[p[0], p[1], p[2]] for p in self.particles])
        # print(poses)
        return np.average(
            self.particles[:, :3], axis=0, weights=self.particles[:, 3]
        )


def main():
    amcl = Amcl(100000, (10000, 10000))
    odom = (0.1, 0.0, 0.05)
    measurements = ((1.0, 0.0), (1.2, 0.1))
    amcl.predict(odom)  # dry run
    start_time = time.time()
    amcl.predict(odom)
    predict_time = time.time()
    amcl.update(measurements)
    update_time = time.time()
    amcl.resample()
    resample_time = time.time()
    estimate = amcl.get_estimate()
    estimate_time = time.time()
    print(f"Estimated pose: {estimate}")
    print(
        f"predict time: {predict_time - start_time}, update time; {update_time - predict_time}, sample time: {resample_time - update_time}, estimate time: {estimate_time - resample_time}"
    )

    odom = (0.3, 0.0, 0.05)
    start_time = time.time()
    amcl.predict(odom)
    predict_time = time.time()
    amcl.update(measurements)
    update_time = time.time()
    amcl.resample()
    resample_time = time.time()
    estimate = amcl.get_estimate()
    estimate_time = time.time()
    print(f"Estimated pose: {estimate}")
    print(
        f"predict time: {predict_time - start_time}, update time; {update_time - predict_time}, sample time: {resample_time - update_time}, estimate time: {estimate_time - resample_time}"
    )
    # amcl.resample()
    # estimate = amcl.get_estimate()
    # print(f"Estimated pose: {estimate}")


if __name__ == "__main__":
    main()
