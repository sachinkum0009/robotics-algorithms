from jax import numpy as jnp
from jax.typing import ArrayLike


class Segmentation:
    def __init__(self):
        self.data = jnp.arange(4)

    def run(self) -> ArrayLike:
        return self.data
