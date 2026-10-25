import numpy as np

# from nptyping import NDArray, Shape, Float
import pytest
from numpy.typing import NDArray
from slam.amcl import Amcl


def test_amcl():
    amcl = Amcl(100, (10, 10))
    odom = (0.1, 0.0, 0.05)
    measurements = ((1.0, 0.0), (1.2, 0.1))
    amcl.predict(odom)
    amcl.update(measurements)
    amcl.resample()
    estimate = amcl.get_estimate()
    print(f"Estimated pose: {estimate}")
    odom = (0.3, 0.0, 0.05)
    amcl.predict(odom)
    amcl.resample()
    estimate = amcl.get_estimate()
    assert 1 + 2 == 3


def test_np():
    a: NDArray[np.float32] = np.zeros((2,2), dtype=np.float32)

    # print(a[3])
    # a = np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.2]])

    # weights = a[:, 3]
    
    # print(weights)
    # indices = np.random.choice(
    #     range(a.shape[0]), size=a.shape[0], p=weights / np.sum(weights)
    # )
    # print(indices)
    # print(a[indices])
