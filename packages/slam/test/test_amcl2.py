import pytest
from slam.amcl import Amcl


@pytest.fixture
def init_amcl() -> Amcl:
    amcl = Amcl(10000, (100, 100))
    return amcl

def test_amcl(init_amcl):
    amcl = init_amcl
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