from estimators.kalman_filter import ExtendedKf


def test_ekf():
    ekf = ExtendedKf()
    state = ekf.get_state()
    assert len(state) == 9
