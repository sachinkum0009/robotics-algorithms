from controller import BaseController


class PIDController(BaseController):
    def __init__(self, kp: float, ki: float, kd: float):
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._previous_error = 0.0
        self._integral = 0.0

    def control(
        self, setpoint: float, measured_value: float, dt: float
    ) -> float:
        error = setpoint - measured_value
        self._integral += error * dt
        derivative = (error - self._previous_error) / dt if dt > 0 else 0.0

        output = (
            (self._kp * error)
            + (self._ki * self._integral)
            + (self._kd * derivative)
        )

        self._previous_error = error
        return output
    
    @property
    def kp(self) -> float:
        return self._kp

    @kp.setter
    def kp(self, value: float) -> None:
        self._kp = value

    @property
    def ki(self) -> float:
        return self._ki

    @ki.setter
    def ki(self, value: float) -> None:
        self._ki = value

    @property
    def kd(self) -> float:
        return self._kd

    @kd.setter
    def kd(self, value: float) -> None:
        self._kd = value
