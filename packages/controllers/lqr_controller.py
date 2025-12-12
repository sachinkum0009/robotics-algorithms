from .controller import BaseController


class LqrController(BaseController):
    def __init__(self, K):
        self.K = K  # State feedback gain matrix

    def control(self, current_state, reference_state) -> float:
        # LQR control law: u = -K * (x - x_ref)
        state_error = current_state - reference_state
        control_input = -self.K @ state_error
        return control_input
