from pid_controller import PIDController


def main():
    pid_controller = PIDController(1.0, 0.1, 0.01)
    set_point = 15
    measured_value = 5
    dt = 0.1
    print(f"kp: {pid_controller.kp}")
    pid_controller.kp = 2.0
    print(f"new kp: {pid_controller.kp}")
    for i in range(20):
        control_output = pid_controller.control(set_point, measured_value, dt)
        print(f"control output: {control_output}")
        if measured_value < 15:
            measured_value += 1


if __name__ == "__main__":
    main()
