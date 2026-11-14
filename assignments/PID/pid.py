from numpy import clip

class PID:
    def __init__(
            self, gain_prop: float, gain_int: float, gain_der: float, sensor_period: float,
            output_limits: tuple[float, float], wrap_angle: bool = True
            ):
        self.gain_prop = gain_prop
        self.gain_der = gain_der
        self.gain_int = gain_int
        self.sensor_period = sensor_period
        # TODO: define additional attributes you might need
        self.output_limits = output_limits
        self.error_sum = 0
        self.wrap_angle = wrap_angle
        # END OF TODO


    # TODO: implement function which computes the output signal
    # The controller should output only in the range of output_limits
    def output_signal(self, commanded_variable: float, sensor_readings: list[float]) -> float:
        error = self.wrap(commanded_variable - sensor_readings[0])
        d_error = self.wrap(sensor_readings[1] - sensor_readings[0]) / self.sensor_period
        self.error_sum += error * self.sensor_period
        res = self.gain_prop * error + self.gain_der * d_error + self.gain_int * self.error_sum
        if not (self.output_limits[0] < res < self.output_limits[1]):
            self.error_sum -= error * self.sensor_period  # dont integrate while already at the limit
        return clip(res, *self.output_limits)

    def reset_errors(self) -> None:
        self.error_sum = 0

    def wrap(self, x) -> float:
        return (x + 180) % 360 - 180 if self.wrap_angle else x
    # END OF TODO
