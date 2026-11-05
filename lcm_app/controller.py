from lcm_app.state import LightCurveState


class TESSController:
    """Coordinates light-curve state transitions for TESS workflows."""

    def __init__(self):
        self.state = LightCurveState()

    def load_new_data(self, time, flux, flux_err):
        self.state.set_new_data(time, flux, flux_err)

    def reset_state(self):
        self.state.reset_all()

    def set_cleaned_data(self, time, flux, flux_err):
        self.state.set_cleaned(time, flux, flux_err)

    def restore_cleaned_or_original(self):
        return self.state.restore_cleaned_or_original()

    def clear_pre_detrend(self):
        self.state.clear_pre_detrend()
