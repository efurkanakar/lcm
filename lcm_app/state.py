class LightCurveState:
    """Holds original/current/cleaned light-curve data and related caches."""

    def __init__(self):
        self.reset_all()

    def reset_all(self):
        self.current_time = None
        self.current_flux = None
        self.current_flux_err = None

        self.original_time = None
        self.original_flux = None
        self.original_flux_err = None

        self.cleaned_time = None
        self.cleaned_flux = None
        self.cleaned_flux_err = None

        self.pre_detrend_time = None
        self.pre_detrend_flux = None
        self.pre_detrend_flux_err = None

    def set_new_data(self, time, flux, flux_err):
        self.original_time = time.copy()
        self.original_flux = flux.copy()
        self.original_flux_err = flux_err.copy()

        self.current_time = time.copy()
        self.current_flux = flux.copy()
        self.current_flux_err = flux_err.copy()

        self.cleaned_time = None
        self.cleaned_flux = None
        self.cleaned_flux_err = None
        self.clear_pre_detrend()

    def set_cleaned(self, time, flux, flux_err):
        self.cleaned_time = time
        self.cleaned_flux = flux
        self.cleaned_flux_err = flux_err

        self.current_time = time
        self.current_flux = flux
        self.current_flux_err = flux_err

    def restore_cleaned_or_original(self):
        if self.cleaned_time is not None and self.cleaned_flux is not None and self.cleaned_flux_err is not None:
            self.current_time = self.cleaned_time.copy()
            self.current_flux = self.cleaned_flux.copy()
            self.current_flux_err = self.cleaned_flux_err.copy()
            return "Cleaned"
        if self.original_time is not None and self.original_flux is not None and self.original_flux_err is not None:
            self.current_time = self.original_time.copy()
            self.current_flux = self.original_flux.copy()
            self.current_flux_err = self.original_flux_err.copy()
            return "Original"
        return "Empty"

    def clear_pre_detrend(self):
        self.pre_detrend_time = None
        self.pre_detrend_flux = None
        self.pre_detrend_flux_err = None
