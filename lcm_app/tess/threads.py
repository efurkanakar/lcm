from PyQt5.QtCore import QThread, pyqtSignal
from datetime import datetime

from tess_util import (search_tess_sector, get_tess_sector_data, tess_point_sectors, resolve_coordinates,
                       resolve_tic, fetch_spoc_lightcurve)


def tess_future_sectors(target_name: str,
                        today: datetime | None = None) -> list[tuple[int, str]]:
    today = today or datetime.utcnow()
    return [
        (sec, mid.strftime("%Y-%m-%d"))
        for sec, (mid, *_) in sorted(tess_point_sectors(resolve_coordinates(target_name)).items())
        if mid > today
    ]


class TESSFutureThread(QThread):
    resultReady = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, star_name: str):
        super().__init__()
        self.star_name = star_name

    def run(self):
        try:
            fut = tess_future_sectors(self.star_name)
            self.resultReady.emit(fut)
        except Exception as e:
            self.error.emit(str(e))


class TESSDownloadThread(QThread):
    resultReady = pyqtSignal(object, object, object, object)
    error = pyqtSignal(str)

    def __init__(self, star_name, sector, exptime, flux_type, apply_filter):
        super().__init__()
        self.star_name = star_name
        self.sector = sector
        self.exptime = exptime
        self.flux_type = flux_type
        self.apply_filter = apply_filter

    def run(self):
        try:
            from tess_util import download_tess_lightcurve
            out = download_tess_lightcurve(self.star_name,
                                           self.sector,
                                           self.exptime,
                                           self.flux_type,
                                           self.apply_filter)
            if out[0] is None:
                self.error.emit(f"No {self.exptime}-s light curve in sector {self.sector}; "
                                f"this sector may have FFI data only.")
            else:
                self.resultReady.emit(*out)
        except Exception as e:
            self.error.emit(str(e))


class TESSPrefetchThread(QThread):
    """Downloads a sector's light curve in the background so a later click opens it from disk."""

    def __init__(self, star_name, sector, exptime):
        super().__init__()
        self.star_name = star_name
        self.sector = sector
        self.exptime = exptime

    def run(self):
        try:
            fetch_spoc_lightcurve(resolve_tic(self.star_name), self.sector, self.exptime)
        except Exception:
            pass  # the user's own download reports any error


class TESSSectorSearchThread(QThread):
    resultReady = pyqtSignal(object, object, object, object)
    error = pyqtSignal(str)

    def __init__(self, star_name: str):
        super().__init__()
        self.star_name = star_name

    def run(self):
        try:
            self.resultReady.emit(*search_tess_sector(self.star_name))
        except Exception as e:
            self.error.emit(str(e))


class TESSDataFetchThread(QThread):
    resultReady = pyqtSignal(list, list)
    error = pyqtSignal(str)

    def __init__(self, star_name: str):
        super().__init__()
        self.star_name = star_name

    def run(self):
        try:
            lc_data, ffi_data = get_tess_sector_data(self.star_name)
            self.resultReady.emit(lc_data, ffi_data)
        except Exception as e:
            self.error.emit(str(e))


class PhaseFoldThread(QThread):
    resultReady = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, ephemeris_function, time, flux):
        super().__init__()
        self.ephemeris_function = ephemeris_function
        self.time = time
        self.flux = flux

    def run(self):
        try:
            self.resultReady.emit(self.ephemeris_function(self.time, self.flux))
        except Exception as e:
            self.error.emit(str(e))
