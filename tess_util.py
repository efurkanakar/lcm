import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import astropy.units as u
from astroquery.mast import Catalogs, Observations, Tesscut
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QMessageBox
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from tess_stars2px import tess_stars2px_function_entry, TESS_Spacecraft_Pointing_Data
import re
from astropy.timeseries import LombScargle, BoxLeastSquares
import numpy as np
from lcm_app.config import load_download_dir

CCD_EDGE_PIXELS = 100
MIN_PERIOD = 0.06
AWS_BUCKET = "https://stpubdata.s3.amazonaws.com/"
AWS_PREFIX = "tess/public/tid/"
MAST_URL = "https://archive.stsci.edu/missions/tess/tid/"
LC_SUFFIX = {120: "s_lc.fits", 20: "a_fast-lc.fits"}
_download_lock = threading.Lock()


def kvw(t, f, half, search, step):
    """Kwee & van Woerden (1956) minimum time and error, relative to t = 0."""
    order = np.argsort(t)
    t, f = t[order], f[order]
    ks = np.arange(1, int(half / step) + 1) * step
    trials = np.arange(-search, search, step / 20)
    s = np.array([np.sum((np.interp(T + ks, t, f) - np.interp(T - ks, t, f)) ** 2) for T in trials])
    near = np.abs(trials - trials[np.argmin(s)]) <= step
    a, b, c = np.polyfit(trials[near], s[near], 2)
    z = np.sum(np.abs(t) < half) / 4
    return -b / (2 * a), np.sqrt((4 * a * c - b * b) / (4 * a * a * (z - 1)))


def eclipse_ephemeris(time, flux):
    """Primary-eclipse T0, period and number of minima used: BLS first guess, then KvW of every eclipse and a linear fit."""
    flux = flux / np.median(flux)
    cadence = np.median(np.diff(time))

    # Coarse BLS on 6-minute bins, then a fine period grid on the full data
    idx = np.round((time - time[0]) / max(cadence, 6 / 1440)).astype(int)
    _, inv, cnt = np.unique(idx, return_inverse=True, return_counts=True)
    coarse = BoxLeastSquares(np.bincount(inv, time) / cnt, np.bincount(inv, flux) / cnt).autopower(
        [0.01, 0.02, 0.04], minimum_period=MIN_PERIOD, maximum_period=np.ptp(time) / 3, frequency_factor=3)
    i = np.argmax(coarse.power)
    fine = BoxLeastSquares(time, flux).power(np.linspace(0.998, 1.002, 801) * coarse.period[i], coarse.duration[i])
    j = np.argmax(fine.power)
    period, t0, dur = fine.period[j], fine.transit_time[j], coarse.duration[i]

    # BLS locks onto half the period when it folds primary and secondary minima together
    half_box = dur / (4 * period)
    ph = ((time - t0) / (2 * period) + 0.25) % 1 - 0.25
    in0, in1 = np.abs(ph) < half_box, np.abs(ph - 0.5) < half_box
    noise = np.std(np.diff(flux)) / np.sqrt(2)  # point-to-point, so the star's own variation does not count
    if abs(flux[in0].mean() - flux[in1].mean()) > 5 * noise * np.sqrt(1 / in0.sum() + 1 / in1.sum()):
        t0 = t0 + period if flux[in1].mean() < flux[in0].mean() else t0
        period *= 2

    epoch = np.round((time - t0) / period).astype(int)
    tau = time - (t0 + epoch * period)
    cut, shoulder, half, search = 1.6 * dur, 0.8 * dur, 0.9 * dur, 0.15 * dur
    minima = []
    for e in np.unique(epoch):
        w = (epoch == e) & (np.abs(tau) < cut)
        if min(np.sum(tau[w] < 0), np.sum(tau[w] > 0)) < 0.8 * cut / cadence:
            continue
        sh = w & (np.abs(tau) > shoulder)
        level = np.polyval(np.polyfit(tau[sh], flux[sh], 1), tau[w])
        minima.append((e, kvw(tau[w], flux[w] / level, half, search, cadence)[0]))
    if len(minima) < 3:
        return t0, period, 0

    e, dt = np.array(minima).T
    t_min = t0 + e * period + dt
    keep = np.abs(dt) < 0.9 * search
    for _ in range(2):
        period, t0 = np.polyfit(e[keep], t_min[keep], 1)
        resid = t_min - (t0 + e * period)
        keep &= np.abs(resid) < 3 * np.std(resid[keep])
    return t0 + e[keep].min() * period, period, int(keep.sum())


def sinusoid_ephemeris(time, flux):
    """Time of minimum and period of the strongest sinusoid, e.g. a reflection effect."""
    ls = LombScargle(time, flux)
    freq, power = ls.autopower(minimum_frequency=3 / np.ptp(time), maximum_frequency=1 / MIN_PERIOD)
    f = freq[np.argmax(power)]
    grid = np.linspace(time[0], time[0] + 1 / f, 1000)
    return grid[np.argmin(ls.model(grid, f))], 1 / f


def resolve_coordinates(star_name):
    """Star name -> SkyCoord via MAST, cached across sessions because resolving takes about a second."""
    settings = QSettings("lcm", "lcm")
    cache = json.loads(settings.value("star_coordinates", "{}"))
    key = star_name.strip().lower()
    if key not in cache:
        coord = Observations.resolve_object(star_name)
        cache[key] = [coord.ra.deg, coord.dec.deg]
        settings.setValue("star_coordinates", json.dumps(cache))
    return SkyCoord(*cache[key], unit="deg")


def resolve_tic(star_name):
    """Star name -> TIC ID, cached across sessions because the catalogue query takes several seconds."""
    match = re.fullmatch(r"TIC\s*(\d+)", star_name.strip(), re.IGNORECASE)
    if match:
        return int(match.group(1))
    settings = QSettings("lcm", "lcm")
    cache = json.loads(settings.value("star_tic", "{}"))
    key = star_name.strip().lower()
    if key not in cache:
        found = Catalogs.query_region(resolve_coordinates(star_name), radius=5 * u.arcsec, catalog="TIC")
        found = found[~np.isin(found["disposition"].filled(""), ["DUPLICATE", "ARTIFACT"])]
        if len(found) == 0:
            raise ValueError(f"No TIC star found at the position of {star_name}.")
        cache[key] = int(found[np.argmin(found["dstArcSec"])]["ID"])
        settings.setValue("star_tic", json.dumps(cache))
    return cache[key]


def list_spoc_files(tic, sector):
    """Base URL and file names of the SPOC products of one target in one sector."""
    tid = f"{tic:016d}"
    folder = f"s{sector:04d}/{tid[:4]}/{tid[4:8]}/{tid[8:12]}/{tid[12:]}/"
    listing = requests.get(AWS_BUCKET, params={"list-type": 2, "prefix": AWS_PREFIX + folder}, timeout=30).text
    names = re.findall(r"/([^/<]+)</Key>", listing)
    if names:
        return AWS_BUCKET + AWS_PREFIX + folder, names
    # new sectors reach AWS later than MAST
    response = requests.get(MAST_URL + folder, timeout=30)
    return MAST_URL + folder, re.findall(r'href="(tess[^"/]+)"', response.text) if response.ok else []


def spoc_cadences(tic, sectors):
    """Sector -> exposure times (s) with a SPOC light curve, or None where the archive could not be reached."""
    def cadences(sector):
        try:
            names = list_spoc_files(tic, sector)[1]
        except requests.RequestException:
            return None
        return [exptime for exptime, suffix in LC_SUFFIX.items() if any(n.endswith("-" + suffix) for n in names)]

    with ThreadPoolExecutor(8) as pool:
        return dict(zip(sectors, pool.map(cadences, sectors)))


def fetch_spoc_lightcurve(tic, sector, exptime):
    """Local path of a SPOC light curve: from disk if downloaded before, else from the AWS mirror, else from MAST."""
    suffix = LC_SUFFIX[exptime]
    cache_dir = (load_download_dir() or Path.cwd()) / "mastDownload" / "TESS"
    with _download_lock:
        local = sorted(cache_dir.glob(f"*/tess*-s{sector:04d}-{tic:016d}-*-{suffix}"))
        if local:
            return local[-1]

        base, names = list_spoc_files(tic, sector)
        names = [n for n in names if n.endswith("-" + suffix)]
        if not names:
            return None

        name = sorted(names)[-1]
        response = requests.get(base + name, timeout=300)
        response.raise_for_status()
        target = cache_dir / re.sub(r"[-_]lc\.fits$", "", name) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        partial.write_bytes(response.content)
        partial.replace(target)
        return target


def tess_point_sectors(coord):
    """Sector -> (mid-point datetime, camera, CCD, near CCD edge) for every past and planned sector at a position."""
    _, _, _, sectors, cams, ccds, cols, rows, _ = tess_stars2px_function_entry(0, coord.ra.deg, coord.dec.deg)
    pointing = TESS_Spacecraft_Pointing_Data()
    mid = dict(zip(pointing.sectors, Time(pointing.midtimes, format="jd").datetime))
    # Science pixels span columns 45-2092 and rows 1-2048; sectors >= 1000 are special pointings
    return {
        int(s): (mid[s], int(cam), int(ccd), min(col - 45, 2092 - col, row - 1, 2048 - row) < CCD_EDGE_PIXELS)
        for s, cam, ccd, col, row in zip(sectors, cams, ccds, cols, rows) if 0 < s < 1000
    }

def get_tess_sector_data(star_name):
    query_result = Observations.query_criteria(obs_collection="TESS", objectname=star_name, dataproduct_type=["timeseries", "image"], radius=0)

    lc_data = []
    ffi_data = []

    for i in query_result:
        if i["dataproduct_type"] == "timeseries":
            if i['dataURL'][-9:-5] == "s_lc" or i['dataURL'][-12:-5] == "fast-lc":
                lc_data.append([i["sequence_number"], i["t_exptime"]])
        elif i["dataproduct_type"] == "image":
            if i["target_name"] == "TESS FFI":
                ffi_data.append([i["sequence_number"], "FFI", round(i["t_exptime"], 1)])

    lc_data.sort(key=lambda x: (x[0], x[1]))
    ffi_data.sort(key=lambda x: (x[0], x[2]))

    return lc_data, ffi_data

def search_tess_sector(star_name):
    star_name = star_name.strip()
    if not star_name:
        return "Please enter a star name.", [], {}, {}

    try:
        coord = resolve_coordinates(star_name)
        query_result = Tesscut.get_sectors(coordinates=coord)
    except Exception as e:
        return f"Error querying TESS sectors: {e}", [], {}, {}

    if len(query_result) == 0:
        return f"No TESS observations found for star: {star_name}", [], {}, {}

    sector_numbers = []
    for sector in query_result['sectorName']:
        parts = sector.split('-')
        if len(parts) > 1:
            try:
                sector_number = int(re.findall(r'\d+', parts[1])[0])
                sector_numbers.append(sector_number)
            except (ValueError, IndexError):
                continue

    sector_numbers = sorted(sector_numbers)
    sectors_text = "<br>".join(map(str, sector_numbers))

    message = (
        f"<div style='text-align: center;'>"
        f"TESS observations for {star_name} found in sectors:<br><br>{sectors_text}</div>"
    )

    return message, sector_numbers, tess_point_sectors(coord), spoc_cadences(resolve_tic(star_name), sector_numbers)

def tess_analysis(star_name, sector, exptime, flux_type, parent=None, apply_filter=True):
    try:
        # Query TESS data
        query = Observations.query_criteria(obs_collection="TESS",
                                            objectname=star_name,
                                            dataproduct_type="timeseries",
                                            sequence_number=sector,
                                            radius=0)
        if len(query) == 0:
            QMessageBox.warning(parent, "Warning", f"No data found for {star_name} in sector {sector}.")
            return None, None, None, None

        product_type = "LC" if exptime == 120 else "FAST-LC"
        products = Observations.get_product_list(query)
        filtered = Observations.filter_products(products, productSubGroupDescription=product_type)
        if len(filtered) == 0:
            QMessageBox.warning(parent, "Warning", f"No {product_type} data found for {star_name} in sector {sector}.")
            return None, None, None, None

        download_table = Observations.download_products(filtered, mrp_only=True)
        path = download_table["Local Path"][0]
        hdul = fits.open(path)

        time = hdul[1].data["TIME"]
        flux = hdul[1].data[flux_type]
        flux_err = hdul[1].data[f"{flux_type}_ERR"]
        quality = hdul[1].data["QUALITY"]

        if apply_filter:
            mask = (quality == 0) & ~np.isnan(time) & ~np.isnan(flux) & ~np.isnan(flux_err)

        else:
            mask = ~np.isnan(time) & ~np.isnan(flux) & ~np.isnan(flux_err)


        time, flux, flux_err, quality = time[mask], flux[mask], flux_err[mask], quality[mask]

        parent.time, parent.flux, parent.flux_err = time, flux, flux_err
        parent.canvas.figure.clear()
        ax_query = parent.canvas.figure.add_subplot(111)
        ax_query.errorbar(time, flux, yerr=flux_err, fmt=".", color="dodgerblue", markersize=5, capsize=1,
                          elinewidth=1.5, ls="none")
        ax_query.set_ylabel("Flux ($electrons^{-1}$)")
        ax_query.set_xlabel("Time - 2457000 ($day$)")
        ax_query.set_title(f"{flux_type.replace('_', ' ')} Light Curve", fontstyle="italic")
        parent.canvas.draw()

        parent.canvas_outliers.figure.clear()
        ax_outliers = parent.canvas_outliers.figure.add_subplot(111)
        ax_outliers.errorbar(time, flux, yerr=flux_err, fmt=".", color="dodgerblue",
                             markersize=5, capsize=1, elinewidth=1.5, ls="none")
        ax_outliers.set_ylabel("Flux ($electrons^{-1}$)")
        ax_outliers.set_xlabel("Time - 2457000 ($day$)")
        ax_outliers.set_title(f"{flux_type.replace('_', ' ')} Light Curve", fontstyle="italic")
        parent.canvas_outliers.draw()

        parent.canvas_detrending.figure.clear()
        ax_detrending = parent.canvas_detrending.figure.add_subplot(111)
        ax_detrending.errorbar(time, flux, yerr=flux_err, fmt=".", color="dodgerblue",
                               markersize=5, capsize=1, elinewidth=1.5, ls="none")
        ax_detrending.set_ylabel("Flux ($electrons^{-1}$)")
        ax_detrending.set_xlabel("Time - 2457000 ($day$)")
        ax_detrending.set_title(f"{flux_type.replace('_', ' ')} Light Curve", fontstyle="italic")
        parent.canvas_detrending.draw()

        return time, flux, flux_err, quality

    except Exception as e:
        QMessageBox.critical(parent, "Error", f"An error occurred while downloading and plotting data: {e}")
        return None, None, None, None

def phase_fold_manually(time, flux, flux_err, t0, period, canvas):

    median_flux = np.median(flux)
    flux = flux / median_flux
    if flux_err is not None:
        flux_err = flux_err / median_flux

    phase = ((time - t0) / period) % 1

    extended_phase = np.concatenate([phase, phase + 1])
    extended_flux = np.concatenate([flux, flux])
    extended_flux_err = np.concatenate([flux_err, flux_err])


    canvas.figure.clear()
    ax = canvas.figure.add_subplot(111)
    ax.errorbar(extended_phase, extended_flux, yerr=extended_flux_err, fmt='.',
                color="dodgerblue", markersize=5, capsize=1, elinewidth=1.5)
    ax.axvline(1, color='g', linestyle='--', label='Phase 1')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Phase-Folded Light Curve')
    ax.set_xlim(0.2, 1.2)
    ax.legend()
    canvas.draw()

    return extended_phase, extended_flux, extended_flux_err

def download_tess_lightcurve(star_name, sector, exptime, flux_type,
                             apply_filter=True):
    path = fetch_spoc_lightcurve(resolve_tic(star_name), sector, exptime)
    if path is None:
        return None, None, None, None
    with fits.open(path) as hdul:
        time = hdul[1].data["TIME"]
        flux = hdul[1].data[flux_type]
        flux_err = hdul[1].data[f"{flux_type}_ERR"]
        quality = hdul[1].data["QUALITY"]

    if apply_filter:
        mask = (quality == 0) & np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    else:
        mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)

    return time[mask], flux[mask], flux_err[mask], quality[mask]
