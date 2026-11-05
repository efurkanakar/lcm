# AstPlan – Fast Streamlit Edition (single-plot + optional Phase overlay)
# - Single graph (Altitude), Phase markers optional (controls always visible)
# - T0 & P inputs are always visible (side-by-side); checkbox below toggles markers
# - 240 samples, twilight bands aligned with table (−0.833/−6/−12/−18°)
# - Sunset | Sunrise on the first row (left | right); other twilights paired below
# - Progress bar fixed at top, English messages, removed after completion
# - Observatory list: top-2 TR entries first, rest alphabetical; timezones auto (TZFinder)

import sys, subprocess, math, copy
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_sun, get_body
from astropy.time import Time
from astroplan import Observer
from astroquery.simbad import Simbad
import astropy.units as u

from datetime import date as pydate
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------
# Optional dependency: timezonefinder -> lat/lon -> IANA TZ
# ---------------------------------------------------------------------
def _ensure_timezonefinder():
    try:
        from timezonefinder import TimezoneFinder  # noqa
        return True
    except Exception:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "timezonefinder>=6.5"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            from timezonefinder import TimezoneFinder  # noqa
            return True
        except Exception:
            return False

_TZF_OK = _ensure_timezonefinder()
if _TZF_OK:
    from timezonefinder import TimezoneFinder
    _tf = TimezoneFinder()
else:
    _tf = None

def infer_tz(lat: float, lon: float) -> str:
    """IANA timezone; fallback to rough Etc/GMT±N if resolver not available."""
    if _tf is not None:
        try:
            tzname = _tf.timezone_at(lat=lat, lng=lon)
            if tzname:
                return tzname
        except Exception:
            pass
    # rough fallback by longitude (15° ~ 1h). Etc/GMT has inverted sign
    offset = int(round(lon / 15.0))
    return f"Etc/GMT{ -offset:+d}".replace("+", "")

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="AstPlan (Fast)", page_icon="🛰️", layout="centered")
st.title("🛰️ AstPlan – Fast Streamlit Edition")

DEFAULT_N_POINTS = 240

# -------------------------
# SIMBAD resolver (cached)
# -------------------------
@st.cache_data(show_spinner=False)
def simbad_coords_cached(name: str):
    name = (name or "").strip()
    if not name:
        return None
    candidates = [name]
    low = name.lower()
    if not (low.startswith("v*") or low.startswith("name ")):
        candidates.extend([f"V* {name}", f"NAME {name}"])

    for cand in candidates:
        try:
            tbl = Simbad().query_object(cand, timeout=10)
        except Exception:
            tbl = None
        if tbl is None:
            continue
        ra_candidates  = ["ra", "RA_d", "RA"]
        dec_candidates = ["dec", "DEC_d", "DEC"]
        ra_val = dec_val = None
        for c in ra_candidates:
            if c in tbl.colnames:
                ra_val = tbl[c][0]; break
        for c in dec_candidates:
            if c in tbl.colnames:
                dec_val = tbl[c][0]; break
        try:
            return float(ra_val), float(dec_val)
        except Exception:
            try:
                sc = SkyCoord(str(ra_val) + " " + str(dec_val), unit=(u.hourangle, u.deg), frame="icrs")
                return float(sc.ra.deg), float(sc.dec.deg)
            except Exception:
                continue

    for cand in candidates:
        try:
            sc = SkyCoord.from_name(cand)
            return float(sc.ra.deg), float(sc.dec.deg)
        except Exception:
            continue
    return None

# -------------------------
# Time & plotting helpers
# -------------------------
def times_to_naive_local(times: Time, tz: ZoneInfo):
    loc = times.to_datetime(timezone=tz)
    return np.array([dt.replace(tzinfo=None) for dt in loc], dtype=object)

def _mask_to_blocks(mask: np.ndarray):
    if mask.size == 0:
        return []
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(mask.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends   = np.where(edges == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]
    return list(zip(starts, ends))

def _std_atm_from_altitude(alt_m: float):
    # Simple standard atmosphere for refraction model
    P_hPa = 1013.25 * math.exp(-alt_m / 8434.5)  # scale height ~8.43 km
    return P_hPa * u.hPa, 10.0 * u.deg_C, 0.55 * u.micron

@st.cache_data(show_spinner=False)
def compute_altitude_series_cached(
    ra_deg: float, dec_deg: float, lat: float, lon: float, alt_m: float,
    tz_name: str, date_str: str, N: int, need_moon: bool
):
    tz = ZoneInfo(tz_name)
    location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=alt_m*u.m)
    observer = Observer(location=location, timezone=tz)
    obj = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)

    P, T, wl = _std_atm_from_altitude(alt_m)

    # Sunset/Sunrise at -0.833°
    start_time = observer.sun_set_time(Time(date_str), which='next', horizon=-0.833*u.deg)
    end_time   = observer.sun_rise_time(Time(date_str) + 1*u.day, which='next', horizon=-0.833*u.deg)

    times = start_time + (end_time - start_time) * np.linspace(0, 1, N)
    altaz = AltAz(obstime=times, location=location, pressure=P, temperature=T, obswl=wl)
    sun_altaz = get_sun(times).transform_to(altaz)

    # "Night": sun altitude < -0.833 deg
    night_mask = sun_altaz.alt < (-0.833 * u.deg)
    times_f = times[night_mask]
    if len(times_f) < 2:
        raise RuntimeError("No night interval for the selected date/location.")

    altaz_f = AltAz(obstime=times_f, location=location, pressure=P, temperature=T, obswl=wl)
    obj_f   = obj.transform_to(altaz_f)
    sun_f   = get_sun(times_f).transform_to(altaz_f)

    moon_alt = None
    if need_moon:
        moon_f  = get_body("moon", times_f).transform_to(altaz_f)
        moon_alt = moon_f.alt.to_value(u.deg)

    t_local = times_to_naive_local(times_f, tz)

    return (
        t_local,
        obj_f.alt.to_value(u.deg),
        obj_f.az.to_value(u.deg),
        sun_f.alt.to_value(u.deg),  # float deg array
        moon_alt,
        start_time, end_time
    )

def add_rise_transit_set(fig: go.Figure, observer: Observer, obj: SkyCoord,
                         tz: ZoneInfo, start_time: Time, end_time: Time):
    try:
        def to_pd(tt: Time):
            return pd.to_datetime(tt.to_datetime(timezone=tz))
        tr = observer.target_rise_time(start_time, target=obj, which='next')
        tt = observer.target_meridian_transit_time(start_time, target=obj, which='next')
        ts = observer.target_set_time(start_time, target=obj, which='next')
        for tmark, label, color in [(tr, 'Rise', 'green'), (tt, 'Transit', 'orange'), (ts, 'Set', 'red')]:
            if (tmark >= start_time) and (tmark <= end_time):
                fig.add_vline(x=to_pd(tmark), line=dict(color=color, dash='dot'),
                              annotation_text=label, annotation_position='top')
    except Exception:
        pass

def add_phase_markers_ephem(fig: go.Figure, tz: ZoneInfo,
                            start_time: Time, end_time: Time,
                            t0_jd: float, P_day: float):
    if P_day <= 0:
        return
    jd_start = start_time.jd
    jd_end   = end_time.jd
    n_start = int(np.ceil((jd_start - t0_jd)/P_day))
    n_end   = int(np.floor((jd_end   - t0_jd)/P_day))

    def add_lines(n_iterable, offset, color):
        for n in n_iterable:
            jd = t0_jd + (n + offset)*P_day
            if jd < jd_start or jd > jd_end:
                continue
            t = Time(jd, format='jd')
            ts = pd.to_datetime(t.to_datetime(timezone=tz))
            fig.add_vline(x=ts, line=dict(color=color, dash='dash'))

    add_lines(range(n_start, n_end+1), 0.0, 'blue')    # Primary
    add_lines(range(n_start, n_end+1), 0.5, 'purple')  # Secondary

def safe_local_str(t: Time, tz: ZoneInfo):
    try:
        return t.to_datetime(timezone=tz).strftime('%H:%M')
    except Exception:
        return "—"

@st.cache_data(show_spinner=False)
def compute_twilight_table(lat: float, lon: float, alt_m: float, tz_name: str, date_str: str):
    location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=alt_m*u.m)
    tz = ZoneInfo(tz_name)
    obs = Observer(location=location, timezone=tz)
    d0 = Time(date_str); d1 = Time(date_str) + 1*u.day

    H = -0.833 * u.deg
    sunset  = obs.sun_set_time(d0, which='next', horizon=H)
    sunrise = obs.sun_rise_time(d1, which='next', horizon=H)

    e_civ  = obs.twilight_evening_civil(d0, which='next')
    e_naut = obs.twilight_evening_nautical(d0, which='next')
    e_ast  = obs.twilight_evening_astronomical(d0, which='next')

    m_ast  = obs.twilight_morning_astronomical(d1, which='next')
    m_naut = obs.twilight_morning_nautical(d1, which='next')
    m_civ  = obs.twilight_morning_civil(d1, which='next')

    tbl = {
        "Sunset": safe_local_str(sunset, tz),
        "Civil twilight (evening)": safe_local_str(e_civ, tz),
        "Nautical twilight (evening)": safe_local_str(e_naut, tz),
        "Astronomical twilight (evening)": safe_local_str(e_ast, tz),
        "Astronomical twilight (morning)": safe_local_str(m_ast, tz),
        "Nautical twilight (morning)": safe_local_str(m_naut, tz),
        "Civil twilight (morning)": safe_local_str(m_civ, tz),
        "Sunrise": safe_local_str(sunrise, tz),
    }
    return tbl, dict(sunset=sunset, sunrise=sunrise, e_civ=e_civ, e_naut=e_naut, e_ast=e_ast,
                     m_ast=m_ast, m_naut=m_naut, m_civ=m_civ)

# -------------------------
# Observatory list (FULL) — paste your entire list here
# -------------------------
OBS_RAW = r"""
# Lines starting with # are ignored
# -------------------------------------------------------
# PASTE THE FULL LIST YOU PROVIDED (ALL LINES) BELOW, EXACTLY AS-IS (TAB-SEPARATED)
# -------------------------------------------------------
Tübitak National Obs. Antalya/Mt. Bakirlitepe	36.825	30.335	2515
Univ. of Ankara Obs. R Ankara	39.8433333333333	32.78	1266
... (PASTE FULL LIST ABOVE THESE DOTS; KEEP TAB-DELIMITERS) ...
"""

def parse_observatories(raw_text: str):
    obs = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or set(line) == {"-"," "}:
            continue
        parts = line.split("\t")
        if len(parts) >= 4:
            name = parts[0].strip()
            try:
                lat = float(parts[1]); lon = float(parts[2]); alt = float(parts[3])
            except Exception:
                # Fallback: last 3 numeric tokens
                tokens = [p for p in parts if p != ""]
                try:
                    lat = float(tokens[-3]); lon = float(tokens[-2]); alt = float(tokens[-1])
                    name = "\t".join(tokens[:-3]).strip()
                except Exception:
                    continue
        else:
            toks = [t for t in line.split() if t]
            try:
                lat = float(toks[-3]); lon = float(toks[-2]); alt = float(toks[-1])
                name = " ".join(toks[:-3])
            except Exception:
                continue

        tzname = infer_tz(lat, lon)
        if name.startswith("Tübitak National Obs.") or name.startswith("Univ. of Ankara Obs."):
            tzname = "Europe/Istanbul"
        obs.append( (name, lat, lon, alt, tzname) )
    return obs

OBS_PARSED = parse_observatories(OBS_RAW)

def build_location_dict():
    # Ensure required top-2 present with enforced tz
    top2_map = {}
    for (name, lat, lon, alt, tz) in OBS_PARSED:
        if name.startswith("Tübitak National Obs."):
            top2_map["Tübitak National Obs. Antalya/Mt. Bakirlitepe"] = (lat, lon, alt, "Europe/Istanbul")
        if name.startswith("Univ. of Ankara Obs. R Ankara"):
            top2_map["Univ. of Ankara Obs. R Ankara"] = (lat, lon, alt, "Europe/Istanbul")
    top2_map.setdefault("Tübitak National Obs. Antalya/Mt. Bakirlitepe", (36.825, 30.335, 2515.0, "Europe/Istanbul"))
    top2_map.setdefault("Univ. of Ankara Obs. R Ankara", (39.8433333333333, 32.78, 1266.0, "Europe/Istanbul"))

    # Rest alphabetical (dedup)
    rest = []
    top_names = set(top2_map.keys())
    for (name, lat, lon, alt, tz) in OBS_PARSED:
        if name in top_names:
            continue
        rest.append( (name, lat, lon, alt, tz) )
    rest_sorted = sorted(rest, key=lambda x: x[0].lower())

    locs = {}
    # Insert top-2 first (required order)
    locs["Tübitak National Obs. Antalya/Mt. Bakirlitepe"] = top2_map["Tübitak National Obs. Antalya/Mt. Bakirlitepe"]
    locs["Univ. of Ankara Obs. R Ankara"] = top2_map["Univ. of Ankara Obs. R Ankara"]
    # Then the rest
    for name, lat, lon, alt, tz in rest_sorted:
        if any(k in name for k in ["Istanbul", "Kandilli", "Ege Univ. Obs. Bornova"]):
            tz = "Europe/Istanbul"
        locs[name] = (lat, lon, alt, tz)
    return locs

PRESET_LOCS = build_location_dict()

# -------------------------
# UI
# -------------------------
left, right = st.columns(2)
with left:
    obj_name = st.text_input("Object (SIMBAD)", value="", help="e.g., Vega, Betelgeuse, MR Del…")
with right:
    date_val = st.date_input("Date", value=pydate.today())

with st.expander("Advanced settings", expanded=True):
    loc_names = list(PRESET_LOCS.keys())
    loc_sel = st.selectbox("Location", loc_names, index=0)
    lat, lon, alt, tz_name = PRESET_LOCS[loc_sel]

    c1, c2 = st.columns(2)
    ra_manual = c1.text_input("Manual RA (deg)", value="", help="If empty, SIMBAD will be used")
    dec_manual = c2.text_input("Manual Dec (deg)", value="")

    min_alt = st.slider("Min altitude guide (deg)", 0, 60, 20, step=1)
    twilight_bands = st.checkbox("Show twilight bands", value=True)
    show_moon = st.checkbox("Show Moon altitude curve", value=True)

    # Phase controls – always visible; checkbox toggles overlay
    p1, p2 = st.columns(2)
    t0_str = p1.text_input("Reference Minima Time T0 (JD)", value="")
    P_str  = p2.text_input("Period P (day)", value="")
    show_phase = st.checkbox("Show Phase markers", value=False, help="Draw minima lines using T0 and P")

run = st.button("Run")

# Layout placeholders (fixed order)
progress_ph = st.empty()
plot_ph = st.empty()
metrics_ph = st.container()

if run:
    prog = progress_ph.progress(1, text="Starting…")
    try:
        prog.progress(3, text="Validating inputs…")
        # Resolve coordinates
        prog.progress(5, text="Resolving coordinates (SIMBAD)…")
        if ra_manual.strip() and dec_manual.strip():
            try:
                ra, dec = float(ra_manual), float(dec_manual)
            except Exception:
                progress_ph.empty()
                st.error("Manual RA/Dec must be numeric.")
                st.stop()
        else:
            coords = simbad_coords_cached(obj_name.strip())
            if coords is None:
                progress_ph.empty()
                st.error("Object not found on SIMBAD. Check the name or provide manual RA/Dec.")
                st.stop()
            ra, dec = coords

        tz = ZoneInfo(tz_name)
        date_str = date_val.strftime("%Y-%m-%d")

        prog.progress(8, text="Preparing time window…")
        prog.progress(12, text="Computing altitude series…")
        (t_local, obj_alt, obj_az, sun_alt, moon_alt,
         start_time, end_time) = compute_altitude_series_cached(
            ra, dec, lat, lon, alt, tz.key if hasattr(tz, 'key') else tz_name,
            date_str, DEFAULT_N_POINTS, need_moon=show_moon
        )

        prog.progress(16, text="Computing twilight times…")
        times_tbl, _raw = compute_twilight_table(lat, lon, alt, tz.key if hasattr(tz,'key') else tz_name, date_str)

        prog.progress(20, text="Building plot…")
        t_series = pd.to_datetime([str(x) for x in t_local])
        fig = go.Figure()

        fig.add_trace(go.Scattergl(
            x=t_series, y=obj_alt, mode='markers',
            marker=dict(size=6, color=obj_az, colorscale='Viridis',
                        colorbar=dict(title='Az (deg)')),
            name='Object altitude',
            hovertemplate='Time: %{x|%H:%M}<br>Alt: %{y:.2f}°<br>Az: %{marker.color:.1f}°<extra></extra>'
        ))

        if show_moon and moon_alt is not None:
            fig.add_trace(go.Scatter(
                x=t_series, y=moon_alt, mode='lines', name='Moon altitude',
                line=dict(width=1),
                hovertemplate='Time: %{x|%H:%M}<br>Moon Alt: %{y:.2f}°<extra></extra>'
            ))

        if twilight_bands:
            prog.progress(24, text="Adding twilight bands…")
            for thr, alpha in [(-0.833, 0.05), (-6, 0.12), (-12, 0.20), (-18, 0.28)]:
                blocks = _mask_to_blocks(sun_alt < thr)  # float–float comparison
                for s, e in blocks:
                    fig.add_vrect(x0=t_series[s], x1=t_series[e-1],
                                  fillcolor='black', opacity=alpha, line_width=0)

        prog.progress(28, text="Adding rise/transit/set lines…")
        add_rise_transit_set(
            fig,
            Observer(location=EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=alt*u.m), timezone=tz),
            SkyCoord(ra=ra*u.deg, dec=dec*u.deg),
            tz, start_time, end_time
        )

        if min_alt > 0:
            fig.add_hline(y= min_alt, line=dict(color='red', width=1),
                          annotation_text=f"Min Alt {min_alt}°",
                          annotation_position="bottom right")

        # Phase overlay if requested
        if show_phase:
            prog.progress(32, text="Adding phase markers…")
            try:
                t0f = float(t0_str); Pf = float(P_str)
            except Exception:
                progress_ph.empty()
                st.error("T0 and P must be numeric."); st.stop()
            add_phase_markers_ephem(fig, tz, start_time, end_time, t0f, Pf)

        title_obj = obj_name if obj_name else f"RA={ra:.3f}°, Dec={dec:.3f}°"
        title_mode = "Altitude + Phase" if show_phase else "Altitude"
        fig.update_layout(
            title=f"{title_mode} plot for {title_obj} on {date_str} ({loc_sel})",
            xaxis_title=f"Time ({tz})",
            yaxis_title="Altitude (deg)",
            margin=dict(l=10, r=10, t=60, b=10),
            legend=dict(orientation='h', y=-0.2, font=dict(size=16)),
            height=640,
            font=dict(size=16)
        )
        fig.update_xaxes(tickfont=dict(size=16), title_font=dict(size=18))
        fig.update_yaxes(tickfont=dict(size=16), title_font=dict(size=18), range=[0, 90])

        prog.progress(36, text="Rendering plot…")
        plot_ph.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        prog.progress(42, text="Rendering times…")
        with metrics_ph:
            lc, rc = st.columns(2)
            with lc:
                st.metric("Sunset",  times_tbl["Sunset"])
            with rc:
                st.metric("Sunrise", times_tbl["Sunrise"])
            lc.metric("Civil twilight",    times_tbl["Civil twilight (evening)"])
            rc.metric("Civil twilight",    times_tbl["Civil twilight (morning)"])
            lc.metric("Nautical twilight", times_tbl["Nautical twilight (evening)"])
            rc.metric("Nautical twilight", times_tbl["Nautical twilight (morning)"])
            lc.metric("Astronomical twilight", times_tbl["Astronomical twilight (evening)"])
            rc.metric("Astronomical twilight", times_tbl["Astronomical twilight (morning)"])

        if show_phase:
            prog.progress(48, text="Preparing minima table…")
            rows = []
            jd_start, jd_end = start_time.jd, end_time.jd
            n_start = int(np.ceil((jd_start - t0f)/Pf))
            n_end   = int(np.floor((jd_end   - t0f)/Pf))
            for n in range(n_start, n_end+1):
                t_pri = Time(t0f + n*Pf, format='jd').to_datetime(timezone=tz)
                rows.append({"Minimum Type": "Primary (ϕ=0)", "Time": t_pri.strftime('%Y-%m-%d %H:%M')})
                jd_sec = t0f + (n+0.5)*Pf
                if jd_start <= jd_sec <= jd_end:
                    t_sec = Time(jd_sec, format='jd').to_datetime(timezone=tz)
                    rows.append({"Minimum Type": "Secondary (ϕ=0.5)", "Time": t_sec.strftime('%Y-%m-%d %H:%M')})
            if rows:
                df_min = pd.DataFrame(rows)
                st.subheader("Minima (Local Times)")
                st.dataframe(df_min, use_container_width=True)
            else:
                st.info("No minima within the selected night window.")

        # Smoothly advance progress to 100 with small steps
        for p, txt in [(58, "Finishing…"), (68, "Finishing…"), (78, "Finishing…"),
                       (88, "Finishing…"), (96, "Finishing…"), (100, "Done!")]:
            prog.progress(p, text=txt)
    finally:
        progress_ph.empty()
