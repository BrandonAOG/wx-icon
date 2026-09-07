"""
Everything you'd want to tweak lives here: which model this repo renders,
which regions and parameters, forecast hours, and how many runs to keep.

One repo renders one model. Pick it with the WX_MODEL environment variable
(set in the GitHub Actions workflow); default is gfs.
"""
import os

SITE_NAME = "WxModels"

# --------------------------------------------------------------- models -----

MODELS = {
    "gfs": {
        "id": "gfs",
        "name": "GFS",
        "resolution": "0.25°",
        "source": "nomads",
        "cycles": [0, 6, 12, 18],
        "min_age_hours": 3.5,          # how long after cycle time f000..f384 are complete
        # 3-hourly to 240 h, 12-hourly to 384 h
        "hours": list(range(0, 241, 6)) + list(range(252, 361, 12)),
        "params": None,                 # None = every product in PARAMS
        "credit": "NOAA/NCEP GFS via NOMADS",
    },
    "ecmwf": {
        "id": "ecmwf",
        "name": "ECMWF",
        "resolution": "0.25°",
        "source": "ecmwf_opendata",
        "cycles": [0, 6, 12, 18],      # 06/18 are published with a shorter range; probed at run time
        "min_age_hours": 7,
        # open data: 3-hourly to 144 h, 6-hourly to 240 h (00/12); 06/18 stop earlier
        "hours": list(range(0, 241, 6)),
        "probe_max_hours": [240, 144, 90],
        "params": None,                 # None = every product whose "ecmwf" spec isn't None
        "credit": "ECMWF open data (CC-BY-4.0)",
    },
}

MODELS["cmc"] = {
    "id": "cmc", "name": "CMC GDPS", "resolution": "15 km", "source": "cmc",
    "cycles": [0, 12], "min_age_hours": 5.5,
    "hours": list(range(0, 241, 6)),                    # GDPS: 3-hourly to 240 h
    "params": None, "credit": "Environment and Climate Change Canada GDPS (MSC Datamart)",
}
MODELS["icon"] = {
    "id": "icon", "name": "ICON", "resolution": "13 km", "source": "icon",
    "cycles": [0, 6, 12, 18], "min_age_hours": 4,
    "hours": list(range(0, 181, 6)),                    # 00/12 to 180 h; 06/18 to 120 h (probed)
    "probe_max_hours": [180, 120],
    "params": None, "credit": "Deutscher Wetterdienst ICON (open data, CC-BY-4.0)",
}

MODEL = MODELS[os.environ.get("WX_MODEL", "gfs").lower()]
FORECAST_HOURS = MODEL["hours"]


# Which generic field names each non-GFS source can supply (see fetch.py tables).
SOURCE_FIELDS = {
    "ecmwf_opendata": {"gh", "t", "u", "v", "r", "msl", "tp", "2t", "10u", "10v", "tcwv", "vo", "skt", "lsm"},
    "cmc":            {"gh", "t", "u", "v", "r", "msl", "tp", "2t", "10u", "10v", "vo", "cape", "snod", "skt", "lsm"},
    "icon":           {"gh", "t", "u", "v", "r", "msl", "tp", "2t", "10u", "10v", "tcwv", "cape", "snod", "skt", "lsm"},
}


def supported(pid: str) -> bool:
    if MODEL["source"] == "nomads":
        return PARAMS[pid].get("fetch") is not None
    spec = PARAMS[pid].get("spec")
    if spec is None:
        return False
    names = {n for n, _ in spec} - {"vo"}        # vorticity is computed from u/v when a source lacks it
    if "vo" in {n for n, _ in spec}:
        names |= {"u", "v"}
    return names <= SOURCE_FIELDS[MODEL["source"]]


def model_params() -> list:
    """Product ids this model can render."""
    if MODEL["params"]:
        return list(MODEL["params"])
    return [pid for pid in PARAMS if supported(pid)]


def param_hours(pid: str) -> list:
    mh = PARAMS[pid].get("max_hour")
    return [h for h in FORECAST_HOURS if mh is None or h <= mh]

# How many runs to keep. Only meaningful when images persist between jobs
# (R2 storage); a plain Pages deploy only ever contains the run just rendered.
KEEP_RUNS = 8

# NOMADS (GFS)
NOMADS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
NOMADS_DIR = "/gfs.{ymd}/{hh}/atmos"
NOMADS_FILE = "gfs.t{hh}z.pgrb2.0p25.f{fhr:03d}"
NOMADS_IDX = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f000.idx"

# -------------------------------------------------------------- regions -----
# lon/lat bounding box (lon in -180..180). Padding is added on fetch so
# contours don't get clipped at the frame edge.
REGIONS = {
    "conus": {"name": "United States", "bbox": (-126, -66, 23, 50)},
    "natl":  {"name": "North Atlantic", "bbox": (-100, -10, 5, 45)},
    "epac":  {"name": "East Pacific",   "bbox": (-150, -85, 3, 35)},
    "namer": {"name": "North America",  "bbox": (-140, -50, 12, 62)},
    "gulf":  {"name": "Gulf of Mexico",  "bbox": (-100, -74, 16, 33)},
    "fl":    {"name": "Florida",         "bbox": (-88.5, -77.5, 23.5, 31.5)},
    "carib": {"name": "Caribbean",       "bbox": (-92, -55, 7, 28)},
}

# ----------------------------------------------------------- parameters -----
# `fetch`: NOMADS grib_filter (VAR, LEVEL) pairs for GFS. `spec`: the generic
# (field, level) list used by every other model via the translation tables in
# fetch.py (ECMWF open data, CMC Datamart, DWD ICON), or None if GFS-only.
# `prev`: extra fields needed from earlier forecast hours — offsets in hours,
# or "f0" for the run's hour 0. `max_hour`: render only this far (keeps the
# site under the Pages size limit); default = model's full range.
# Field names the plot functions see are normalised in fetch.py so one plot
# function serves every model.

_MSLP = [("PRMSL", "mean_sea_level")]
_E_MSLP = [("msl", None)]
_PTYPE = [("CSNOW", "surface"), ("CICEP", "surface"), ("CFRZR", "surface"), ("CRAIN", "surface")]

PARAMS = {
    # ------------------------------------------------------ precipitation ---
    "mslp_precip": {
        "name": "MSLP & 6-hr precip", "group": "Precipitation", "plot": "plot_mslp_precip",
        "fetch": _MSLP + [("APCP", "surface"), ("HGT", "1000_mb"), ("HGT", "500_mb")],
        "spec": _E_MSLP + [("tp", None), ("gh", 1000), ("gh", 500)],
        "prev": {"offsets": [6], "fetch": [], "spec": [("tp", None)]},
    },
    "mslp_ptype": {
        "name": "MSLP & 6-hr precip (rain / frozen)", "group": "Precipitation", "plot": "plot_mslp_ptype",
        "fetch": _MSLP + [("APCP", "surface")] + _PTYPE, "spec": None
    },
    "refc": {
        "name": "Simulated radar (rain / frozen)", "group": "Precipitation", "plot": "plot_refc",
        "fetch": _MSLP + [("REFC", "entire_atmosphere")] + _PTYPE, "spec": None
    },
    "precip24": {
        "name": "24-hr accumulated precip", "group": "Precipitation", "plot": "plot_precip24",
        "fetch": _MSLP + [("APCP", "surface")], "spec": _E_MSLP + [("tp", None)],
        "prev": {"offsets": [24], "fetch": [("APCP", "surface")], "spec": [("tp", None)]},
    },
    "precip_total": {
        "name": "Total accumulated precip", "group": "Precipitation", "plot": "plot_precip_total",
        "fetch": _MSLP + [("APCP", "surface")], "spec": _E_MSLP + [("tp", None)],
    },
    "snow24": {
        "name": "24-hr snowfall (10:1)", "group": "Precipitation", "plot": "plot_snow24",
        "fetch": _MSLP + [("APCP", "surface"), ("CSNOW", "surface")], "spec": None,
        "prev": {"offsets": [6, 12, 18], "fetch": [("APCP", "surface"), ("CSNOW", "surface")], "spec": []},
    },
    "snod_total": {
        "name": "Total snow-depth change", "group": "Precipitation", "plot": "plot_snod_total",
        "fetch": _MSLP + [("SNOD", "surface")], "spec": _E_MSLP + [("snod", None)],
        "prev": {"offsets": ["f0"], "fetch": [("SNOD", "surface")], "spec": []},
    },
    "snod24": {
        "name": "24-hr snow-depth change", "group": "Precipitation", "plot": "plot_snod24",
        "fetch": _MSLP + [("SNOD", "surface")], "spec": _E_MSLP + [("snod", None)],
        "prev": {"offsets": [24], "fetch": [("SNOD", "surface")], "spec": []},
    },
    "pwat": {
        "name": "MSLP & precipitable water", "group": "Precipitation", "plot": "plot_pwat",
        "fetch": _MSLP + [("PWAT", "entire_atmosphere_\\(considered_as_a_single_layer\\)")],
        "spec": _E_MSLP + [("tcwv", None)],
    },
    "rh700_300": {
        "name": "700–300 mb relative humidity", "group": "Precipitation", "plot": "plot_rh700_300",
        "fetch": [("RH", "700_mb"), ("RH", "500_mb"), ("RH", "300_mb"), ("HGT", "500_mb")],
        "spec": [("r", 700), ("r", 500), ("r", 300), ("gh", 500)]
    },
    # ------------------------------------------------------ upper dynamics --
    "z500_vort": {
        "name": "500 mb height, vorticity & wind", "group": "Upper dynamics", "plot": "plot_z500_vort",
        "fetch": [("HGT", "500_mb"), ("ABSV", "500_mb"), ("UGRD", "500_mb"), ("VGRD", "500_mb")],
        "spec": [("gh", 500), ("vo", 500), ("u", 500), ("v", 500)],
    },
    "z500_mslp": {
        "name": "500 mb height & MSLP", "group": "Upper dynamics", "plot": "plot_z500_mslp",
        "fetch": _MSLP + [("HGT", "500_mb")], "spec": _E_MSLP + [("gh", 500)]
    },
    "z700_vort": {
        "name": "700 mb height, vorticity & wind", "group": "Upper dynamics", "plot": "plot_z700_vort",
        "fetch": [("HGT", "700_mb"), ("UGRD", "700_mb"), ("VGRD", "700_mb")],
        "spec": [("gh", 700), ("u", 700), ("v", 700)]
    },
    "z850_vort": {
        "name": "850 mb height, vorticity & wind", "group": "Upper dynamics", "plot": "plot_z850_vort",
        "fetch": [("HGT", "850_mb"), ("UGRD", "850_mb"), ("VGRD", "850_mb")],
        "spec": [("gh", 850), ("u", 850), ("v", 850)]
    },
    "z850_wind": {
        "name": "850 mb height & wind speed", "group": "Upper dynamics", "plot": "plot_z850_wind",
        "fetch": [("HGT", "850_mb"), ("UGRD", "850_mb"), ("VGRD", "850_mb")],
        "spec": [("gh", 850), ("u", 850), ("v", 850)]
    },
    "wind250": {
        "name": "250 mb wind & height", "group": "Upper dynamics", "plot": "plot_wind250",
        "fetch": [("HGT", "250_mb"), ("UGRD", "250_mb"), ("VGRD", "250_mb")],
        "spec": [("gh", 250), ("u", 250), ("v", 250)]
    },
    "pv2": {
        "name": "2 PVU pressure & wind", "group": "Upper dynamics", "plot": "plot_pv2",
        "fetch": [("PRES", "PV=2e-06_(Km^2/kg/s)_surface"), ("UGRD", "PV=2e-06_(Km^2/kg/s)_surface"),
                  ("VGRD", "PV=2e-06_(Km^2/kg/s)_surface")],
        "spec": None
    },
    "sim_ir": {
        "name": "Simulated IR satellite", "group": "Upper dynamics", "plot": "plot_sim_ir",
        "fetch": _MSLP + [("SBT124", "top_of_atmosphere")], "spec": None
    },
    "shear": {
        "name": "850–200 mb wind shear", "group": "Tropical", "plot": "plot_shear",
        "fetch": [("UGRD", "850_mb"), ("VGRD", "850_mb"), ("UGRD", "200_mb"), ("VGRD", "200_mb"), ("HGT", "500_mb")],
        "spec": [("u", 850), ("v", 850), ("u", 200), ("v", 200), ("gh", 500)]
    },
    "steering": {
        "name": "850–300 mb steering flow", "group": "Tropical", "plot": "plot_steering",
        "fetch": _MSLP + [("UGRD", "850_mb"), ("VGRD", "850_mb"), ("UGRD", "500_mb"), ("VGRD", "500_mb"), ("UGRD", "300_mb"), ("VGRD", "300_mb")],
        "spec": _E_MSLP + [("u", 850), ("v", 850), ("u", 500), ("v", 500), ("u", 300), ("v", 300)]
    },
    "div200": {
        "name": "200 mb divergence & wind", "group": "Tropical", "plot": "plot_div200",
        "fetch": [("UGRD", "200_mb"), ("VGRD", "200_mb"), ("HGT", "200_mb")],
        "spec": [("u", 200), ("v", 200), ("gh", 200)]
    },
    "rh700": {
        "name": "700 mb relative humidity & wind", "group": "Tropical", "plot": "plot_rh700",
        "fetch": [("RH", "700_mb"), ("UGRD", "700_mb"), ("VGRD", "700_mb"), ("HGT", "700_mb")],
        "spec": [("r", 700), ("u", 700), ("v", 700), ("gh", 700)]
    },
    "sst": {
        "name": "Sea surface temperature", "group": "Tropical", "plot": "plot_sst",
        "fetch": _MSLP + [("TMP", "surface"), ("LAND", "surface")],
        "spec": _E_MSLP + [("skt", None), ("lsm", None)]
    },
    "vort_layer": {
        "name": "850–500 mb layer vorticity & 700 mb wind", "group": "Tropical", "plot": "plot_vort_layer",
        "fetch": _MSLP + [("UGRD", "850_mb"), ("VGRD", "850_mb"), ("UGRD", "700_mb"), ("VGRD", "700_mb"), ("UGRD", "500_mb"), ("VGRD", "500_mb")],
        "spec": _E_MSLP + [("u", 850), ("v", 850), ("u", 700), ("v", 700), ("u", 500), ("v", 500)]
    },
    # ------------------------------------------------------ thermodynamics --
    "t2m": {
        "name": "2 m temperature", "group": "Thermodynamics", "plot": "plot_t2m",
        "fetch": _MSLP + [("TMP", "2_m_above_ground")], "spec": _E_MSLP + [("2t", None)],
    },
    "t850_wind": {
        "name": "850 mb temperature, wind & MSLP", "group": "Thermodynamics", "plot": "plot_t850_wind",
        "fetch": _MSLP + [("TMP", "850_mb"), ("UGRD", "850_mb"), ("VGRD", "850_mb"), ("HGT", "850_mb")],
        "spec": _E_MSLP + [("t", 850), ("u", 850), ("v", 850), ("gh", 850)],
    },
    "t700_wind": {
        "name": "700 mb temperature, wind & MSLP", "group": "Thermodynamics", "plot": "plot_t700_wind",
        "fetch": _MSLP + [("TMP", "700_mb"), ("UGRD", "700_mb"), ("VGRD", "700_mb"), ("HGT", "700_mb")],
        "spec": _E_MSLP + [("t", 700), ("u", 700), ("v", 700), ("gh", 700)]
    },
    "cape": {
        "name": "SBCAPE & wind crossovers", "group": "Thermodynamics", "plot": "plot_cape",
        "fetch": [("CAPE", "surface"), ("UGRD", "850_mb"), ("VGRD", "850_mb"), ("UGRD", "500_mb"), ("VGRD", "500_mb")],
        "spec": [("cape", None), ("u", 850), ("v", 850), ("u", 500), ("v", 500)],
    },
    # ------------------------------------------------------ surface ---------
    "wind10m": {
        "name": "MSLP & 10 m wind", "group": "Surface", "plot": "plot_wind10m",
        "fetch": _MSLP + [("UGRD", "10_m_above_ground"), ("VGRD", "10_m_above_ground")],
        "spec": _E_MSLP + [("10u", None), ("10v", None)],
    },
    # ------------------------------------------------------ diagnostics -----
    "fgen700": {
        "name": "700 mb temp advection & frontogenesis", "group": "Diagnostics", "plot": "plot_fgen700",
        "fetch": [("TMP", "700_mb"), ("UGRD", "700_mb"), ("VGRD", "700_mb"), ("HGT", "700_mb")],
        "spec": [("t", 700), ("u", 700), ("v", 700), ("gh", 700)]
    },
    "fgen850": {
        "name": "850 mb temp advection & frontogenesis", "group": "Diagnostics", "plot": "plot_fgen850",
        "fetch": [("TMP", "850_mb"), ("UGRD", "850_mb"), ("VGRD", "850_mb"), ("HGT", "850_mb")],
        "spec": [("t", 850), ("u", 850), ("v", 850), ("gh", 850)]
    },
    "okubo850": {
        "name": "850 mb Okubo-Weiss & dilatation axes", "group": "Diagnostics", "plot": "plot_okubo850",
        "fetch": [("HGT", "850_mb"), ("UGRD", "850_mb"), ("VGRD", "850_mb")],
        "spec": [("gh", 850), ("u", 850), ("v", 850)]
    },
}

# Output image size (inches × dpi)
FIG_SIZE = (12, 8)
DPI = 100
