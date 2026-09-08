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
        "probe_max_hours": [360, 240],  # publish a 240-h run as soon as it's there; re-render to 360 when the rest lands
        "params": None,                 # None = every product in PARAMS
        "credit": "NOAA/NCEP GFS via NOMADS",
    },
    "ecmwf": {
        "id": "ecmwf",
        "name": "ECMWF",
        "resolution": "0.25°",
        "source": "ecmwf_opendata",
        "cycles": [0, 6, 12, 18],      # 06/18 are published with a shorter range; probed at run time
        "min_age_hours": 6.5,
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

# ---------------------------------------------------------------- ensembles --
MODELS["gefs"] = {
    "id": "gefs", "name": "GEFS", "resolution": "0.5°", "source": "gefs", "kind": "ensemble",
    "cycles": [0, 6, 12, 18], "min_age_hours": 5.5,
    "hours": list(range(0, 241, 6)),
    "members": ["c00"] + [f"p{i:02d}" for i in range(1, 31)],
    # one regional subset per member per hour covering every region we render
    "domain": (-150, -10, 0, 66),
    "params": None, "credit": "NOAA/NCEP GEFS via NOMADS",
}

MODELS["ecens"] = {
    "id": "ecens", "name": "ECMWF ENS", "resolution": "0.25°", "source": "ecmwf_ens", "kind": "ensemble",
    "cycles": [0, 6, 12, 18], "min_age_hours": 7.5,
    "hours": list(range(0, 241, 6)),
    "probe_max_hours": [240, 144],           # 06/18Z ENS runs are published to 144 h
    "members": ["c00"] + [f"p{i:02d}" for i in range(1, 51)],
    "domain": (-150, -10, 0, 66),
    "params": None, "credit": "ECMWF open data ENS (CC-BY-4.0)",
    "ens_fields": [("msl", None), ("gh", 500), ("t", 850), ("2t", None), ("10u", None), ("10v", None), ("tp", None)],
}

MODELS["aifsens"] = dict(MODELS["ecens"], id="aifsens", name="ECMWF AIFS ENS", source="ecmwf_aifs_ens",
                         min_age_hours=7.0, cycles=[0, 6, 12, 18], probe_max_hours=[240, 144],
                         hours=list(range(0, 241, 6)), credit="ECMWF open data AIFS-ENS (CC-BY-4.0)")
MODELS["aigefs"] = dict(MODELS["gefs"], id="aigefs", name="AI-GEFS", resolution="0.25°", source="aigefs", credit="NOAA/NCEP AIGEFS via NOMADS",
                        min_age_hours=4.0, hours=list(range(0, 241, 6)), probe_max_hours=[240, 120],
                        # NOMADS layout (no grib_filter): fields are byte-ranged out of each member's file via its .idx
                        path="https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigefs/v1.0/aigefs.{ymd}/{hh}/mem{mem:03d}/model/atmos/grib2/aigefs.t{hh}z.pres.f{fhr:03d}.grib2",
                        idx_fields=[("PRMSL", "mean sea level"), ("HGT", "500 mb"), ("TMP", "850 mb"), ("TMP", "2 m above ground"),
                                    ("UGRD", "10 m above ground"), ("VGRD", "10 m above ground"), ("APCP", "surface")])

MODELS["geps"] = {
    "id": "geps", "name": "GEPS", "resolution": "0.5°", "source": "geps", "kind": "ensemble",
    "cycles": [0, 12], "min_age_hours": 6.5,
    "hours": list(range(0, 241, 6)),
    "members": ["c00"] + [f"p{i:02d}" for i in range(1, 21)],
    "domain": (-150, -10, 0, 66),
    "params": None, "credit": "Environment and Climate Change Canada GEPS (MSC Datamart)",
    "ens_fields": [("msl", None), ("gh", 500), ("t", 850), ("2t", None), ("10u", None), ("10v", None), ("tp", None)],
}

# ---------------------------------------------------------------- mesoscale --
# NOAA CONUS models on NOMADS grib_filter. Their grids are Lambert conformal, so
# fetch.load_grib regrids them to lat/lon on the way in. No subregion (grib_filter
# only subsets lat/lon grids), so whole-CONUS files are downloaded per hour.
_MESO_PARAMS = ["mslp_precip", "mslp_ptype", "refc", "precip24", "precip_total", "t2m", "wind10m", "gust", "cape", "pwat",
                "t850_wind", "rh700", "z500_vort", "z500_mslp", "shear", "steering"]
# the 3-km models: only products where the resolution earns its cost (upper air comes from NAM 12 km / GFS)
_HIRES_PARAMS = ["mslp_precip", "mslp_ptype", "refc", "precip24", "precip_total", "t2m", "wind10m", "gust", "cape", "pwat"]
MODELS["hrrr"] = {
    "id": "hrrr", "name": "HRRR", "resolution": "3 km", "source": "nomads_grid", "kind": "mesoscale",
    "filter": "filter_hrrr_2d.pl", "dir": "/hrrr.{ymd}/conus", "file": "hrrr.t{hh}z.wrfsfcf{fhr:02d}.grib2",
    "idx": "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{ymd}/conus/hrrr.t{hh}z.wrfsfcf{fhr:02d}.grib2.idx",
    "cycles": list(range(24)), "min_age_hours": 1.75,          # every hourly run; 00/06/12/18 reach 48 h, others 18 h
    "hours": list(range(0, 49, 1)), "probe_max_hours": [48, 18],
    # HRRR's surface file has no 700 mb RH or 200 mb wind (per its .idx), so no rh700 / shear
    "regions": ["conus", "seast", "gulf", "fl"], "params": _HIRES_PARAMS, "grid_res": 0.05, "workers": 3,
    "credit": "NOAA/NCEP HRRR via NOMADS",
}
MODELS["nam"] = {
    "id": "nam", "name": "NAM 12 km", "resolution": "12 km", "source": "nomads_grid", "kind": "mesoscale",
    "filter": "filter_nam.pl", "dir": "/nam.{ymd}", "file": "nam.t{hh}z.awphys{fhr:02d}.tm00.grib2",
    "idx": "https://nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod/nam.{ymd}/nam.t{hh}z.awphys{fhr:02d}.tm00.grib2.idx",
    "cycles": [0, 6, 12, 18], "min_age_hours": 2.5,
    "hours": list(range(0, 37, 1)) + list(range(39, 85, 3)), "probe_max_hours": [84],
    "regions": ["conus", "seast", "gulf", "fl"], "params": _MESO_PARAMS, "grid_res": 0.1,
    "credit": "NOAA/NCEP NAM via NOMADS",
}
MODELS["namnest"] = {
    "id": "namnest", "name": "NAM 3 km nest", "resolution": "3 km", "source": "nomads_grid", "kind": "mesoscale",
    "filter": "filter_nam_conusnest.pl", "dir": "/nam.{ymd}", "file": "nam.t{hh}z.conusnest.hiresf{fhr:02d}.tm00.grib2",
    "idx": "https://nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod/nam.{ymd}/nam.t{hh}z.conusnest.hiresf{fhr:02d}.tm00.grib2.idx",
    "cycles": [0, 6, 12, 18], "min_age_hours": 2.5,
    "hours": list(range(0, 61, 1)), "probe_max_hours": [60],
    "regions": ["conus", "seast", "gulf", "fl"], "params": _HIRES_PARAMS, "grid_res": 0.05, "workers": 3,
    "credit": "NOAA/NCEP NAM CONUS nest via NOMADS",
}
MODELS["nbm"] = {
    "id": "nbm", "name": "National Blend (NBM)", "resolution": "2.5 km", "source": "nomads_grid", "kind": "mesoscale",
    "filter": "filter_blend.pl", "dir": "/blend.{ymd}/{hh}/core", "file": "blend.t{hh}z.core.f{fhr:03d}.co.grib2",
    "idx": "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/blend.{ymd}/{hh}/core/blend.t{hh}z.core.f{fhr:03d}.co.grib2.idx",
    "cycles": [1, 7, 13, 19], "min_age_hours": 1.5,
    "hours": list(range(1, 37, 1)) + list(range(39, 193, 3)), "probe_max_hours": [192, 36],
    "regions": ["conus", "seast", "gulf", "fl"], "params": ["t2m", "wind10m", "precip6", "gust"], "grid_res": 0.05, "workers": 3,
    "credit": "NOAA/NWS National Blend of Models via NOMADS",
}

MODEL = MODELS[os.environ.get("WX_MODEL", "gfs").lower()]
# Several models can share one Pages site (e.g. two ensembles in one repo): give
# each its own manifest file name via WX_MANIFEST.
MANIFEST_NAME = os.environ.get("WX_MANIFEST", "manifest.json")
FORECAST_HOURS = MODEL["hours"]


# Which generic field names each non-GFS source can supply (see fetch.py tables).
SOURCE_FIELDS = {
    "ecmwf_opendata": {"gh", "t", "u", "v", "r", "msl", "tp", "2t", "10u", "10v", "tcwv", "vo", "skt", "lsm"},
    "cmc":            {"gh", "t", "u", "v", "r", "msl", "tp", "2t", "10u", "10v", "vo", "cape", "snod", "skt", "lsm"},
    "icon":           {"gh", "t", "u", "v", "r", "msl", "tp", "2t", "10u", "10v", "tcwv", "cape", "snod", "skt", "lsm"},
}


def supported(pid: str) -> bool:
    if MODEL["source"] in ("nomads", "nomads_grid"):
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
    if MODEL.get("kind") == "ensemble":
        return list(ENS_PARAMS)
    return [pid for pid in PARAMS if supported(pid)]


def products() -> dict:
    """The product table for this model (deterministic or ensemble)."""
    return ENS_PARAMS if MODEL.get("kind") == "ensemble" else PARAMS


def param_hours(pid: str) -> list:
    mh = products()[pid].get("max_hour")
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
    "seast": {"name": "Southeast US",    "bbox": (-95, -74, 23.5, 37.5)},
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

_MSLP = [("PRMSL", "mean_sea_level"), ("MSLMA", "mean_sea_level")]   # HRRR/NAM publish MSLMA; absent pairs are dropped
_E_MSLP = [("msl", None)]
_PTYPE = [("CSNOW", "surface"), ("CICEP", "surface"), ("CFRZR", "surface"), ("CRAIN", "surface")]

PARAMS = {
    # ------------------------------------------------------ precipitation ---
    "mslp_precip": {
        "name": "MSLP & 6-hr precip", "group": "Precipitation", "plot": "plot_mslp_precip",
        "fetch": _MSLP + [("APCP", "surface"), ("HGT", "1000_mb"), ("HGT", "500_mb")],
        "spec": _E_MSLP + [("tp", None), ("gh", 1000), ("gh", 500)],
        "prev": {"offsets": [6], "fetch": [("APCP", "surface")], "spec": [("tp", None)]},
    },
    "mslp_ptype": {
        "name": "MSLP & 6-hr precip (rain / frozen)", "group": "Precipitation", "plot": "plot_mslp_ptype",
        "fetch": _MSLP + [("APCP", "surface")] + _PTYPE, "spec": None
    },
    "refc": {
        "name": "Simulated radar (rain / frozen)", "group": "Precipitation", "plot": "plot_refc",
        "fetch": _MSLP + [("REFC", "entire_atmosphere")] + _PTYPE, "spec": None
    },
    "precip6": {
        "name": "6-hr precipitation", "group": "Precipitation", "plot": "plot_precip6",
        "fetch": [("APCP", "surface")], "spec": None,
        "prev": {"offsets": [6], "fetch": [("APCP", "surface")], "spec": []},
    },
    "gust": {
        "name": "10 m wind gust", "group": "Surface", "plot": "plot_gust",
        "fetch": [("GUST", "10_m_above_ground"), ("GUST", "surface"), ("UGRD", "10_m_above_ground"), ("VGRD", "10_m_above_ground"),
                  ("WIND", "10_m_above_ground"), ("WDIR", "10_m_above_ground")], "spec": None,
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
    # "sim_ir": simulated IR brightness temperature (SBT124). Not present in NOAA's
    # 0.25° GFS files (verified from the .idx listings, Sep 2026); kept out until it is.
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
        "fetch": _MSLP + [("UGRD", "10_m_above_ground"), ("VGRD", "10_m_above_ground"),
                          ("WIND", "10_m_above_ground"), ("WDIR", "10_m_above_ground")],   # NBM: speed + direction
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

# ----------------------------------------------------- ensemble products -----
# Computed from the member stack. `fetch` = NOMADS grib_filter pairs fetched for
# EVERY member; all products share one download per member per hour.
_ENS_FETCH = [("HGT", "500_mb"), ("PRMSL", "mean_sea_level"), ("TMP", "850_mb"), ("TMP", "2_m_above_ground"),
              ("UGRD", "10_m_above_ground"), ("VGRD", "10_m_above_ground"), ("APCP", "surface")]
ENS_PARAMS = {
    "ens_mslp":      {"name": "MSLP mean & spread",           "group": "Mean & spread", "plot": "ens_mslp",      "fetch": _ENS_FETCH},
    "ens_z500":      {"name": "500 mb height mean & spread",  "group": "Mean & spread", "plot": "ens_z500",      "fetch": _ENS_FETCH},
    "ens_t850":      {"name": "850 mb temp mean & spread",    "group": "Mean & spread", "plot": "ens_t850",      "fetch": _ENS_FETCH},
    "ens_t2m":       {"name": "2 m temp mean & spread",       "group": "Mean & spread", "plot": "ens_t2m",       "fetch": _ENS_FETCH},
    "ens_precip6":   {"name": "6-hr precip mean",             "group": "Mean & spread", "plot": "ens_precip6",   "fetch": _ENS_FETCH},
    "ens_lows":      {"name": "Member low centres",           "group": "Mean & spread", "plot": "ens_lows",      "fetch": _ENS_FETCH},
    "spag_z500":     {"name": "500 mb spaghetti (564, 582 dam)", "group": "Spaghetti",  "plot": "spag_z500",     "fetch": _ENS_FETCH},
    "spag_mslp":     {"name": "MSLP spaghetti (1000, 1012 mb)", "group": "Spaghetti",   "plot": "spag_mslp",     "fetch": _ENS_FETCH},
    "prob_wind34":   {"name": "Prob. 10 m wind ≥ 34 kt",      "group": "Probability",   "plot": "prob_wind34",   "fetch": _ENS_FETCH},
    "prob_wind50":   {"name": "Prob. 10 m wind ≥ 50 kt",      "group": "Probability",   "plot": "prob_wind50",   "fetch": _ENS_FETCH},
    "prob_wind64":   {"name": "Prob. 10 m wind ≥ 64 kt",      "group": "Probability",   "plot": "prob_wind64",   "fetch": _ENS_FETCH},
    "prob_precip05": {"name": "Prob. 6-hr precip ≥ 0.5 in",   "group": "Probability",   "plot": "prob_precip05", "fetch": _ENS_FETCH},
    "prob_precip1":  {"name": "Prob. 6-hr precip ≥ 1 in",     "group": "Probability",   "plot": "prob_precip1",  "fetch": _ENS_FETCH},
    "prob_mslp1000": {"name": "Prob. MSLP ≤ 1000 mb",         "group": "Probability",   "plot": "prob_mslp1000", "fetch": _ENS_FETCH},
    "prob_t850frz":  {"name": "Prob. 850 mb temp ≤ 0 °C",     "group": "Probability",   "plot": "prob_t850frz",  "fetch": _ENS_FETCH},
}

# Cities whose values get printed on the 2 m temperature and 10 m wind maps
# (only on the Florida / Gulf / Southeast views, where the labels fit).
LABEL_CITIES = [
    ("Miami", 25.76, -80.19), ("West Palm Beach", 26.71, -80.05), ("Key West", 24.56, -81.78),
    ("Naples", 26.14, -81.79), ("Fort Myers", 26.64, -81.87), ("Sarasota", 27.34, -82.53),
    ("Tampa", 27.95, -82.46), ("Orlando", 28.54, -81.38), ("Daytona Beach", 29.21, -81.02), ("Jacksonville", 30.33, -81.66),
    ("Tallahassee", 30.44, -84.28), ("Panama City", 30.16, -85.66), ("Pensacola", 30.42, -87.22),
    ("Mobile", 30.69, -88.04), ("New Orleans", 29.95, -90.07), ("Houston", 29.76, -95.37), ("Corpus Christi", 27.80, -97.40),
    ("Brownsville", 25.90, -97.50), ("Atlanta", 33.75, -84.39), ("Savannah", 32.08, -81.10), ("Charleston", 32.78, -79.93),
    ("Nassau", 25.05, -77.35), ("Havana", 23.13, -82.38), ("Cancún", 21.16, -86.85),
]
LABEL_REGIONS = {"fl", "gulf", "seast"}

# Output image size (inches × dpi)
FIG_SIZE = (12, 8)
DPI = 100
