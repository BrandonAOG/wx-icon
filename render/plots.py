"""
One plotting function per parameter. Each takes (fields, meta) and returns a
matplotlib Figure. `meta` carries region bbox, run time, forecast hour, etc.

Style notes: Lambert Conformal for mid-latitude regions, Plate Carrée near the
tropics; heights in dam, MSLP in hPa every 4, temperatures in °F/°C as labelled.
"""
from __future__ import annotations

import datetime as dt
import logging

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches
import matplotlib.patheffects
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.ndimage import gaussian_filter

from config import DPI, FIG_SIZE, LABEL_CITIES, LABEL_REGIONS, MODEL, SITE_NAME

log = logging.getLogger("plots")
PC = ccrs.PlateCarree()


# ---------------------------------------------------------------- helpers ---

def pick(f, *keys):
    for k in keys:
        if k in f:
            return f[k]
    raise KeyError(f"none of {keys} in fields {list(f)}")


def projection_for(bbox):
    lon0, lon1, lat0, lat1 = bbox
    if lat0 >= 0 and lat1 <= 40:      # tropics: keep it flat
        return PC
    return ccrs.LambertConformal(central_longitude=(lon0 + lon1) / 2,
                                 central_latitude=(lat0 + lat1) / 2,
                                 standard_parallels=(lat0 + 5, lat1 - 5))


CURRENT = {"small": False}     # set by new_map: is this a small region (tighter contour intervals)?


def small_region(meta) -> bool:
    lon0, lon1, lat0, lat1 = meta["bbox"]
    return (lon1 - lon0) <= 16 or (lat1 - lat0) <= 10


def mslp_levels():
    return np.arange(940, 1060, 2) if CURRENT["small"] else np.arange(940, 1060, 4)


def z500_levels():
    return np.arange(480, 620, 3) if CURRENT["small"] else np.arange(480, 620, 6)


def new_map(meta):
    CURRENT["small"] = small_region(meta)
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    proj = projection_for(meta["bbox"])
    ax = fig.add_axes([0.01, 0.08, 0.98, 0.825], projection=proj)
    ax.set_extent(meta["bbox"], crs=PC)
    return fig, ax


_BASEMAP = None


def _basemap_layers():
    """Load coastline/border/state geometries once per process. cartopy
    re-reads the shapefiles for every new Feature object, which dominated
    plot time. A missing Natural Earth download (no network) just skips the layer."""
    global _BASEMAP
    if _BASEMAP is None:
        _BASEMAP = []
        for feat, lw, col in [(cfeature.COASTLINE, 0.8, "#222"), (cfeature.BORDERS, 0.6, "#333"),
                              (cfeature.STATES, 0.4, "#555")]:
            try:
                geoms = list(feat.with_scale("50m").geometries())
                _BASEMAP.append((cfeature.ShapelyFeature(geoms, PC), lw, col))
            except Exception as e:  # noqa: BLE001
                log.warning("basemap layer unavailable (%s); skipping", str(e)[:60])
    return _BASEMAP


def add_basemap(ax):
    for feat, lw, col in _basemap_layers():
        ax.add_feature(feat, lw=lw, edgecolor=col, facecolor="none", zorder=5)
    gl = ax.gridlines(draw_labels=False, lw=0.3, color="#888", alpha=0.5, linestyle=":")
    gl.xlocator = matplotlib.ticker.MultipleLocator(10)
    gl.ylocator = matplotlib.ticker.MultipleLocator(10)


def title(fig, ax, meta, left, right_units=""):
    valid = meta["run"] + dt.timedelta(hours=meta["fhr"])
    fig.text(0.01, 0.972, f"{MODEL['name']} {MODEL['resolution']}  |  {left}",
             fontsize=13, fontweight="bold", ha="left", va="center")
    fig.text(0.01, 0.932,
             f"Init: {meta['run']:%a %d %b %Y %HZ}     Forecast hour {meta['fhr']:03d}     "
             f"Valid: {valid:%a %d %b %Y %HZ}",
             fontsize=10.5, ha="left", va="center", color="#333")
    fig.text(0.01, 0.015, f"{SITE_NAME}  ·  data: {MODEL.get('credit', '')}  ·  {meta['region_name']}",
             fontsize=8.5, ha="left", va="center", color="#666")
    if right_units:
        fig.text(0.99, 0.015, right_units, fontsize=8.5, ha="right", va="center", color="#666")


def colorbar(fig, mappable, label, ticks=None):
    cax = fig.add_axes([0.25, 0.045, 0.5, 0.014])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
    cb.ax.tick_params(labelsize=8, length=2, pad=1)
    cb.set_label(label, fontsize=8.5, labelpad=2)
    return cb


def contour_labeled(ax, lon, lat, data, levels, color, lw, fmt="%d", linestyles="solid", zorder=6):
    cs = ax.contour(lon, lat, data, levels=levels, colors=color, linewidths=lw,
                    linestyles=linestyles, transform=PC, zorder=zorder)
    ax.clabel(cs, fmt=fmt, fontsize=7, inline=True, inline_spacing=2)
    return cs


def hilo(ax, lon, lat, mslp_hpa, size=40):
    """Mark pressure highs and lows."""
    from scipy.ndimage import maximum_filter, minimum_filter
    mx = maximum_filter(mslp_hpa, size)
    mn = minimum_filter(mslp_hpa, size)
    LON, LAT = np.meshgrid(lon, lat)
    for mask, sym, col in [(mslp_hpa == mx, "H", "#1848a8"), (mslp_hpa == mn, "L", "#c81e1e")]:
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            if 2 < y < len(lat) - 3 and 2 < x < len(lon) - 3:
                ax.text(LON[y, x], LAT[y, x], sym, color=col, fontsize=15, fontweight="bold",
                        ha="center", va="center", transform=PC, zorder=8,
                        path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground="w")])
                ax.text(LON[y, x], LAT[y, x] - 1.2, f"{mslp_hpa[y, x]:.0f}", color=col, fontsize=7,
                        ha="center", va="top", transform=PC, zorder=8)


def barbs(ax, lon, lat, u, v, every=None, **kw):
    if every is None:
        every = max(1, len(lon) // 28)
    ax.barbs(lon[::every], lat[::every], u[::every, ::every], v[::every, ::every],
             length=5, linewidth=0.5, transform=PC, zorder=7, **kw)


def smooth(a, s=1.5):
    return gaussian_filter(a, s)



def refine(lon, lat, *arrays, target=360, order=1):
    """Upsample fields so shaded products don't show raw 0.25° cells on small
    regions. Factor adapts to the grid: ~360 columns after refinement.
    order=1 bilinear for continuous fields; pass order=0 for masks."""
    from scipy.ndimage import zoom
    factor = int(max(1, min(8, round(target / max(len(lon), 1)))))
    if factor == 1:
        return (lon, lat) + tuple(arrays)
    lon2 = np.linspace(lon[0], lon[-1], (len(lon) - 1) * factor + 1)
    lat2 = np.linspace(lat[0], lat[-1], (len(lat) - 1) * factor + 1)
    out = []
    for a in arrays:
        a = np.asarray(a, dtype=float)
        z = zoom(np.nan_to_num(a, nan=0.0), factor, order=order, mode="nearest", grid_mode=True)
        out.append(z[:len(lat2), :len(lon2)])
    return (lon2, lat2) + tuple(out)


def mesh(ax, lon, lat, data, mask_below=None, mask=None, **kw):
    """pcolormesh with adaptive upsampling. mask_below hides values under a
    threshold; mask (bool, same grid as data) hides True cells (nearest-neighbour)."""
    arrays = [data] + ([mask.astype(float)] if mask is not None else [])
    res = refine(lon, lat, *arrays)
    lon2, lat2, d2 = res[0], res[1], res[2]
    if mask is not None:
        m2 = refine(lon, lat, mask.astype(float), order=0)[2] > 0.5
        d2 = np.ma.masked_where(m2, d2)
    if mask_below is not None:
        d2 = np.ma.masked_less(d2, mask_below)
    kw.setdefault("shading", "auto")
    return ax.pcolormesh(lon2, lat2, d2, **kw)


# ------------------------------------------------------------- parameters ---

def plot_z500_vort(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    z = smooth(pick(f, "gh500") / 10)           # dam
    vort = pick(f, "absv500", "absv") * 1e5     # 1e-5 s^-1
    levels = np.arange(8, 60, 2)
    cf = ax.contourf(lon, lat, vort, levels=levels, cmap=VORT_CMAP, extend="max", transform=PC, zorder=2)
    contour_labeled(ax, lon, lat, z, z500_levels(), "black", 1.0)
    if "u500" in f:
        barbs(ax, lon, lat, pick(f, "u500") * 1.944, pick(f, "v500") * 1.944, color="#333")
    add_basemap(ax)
    colorbar(fig, cf, "Absolute vorticity (10⁻⁵ s⁻¹)", ticks=levels[::2])
    title(fig, ax, meta, "500 mb height (dam), absolute vorticity & wind (kt)")
    return fig


def _precip_cmap():
    bounds = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8]
    colors = ["#c9e8f5", "#8fd1ee", "#4fb2e0", "#1d81c8", "#1c4fa5", "#3ec24d", "#1e8f2a",
              "#f7e530", "#f7a020", "#ea3b1a", "#b41313", "#7a0f5e"]
    return mcolors.ListedColormap(colors), mcolors.BoundaryNorm(bounds, len(colors)), bounds


def plot_mslp_precip(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    mslp = smooth(pick(f, "prmsl") / 100)
    cmap, norm, bounds = _precip_cmap()
    if "tp_6" in f:
        precip_in = pick(f, "tp_6") / 25.4
        cf = mesh(ax, lon, lat, precip_in, 0.01, cmap=cmap, norm=norm, transform=PC, zorder=2)
        colorbar(fig, cf, "6-hr precipitation (in)", ticks=bounds)
    else:
        ax.text(0.5, 0.5, "No accumulated precipitation at hour 000", transform=ax.transAxes,
                ha="center", fontsize=11, color="#666", zorder=9)
    if "gh1000" in f and "gh500" in f:
        thk = smooth((pick(f, "gh500") - pick(f, "gh1000")) / 10)
        ax.contour(lon, lat, thk, levels=np.arange(480, 540, 6), colors="#1a4fd6", linewidths=0.8,
                   linestyles="dashed", transform=PC, zorder=5)
        ax.contour(lon, lat, thk, levels=[540], colors="#1a4fd6", linewidths=1.6,
                   linestyles="dashed", transform=PC, zorder=5)
        ax.contour(lon, lat, thk, levels=np.arange(546, 600, 6), colors="#d62828", linewidths=0.8,
                   linestyles="dashed", transform=PC, zorder=5)
    contour_labeled(ax, lon, lat, mslp, mslp_levels(), "black", 1.0)
    hilo(ax, lon, lat, mslp)
    add_basemap(ax)
    title(fig, ax, meta, "MSLP (mb), 1000–500 mb thickness (dam) & 6-hr precipitation (in)")
    return fig


def plot_t850_wind(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    t = pick(f, "t850") - 273.15
    levels = np.arange(-30, 36, 2)
    cf = ax.contourf(lon, lat, t, levels=levels, cmap="RdYlBu_r", extend="both", transform=PC, zorder=2)
    ax.contour(lon, lat, t, levels=[0], colors="k", linewidths=1.2, linestyles="dashed", transform=PC, zorder=4)
    if "prmsl" in f:
        contour_labeled(ax, lon, lat, smooth(pick(f, "prmsl") / 100), mslp_levels(), "black", 0.8)
    barbs(ax, lon, lat, pick(f, "u850") * 1.944, pick(f, "v850") * 1.944, color="#222")
    add_basemap(ax)
    colorbar(fig, cf, "850 mb temperature (°C)", ticks=levels[::3])
    title(fig, ax, meta, "850 mb temperature (°C), wind (kt) & MSLP (mb)")
    return fig


def plot_t2m(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    tf = (pick(f, "t2m") - 273.15) * 9 / 5 + 32
    levels = np.arange(-30, 121, 5)
    cf = ax.contourf(lon, lat, tf, levels=levels, cmap="turbo", extend="both", transform=PC, zorder=2)
    ax.contour(lon, lat, tf, levels=[32], colors="k", linewidths=1.0, linestyles="dashed", transform=PC, zorder=4)
    step = 5 if meta.get("region") in LABEL_REGIONS else 10
    cs = ax.contour(lon, lat, smooth(tf, 1.0), levels=np.arange(-30, 125, step), colors="#222", linewidths=0.5, alpha=0.7, transform=PC, zorder=4)
    ax.clabel(cs, fmt="%d", fontsize=7, inline=True, inline_spacing=2)
    if "prmsl" in f:
        contour_labeled(ax, lon, lat, smooth(pick(f, "prmsl") / 100), mslp_levels(), "#555", 0.5, linestyles="dashed")
    f_tf = dict(f); f_tf["tf"] = tf
    ff = type("F", (dict,), {})(f_tf); ff.lon, ff.lat = lon, lat
    city_values(ax, ff, "tf", meta, fmt="{:.0f}°")
    add_basemap(ax)
    colorbar(fig, cf, "2 m temperature (°F) — contours every %d °F" % step, ticks=levels[::2])
    title(fig, ax, meta, "2 m temperature (°F) & MSLP (mb)")
    return fig


def plot_wind10m(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    u = pick(f, "u10") * 1.944
    v = pick(f, "v10") * 1.944
    spd = np.hypot(u, v)
    bounds = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100]
    colors = ["#cfe8ff", "#9dcbff", "#5aa7f5", "#2e7ad8", "#2bb673", "#7ed321", "#f8e71c",
              "#f5a623", "#f05a28", "#d0021b", "#9b0c3d", "#5e0a5e"]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, len(colors))
    cf = mesh(ax, lon, lat, spd, 10, cmap=cmap, norm=norm, transform=PC, zorder=2)
    barbs(ax, lon, lat, u, v, color="#222")
    if "prmsl" in f:
        mslp = smooth(pick(f, "prmsl") / 100)
        contour_labeled(ax, lon, lat, mslp, mslp_levels(), "black", 0.9)
        hilo(ax, lon, lat, mslp)
    fs = dict(f); fs["spd"] = spd
    ff = type("F", (dict,), {})(fs); ff.lon, ff.lat = lon, lat
    city_values(ax, ff, "spd", meta, fmt="{:.0f} kt")
    add_basemap(ax)
    colorbar(fig, cf, "10 m wind speed (kt)", ticks=bounds)
    title(fig, ax, meta, "MSLP (mb) & 10 m wind (kt)")
    return fig


def plot_pwat(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    pw = pick(f, "pwat")
    levels = np.arange(10, 75, 2.5)
    cmap = plt.get_cmap("gist_earth_r")
    cf = ax.contourf(lon, lat, pw, levels=levels, cmap=cmap, extend="both", transform=PC, zorder=2)
    if "prmsl" in f:
        contour_labeled(ax, lon, lat, smooth(pick(f, "prmsl") / 100), mslp_levels(), "white", 0.7)
    add_basemap(ax)
    colorbar(fig, cf, "Precipitable water (mm)", ticks=levels[::4])
    title(fig, ax, meta, "MSLP (mb) & precipitable water (mm)")
    return fig


def plot_cape(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    cape = pick(f, "cape")
    bounds = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000]
    colors = ["#e0f3db", "#a8ddb5", "#7bccc4", "#4eb3d3", "#2b8cbe", "#f7e530", "#f5a623",
              "#f05a28", "#d0021b", "#9b0c3d", "#5e0a5e"]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, len(colors))
    cf = mesh(ax, lon, lat, cape, 100, cmap=cmap, norm=norm, transform=PC, zorder=2)
    if "u850" in f:
        barbs(ax, lon, lat, pick(f, "u850") * 1.944, pick(f, "v850") * 1.944, color="#c81e1e")
    if "u500" in f:
        barbs(ax, lon, lat, pick(f, "u500") * 1.944, pick(f, "v500") * 1.944, color="#1848a8")
    add_basemap(ax)
    colorbar(fig, cf, "Surface-based CAPE (J/kg)", ticks=bounds)
    title(fig, ax, meta, "Surface-based CAPE (J/kg), 850 mb (red) & 500 mb (blue) wind (kt)")
    return fig


def city_values(ax, f, key, meta, fmt="{:.0f}", scale=1.0, offset=0.0, color="#111"):
    """Print a field's value at each labelled city (Florida/Gulf/Southeast views only)."""
    if meta.get("region") not in LABEL_REGIONS or key not in f:
        return
    lon0, lon1, lat0, lat1 = meta["bbox"]
    small = (lon1 - lon0) <= 16
    for name, la, lo in LABEL_CITIES:
        if not (lon0 + 0.4 <= lo <= lon1 - 0.4 and lat0 + 0.3 <= la <= lat1 - 0.3):
            continue
        j = int(np.argmin(np.abs(f.lat - la))); i = int(np.argmin(np.abs(f.lon - lo)))
        v = f[key][j, i]
        if not np.isfinite(v):
            continue
        txt = fmt.format(v * scale + offset)
        ax.plot(lo, la, "o", ms=3, color=color, mec="white", mew=0.6, transform=PC, zorder=9)
        ax.text(lo, la + (0.12 if small else 0.25), txt, fontsize=9 if small else 8, fontweight="bold", ha="center", va="bottom",
                color=color, transform=PC, zorder=9, path_effects=[matplotlib.patheffects.withStroke(linewidth=2.5, foreground="white")])
        if small:
            ax.text(lo, la - 0.12, name, fontsize=6.5, ha="center", va="top", color="#333", transform=PC, zorder=9,
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground="white")])


# ---------------------------------------------------- colour tables ---------

VORT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "vort", ["#ffffff", "#fff5c2", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026", "#7a0177", "#3f007d"])

# NWS-style reflectivity, 5 dBZ steps 5..75
REFC_BOUNDS = np.arange(5, 80, 5)
REFC_RAIN = mcolors.ListedColormap(["#04e9e7", "#019ff4", "#0300f4", "#02fd02", "#01c501", "#008e00", "#fdf802",
                                    "#e5bc00", "#fd9500", "#fd0000", "#d40000", "#bc0000", "#f800fd", "#9854c6"])
REFC_SNOW = mcolors.ListedColormap(["#e3ecf7", "#c4d6ee", "#a5c0e6", "#87aadd", "#6894d4", "#4a7ecb", "#2b68c2",
                                    "#1f53a6", "#173e8a", "#10296d", "#3a1f7a", "#5c2d91", "#7f3aa8", "#a247bf"])
REFC_ICE = mcolors.ListedColormap(["#f7e3f2", "#efc6e5", "#e7a9d8", "#df8ccb", "#d76fbe", "#cf52b1", "#c735a4",
                                   "#ad2b8f", "#93227a", "#791965", "#601050", "#46073b", "#33052b", "#20031b"])
REFC_FRZR = mcolors.ListedColormap(["#ffe6e6", "#ffcccc", "#ffb3b3", "#ff9999", "#ff8080", "#ff6666", "#ff4d4d",
                                    "#ff3333", "#ff1a1a", "#ff0000", "#e60000", "#cc0000", "#b30000", "#990000"])

SNOW_BOUNDS = [0.1, 1, 2, 3, 4, 6, 8, 12, 18, 24, 30, 36, 48]
SNOW_CMAP = mcolors.ListedColormap(["#e0f3ff", "#b8e2ff", "#8ecbff", "#5ba8f5", "#2d7fe0", "#1f5fc4", "#3b2ea8",
                                    "#6b2d9e", "#9b3aa0", "#c74ba0", "#e06aa9", "#f090c0"])
PRECIP_BIG_BOUNDS = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20]
PRECIP_BIG_CMAP = mcolors.ListedColormap(["#d9f0f7", "#a6dcee", "#66c2e0", "#2e9fd0", "#1f6fb5", "#3ec24d", "#1e8f2a",
                                          "#0f5c1a", "#f7e530", "#f7a020", "#ea3b1a", "#b41313", "#7a0f5e", "#4b0a3f",
                                          "#2a0524"])


# ---------------------------------------------------- diagnostics helpers ---

R_EARTH = 6.371e6


def grid_spacing(lon, lat):
    """dx, dy in metres as 2-D arrays for a regular lat/lon grid."""
    LON, LAT = np.meshgrid(lon, lat)
    dlon = np.gradient(LON, axis=1)
    dlat = np.gradient(LAT, axis=0)
    dx = np.radians(dlon) * R_EARTH * np.cos(np.radians(LAT))
    dy = np.radians(dlat) * R_EARTH
    return dx, dy


def ddx(a, dx):
    return np.gradient(a, axis=1) / dx


def ddy(a, dy):
    return np.gradient(a, axis=0) / dy


def rel_vort(u, v, lon, lat):
    dx, dy = grid_spacing(lon, lat)
    return ddx(v, dx) - ddy(u, dy)


def temp_advection(t, u, v, lon, lat):
    """-(V·∇T) in K/hr"""
    dx, dy = grid_spacing(lon, lat)
    return -(u * ddx(t, dx) + v * ddy(t, dy)) * 3600


def frontogenesis(t, u, v, lon, lat):
    """Petterssen 2-D frontogenesis in K / 100 km / 3 hr (positive = frontogenetic)."""
    dx, dy = grid_spacing(lon, lat)
    tx, ty = ddx(t, dx), ddy(t, dy)
    mag = np.hypot(tx, ty) + 1e-12
    ux, uy, vx, vy = ddx(u, dx), ddy(u, dy), ddx(v, dx), ddy(v, dy)
    F = -(tx ** 2 * ux + tx * ty * (vx + uy) + ty ** 2 * vy) / mag
    return F * 1e5 * 3 * 3600


def okubo_weiss(u, v, lon, lat):
    dx, dy = grid_spacing(lon, lat)
    ux, uy, vx, vy = ddx(u, dx), ddy(u, dy), ddx(v, dx), ddy(v, dy)
    s_n = ux - vy                  # normal strain
    s_s = vx + uy                  # shear strain
    omega = vx - uy                # vorticity
    ow = s_n ** 2 + s_s ** 2 - omega ** 2
    axis = 0.5 * np.arctan2(s_s, s_n)   # dilatation axis angle
    return ow, axis, np.hypot(s_n, s_s)


def mslp_contours(ax, f, color="black", lw=0.9, labels=True):
    if "prmsl" not in f:
        return None
    m = smooth(f["prmsl"] / 100)
    if labels:
        return contour_labeled(ax, f.lon, f.lat, m, mslp_levels(), color, lw)
    return ax.contour(f.lon, f.lat, m, levels=mslp_levels(), colors=color, linewidths=lw, transform=PC, zorder=6)


def ptype_masks(f):
    """Boolean masks for snow / sleet / freezing rain / rain from categorical fields."""
    z = np.zeros_like(next(iter(f.values())), dtype=bool)
    snow = f.get("csnow", z) > 0.5
    ice = (f.get("cicep", z) > 0.5) & ~snow
    frzr = (f.get("cfrzr", z) > 0.5) & ~snow & ~ice
    rain = ~(snow | ice | frzr)
    return snow, ice, frzr, rain


def ptype_legend(fig):
    handles = [matplotlib.patches.Patch(color=c, label=l) for c, l in
               [("#1e8f2a", "Rain"), ("#2b68c2", "Snow"), ("#cf52b1", "Sleet"), ("#ff4d4d", "Freezing rain")]]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99, 0.035), ncol=4, fontsize=8, frameon=False)


# ---------------------------------------------------- precipitation ---------

def plot_mslp_ptype(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    if "tp_6" in f:
        p = pick(f, "tp_6") / 25.4
        bounds = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6]
        snow, ice, frzr, rain = ptype_masks(f)
        for mask, cmap in [(rain, plt.get_cmap("Greens")), (snow, plt.get_cmap("Blues")),
                           (ice, plt.get_cmap("Purples")), (frzr, plt.get_cmap("Reds"))]:
            cm = mcolors.ListedColormap(cmap(np.linspace(0.25, 1, len(bounds) - 1)))
            norm = mcolors.BoundaryNorm(bounds, cm.N)
            mesh(ax, lon, lat, p, 0.01, mask=~mask, cmap=cm, norm=norm, transform=PC, zorder=2)
        cf = ax.pcolormesh(lon, lat, np.ma.masked_all(p.shape), cmap=mcolors.ListedColormap(plt.get_cmap("Greens")(np.linspace(0.25, 1, len(bounds) - 1))),
                           norm=mcolors.BoundaryNorm(bounds, len(bounds) - 1), transform=PC, zorder=1, shading="auto")
        colorbar(fig, cf, "6-hr precipitation (in) — colour = type", ticks=bounds)
        ptype_legend(fig)
    else:
        ax.text(0.5, 0.5, "No accumulated precipitation at hour 000", transform=ax.transAxes, ha="center", fontsize=11, color="#666", zorder=9)
    mslp_contours(ax, f)
    if "prmsl" in f:
        hilo(ax, lon, lat, smooth(f["prmsl"] / 100))
    add_basemap(ax)
    title(fig, ax, meta, "MSLP (mb) & 6-hr precipitation by type (in)")
    return fig


def plot_refc(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    refc = pick(f, "refc")
    snow, ice, frzr, rain = ptype_masks(f)
    norm = mcolors.BoundaryNorm(REFC_BOUNDS, 14)
    for mask, cmap in [(rain, REFC_RAIN), (snow, REFC_SNOW), (ice, REFC_ICE), (frzr, REFC_FRZR)]:
        mesh(ax, lon, lat, refc, 5, mask=~mask, cmap=cmap, norm=norm, transform=PC, zorder=2)
    cf = ax.pcolormesh(lon, lat, np.ma.masked_all(refc.shape), cmap=REFC_RAIN, norm=norm, transform=PC, zorder=1, shading="auto")
    mslp_contours(ax, f, lw=0.7)
    add_basemap(ax)
    colorbar(fig, cf, "Composite reflectivity (dBZ) — rain scale; snow blue, sleet purple, freezing rain red", ticks=REFC_BOUNDS[::2])
    ptype_legend(fig)
    title(fig, ax, meta, "Simulated composite reflectivity (dBZ) & MSLP (mb)")
    return fig


def _accum_plot(f, meta, key, label, title_txt):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    if key in f:
        p = f[key] / 25.4
        norm = mcolors.BoundaryNorm(PRECIP_BIG_BOUNDS, PRECIP_BIG_CMAP.N)
        cf = mesh(ax, lon, lat, p, 0.01, cmap=PRECIP_BIG_CMAP, norm=norm, transform=PC, zorder=2)
        colorbar(fig, cf, label, ticks=PRECIP_BIG_BOUNDS)
    else:
        ax.text(0.5, 0.5, "Not available at this hour", transform=ax.transAxes, ha="center", fontsize=11, color="#666", zorder=9)
    mslp_contours(ax, f, lw=0.7)
    add_basemap(ax)
    title(fig, ax, meta, title_txt)
    return fig


def plot_precip24(f, meta):
    return _accum_plot(f, meta, "tp_24", "24-hr precipitation (in)", "24-hr accumulated precipitation (in) & MSLP (mb)")


def plot_precip_total(f, meta):
    return _accum_plot(f, meta, "tp_acc", "Total precipitation since hour 0 (in)", "Total accumulated precipitation (in) & MSLP (mb)")


def plot_snow24(f, meta):
    """10:1 snowfall from the four 6-h buckets ending now, counting only buckets flagged as snow."""
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    total = None
    for tag in ("", "_m6", "_m12", "_m18"):
        if f"tp_6{tag}" in f and f"csnow{tag}" in f:
            contrib = f[f"tp_6{tag}"] * (f[f"csnow{tag}"] > 0.5)
            total = contrib if total is None else total + contrib
    if total is not None:
        inches = total / 25.4 * 10
        norm = mcolors.BoundaryNorm(SNOW_BOUNDS, SNOW_CMAP.N)
        cf = mesh(ax, lon, lat, inches, 0.1, cmap=SNOW_CMAP, norm=norm, transform=PC, zorder=2)
        colorbar(fig, cf, "24-hr snowfall, 10:1 ratio (in)", ticks=SNOW_BOUNDS)
    else:
        ax.text(0.5, 0.5, "Not available at this hour", transform=ax.transAxes, ha="center", fontsize=11, color="#666", zorder=9)
    mslp_contours(ax, f, lw=0.7)
    add_basemap(ax)
    title(fig, ax, meta, "24-hr snowfall, 10:1 ratio (in) & MSLP (mb)")
    return fig


def _snod_change(f, meta, other, label, title_txt):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    if "snod" in f and other in f:
        change = (f["snod"] - f[other]) * 39.37
        norm = mcolors.BoundaryNorm(SNOW_BOUNDS, SNOW_CMAP.N)
        cf = mesh(ax, lon, lat, change, 0.1, cmap=SNOW_CMAP, norm=norm, transform=PC, zorder=2)
        colorbar(fig, cf, label, ticks=SNOW_BOUNDS)
    else:
        ax.text(0.5, 0.5, "Not available at this hour", transform=ax.transAxes, ha="center", fontsize=11, color="#666", zorder=9)
    mslp_contours(ax, f, lw=0.7)
    add_basemap(ax)
    title(fig, ax, meta, title_txt)
    return fig


def plot_snod_total(f, meta):
    return _snod_change(f, meta, "snod_f0", "Positive snow-depth change since hour 0 (in)", "Total positive snow-depth change (in) & MSLP (mb)")


def plot_snod24(f, meta):
    return _snod_change(f, meta, "snod_m24", "24-hr positive snow-depth change (in)", "24-hr positive snow-depth change (in) & MSLP (mb)")


def plot_rh700_300(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    rh = np.mean([pick(f, "r700"), pick(f, "r500"), pick(f, "r300")], axis=0)
    levels = np.arange(0, 101, 10)
    cf = ax.contourf(lon, lat, rh, levels=levels, cmap="BrBG", transform=PC, zorder=2)
    if "gh500" in f:
        contour_labeled(ax, lon, lat, smooth(f["gh500"] / 10), z500_levels(), "black", 0.8)
    add_basemap(ax)
    colorbar(fig, cf, "700–300 mb mean relative humidity (%)", ticks=levels)
    title(fig, ax, meta, "700–300 mb mean relative humidity (%) & 500 mb height (dam)")
    return fig


# ---------------------------------------------------- upper dynamics --------

def plot_z500_mslp(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    z = smooth(pick(f, "gh500") / 10)
    levels = np.arange(492, 600, 3)
    cf = ax.contourf(lon, lat, z, levels=levels, cmap="turbo", extend="both", transform=PC, zorder=2)
    ax.contour(lon, lat, z, levels=z500_levels(), colors="black", linewidths=0.7, transform=PC, zorder=4)
    mslp_contours(ax, f, color="white", lw=1.0)
    add_basemap(ax)
    colorbar(fig, cf, "500 mb height (dam)", ticks=levels[::4])
    title(fig, ax, meta, "500 mb height (dam) & MSLP (mb, white)")
    return fig


def _vort_level(f, meta, lev, zlevels):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    u, v = pick(f, f"u{lev}"), pick(f, f"v{lev}")
    vort = smooth(rel_vort(u, v, lon, lat), 1.0) * 1e5
    levels = np.arange(4, 44, 2)
    cf = ax.contourf(lon, lat, vort, levels=levels, cmap=VORT_CMAP, extend="max", transform=PC, zorder=2)
    contour_labeled(ax, lon, lat, smooth(pick(f, f"gh{lev}") / 10), zlevels, "black", 0.9)
    barbs(ax, lon, lat, u * 1.944, v * 1.944, color="#333")
    add_basemap(ax)
    colorbar(fig, cf, f"{lev} mb relative vorticity (10⁻⁵ s⁻¹)", ticks=levels[::2])
    title(fig, ax, meta, f"{lev} mb height (dam), relative vorticity & wind (kt)")
    return fig


def plot_z700_vort(f, meta):
    return _vort_level(f, meta, 700, np.arange(270, 330, 3))


def plot_z850_vort(f, meta):
    return _vort_level(f, meta, 850, np.arange(100, 180, 3))


def plot_z850_wind(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    u, v = pick(f, "u850") * 1.944, pick(f, "v850") * 1.944
    spd = np.hypot(u, v)
    bounds = [20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120]
    cmap = plt.get_cmap("plasma_r", len(bounds) - 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    cf = mesh(ax, lon, lat, spd, 20, cmap=cmap, norm=norm, transform=PC, zorder=2)
    contour_labeled(ax, lon, lat, smooth(pick(f, "gh850") / 10), np.arange(100, 180, 3), "black", 0.9)
    barbs(ax, lon, lat, u, v, color="#333")
    add_basemap(ax)
    colorbar(fig, cf, "850 mb wind speed (kt)", ticks=bounds)
    title(fig, ax, meta, "850 mb height (dam) & wind (kt)")
    return fig


def plot_wind250(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    u, v = pick(f, "u250") * 1.944, pick(f, "v250") * 1.944
    spd = np.hypot(u, v)
    bounds = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 160, 180, 200]
    cmap = plt.get_cmap("viridis", len(bounds) - 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    cf = mesh(ax, lon, lat, spd, 50, cmap=cmap, norm=norm, transform=PC, zorder=2)
    contour_labeled(ax, lon, lat, smooth(pick(f, "gh250") / 10), np.arange(960, 1140, 12), "black", 0.9)
    barbs(ax, lon, lat, u, v, color="#333")
    add_basemap(ax)
    colorbar(fig, cf, "250 mb wind speed (kt)", ticks=bounds)
    title(fig, ax, meta, "250 mb wind (kt) & height (dam)")
    return fig


def plot_pv2(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    p = pick(f, "pres_pv") / 100
    levels = np.arange(100, 725, 25)
    cf = ax.contourf(lon, lat, p, levels=levels, cmap="nipy_spectral_r", extend="both", transform=PC, zorder=2)
    if "u_pv" in f:
        barbs(ax, lon, lat, pick(f, "u_pv") * 1.944, pick(f, "v_pv") * 1.944, color="#111")
    add_basemap(ax)
    colorbar(fig, cf, "Pressure on the 2 PVU surface (mb)", ticks=levels[::4])
    title(fig, ax, meta, "Dynamic tropopause: 2 PVU pressure (mb) & wind (kt)")
    return fig


def plot_sim_ir(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    tb = pick(f, "sbt124") - 273.15
    # classic IR enhancement: greys for warm scenes, colours for cold cloud tops
    colors = ["#ffffff", "#ff00ff", "#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff8000", "#ff0000", "#600000"]
    cold = mcolors.LinearSegmentedColormap.from_list("ir_cold", colors[::-1])
    grey = plt.get_cmap("gray_r")
    warm = ax.contourf(lon, lat, tb, levels=np.arange(-30, 41, 2), cmap=grey, extend="max", transform=PC, zorder=2)
    cf = ax.contourf(lon, lat, np.ma.masked_greater(tb, -30), levels=np.arange(-90, -29, 3), cmap=cold, transform=PC, zorder=3)
    mslp_contours(ax, f, color="#ffd400", lw=0.6, labels=False)
    add_basemap(ax)
    cb = colorbar(fig, cf, "Simulated IR brightness temperature (°C); greys −30 to +40", ticks=np.arange(-90, -29, 12))
    title(fig, ax, meta, "Simulated IR satellite (10.7 µm) & MSLP (mb)")
    return fig


# ---------------------------------------------------- thermodynamics --------

def plot_t700_wind(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    t = pick(f, "t700") - 273.15
    levels = np.arange(-40, 26, 2)
    cf = ax.contourf(lon, lat, t, levels=levels, cmap="RdYlBu_r", extend="both", transform=PC, zorder=2)
    ax.contour(lon, lat, t, levels=[0], colors="k", linewidths=1.2, linestyles="dashed", transform=PC, zorder=4)
    mslp_contours(ax, f, lw=0.8)
    barbs(ax, lon, lat, pick(f, "u700") * 1.944, pick(f, "v700") * 1.944, color="#222")
    add_basemap(ax)
    colorbar(fig, cf, "700 mb temperature (°C)", ticks=levels[::3])
    title(fig, ax, meta, "700 mb temperature (°C), wind (kt) & MSLP (mb)")
    return fig


# ---------------------------------------------------- diagnostics -----------

def _fgen(f, meta, lev, zlevels):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    t, u, v = pick(f, f"t{lev}"), pick(f, f"u{lev}"), pick(f, f"v{lev}")
    adv = smooth(temp_advection(t, u, v, lon, lat), 1.2)
    fg = smooth(frontogenesis(t, u, v, lon, lat), 1.5)
    levels = np.arange(-3, 3.25, 0.25)
    cf = ax.contourf(lon, lat, adv, levels=levels, cmap="RdBu_r", extend="both", transform=PC, zorder=2)
    ax.contour(lon, lat, fg, levels=[1, 2, 4, 8, 16], colors="#7a0177", linewidths=[0.7, 0.9, 1.1, 1.3, 1.5], transform=PC, zorder=5)
    contour_labeled(ax, lon, lat, smooth(pick(f, f"gh{lev}") / 10), zlevels, "black", 0.7)
    add_basemap(ax)
    colorbar(fig, cf, f"{lev} mb temperature advection (°C/hr); purple = frontogenesis 1,2,4,8,16 K/100km/3hr", ticks=levels[::4])
    title(fig, ax, meta, f"{lev} mb temperature advection & Petterssen frontogenesis")
    return fig


def plot_fgen700(f, meta):
    return _fgen(f, meta, 700, np.arange(270, 330, 3))


def plot_fgen850(f, meta):
    return _fgen(f, meta, 850, np.arange(100, 180, 3))


def plot_okubo850(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    u, v = pick(f, "u850"), pick(f, "v850")
    ow, axis, strain = okubo_weiss(smooth(u, 1.0), smooth(v, 1.0), lon, lat)
    ow = smooth(ow, 1.0) * 1e9
    levels = np.arange(-20, 0.1, 2)
    cf = ax.contourf(lon, lat, np.ma.masked_greater(ow, -0.5), levels=levels, cmap="YlOrRd_r", extend="min", transform=PC, zorder=2)
    # dilatation axes: short segments where strain is meaningful
    every = max(1, len(lon) // 34)
    LON, LAT = np.meshgrid(lon, lat)
    sub = (slice(None, None, every), slice(None, None, every))
    st = strain[sub]; ang = axis[sub]
    L = 0.9 * (lon[every] - lon[0]) if len(lon) > every else 0.5
    keep = st > np.nanpercentile(st, 60)
    for x0, y0, a, k in zip(LON[sub].ravel(), LAT[sub].ravel(), ang.ravel(), keep.ravel()):
        if k:
            ax.plot([x0 - L * np.cos(a) / 2, x0 + L * np.cos(a) / 2], [y0 - L * np.sin(a) / 2, y0 + L * np.sin(a) / 2],
                    color="#1848a8", lw=0.9, transform=PC, zorder=6)
    contour_labeled(ax, lon, lat, smooth(pick(f, "gh850") / 10), np.arange(100, 180, 3), "black", 0.7)
    add_basemap(ax)
    colorbar(fig, cf, "Okubo-Weiss (10⁻⁹ s⁻²; negative = rotation-dominated) · blue segments = dilatation axes", ticks=levels[::2])
    title(fig, ax, meta, "850 mb Okubo-Weiss parameter & dilatation axes")
    return fig


# ---------------------------------------------------- tropical --------------

def plot_shear(f, meta):
    """850–200 mb bulk shear: the classic deep-layer shear a tropical cyclone feels."""
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    du = (pick(f, "u200") - pick(f, "u850")) * 1.944
    dv = (pick(f, "v200") - pick(f, "v850")) * 1.944
    mag = smooth(np.hypot(du, dv), 1.0)
    bounds = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80]
    colors = ["#e8f6e8", "#a8dba8", "#59b559", "#f7e530", "#f5a623", "#f05a28", "#d0021b", "#9b0c3d", "#5e0a5e"]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    cf = mesh(ax, lon, lat, mag, 5, cmap=cmap, norm=norm, transform=PC, zorder=2)
    every = max(1, len(lon) // 26)
    ax.quiver(lon[::every], lat[::every], du[::every, ::every], dv[::every, ::every], transform=PC, zorder=7,
              scale=900, width=0.0016, color="#222", pivot="middle")
    if "gh500" in f:
        contour_labeled(ax, lon, lat, smooth(f["gh500"] / 10), z500_levels(), "black", 0.7)
    add_basemap(ax)
    colorbar(fig, cf, "850–200 mb shear (kt) — under 20 kt favours tropical development", ticks=bounds)
    title(fig, ax, meta, "850–200 mb wind shear (kt, arrows show shear vector) & 500 mb height (dam)")
    return fig


def plot_steering(f, meta):
    """Pressure-weighted 850–300 mb mean wind, a proxy for tropical cyclone steering."""
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    w = {850: 0.4, 500: 0.35, 300: 0.25}
    u = sum(pick(f, f"u{p}") * k for p, k in w.items()) * 1.944
    v = sum(pick(f, f"v{p}") * k for p, k in w.items()) * 1.944
    spd = smooth(np.hypot(u, v), 1.0)
    bounds = [5, 10, 15, 20, 25, 30, 40, 50, 60]
    cmap = plt.get_cmap("YlGnBu", len(bounds) - 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    cf = mesh(ax, lon, lat, spd, 5, cmap=cmap, norm=norm, transform=PC, zorder=2)
    ax.streamplot(lon, lat, u, v, density=1.6, color="#222", linewidth=0.6, arrowsize=0.7, transform=PC, zorder=6)
    mslp_contours(ax, f, color="#8b0000", lw=0.7, labels=False)
    add_basemap(ax)
    colorbar(fig, cf, "850–300 mb layer-mean wind (kt)", ticks=bounds)
    title(fig, ax, meta, "850–300 mb steering flow (kt, streamlines) & MSLP (mb, dark red)")
    return fig


def plot_div200(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    u, v = pick(f, "u200"), pick(f, "v200")
    dx, dy = grid_spacing(lon, lat)
    div = smooth(ddx(u, dx) + ddy(v, dy), 1.5) * 1e5
    levels = np.arange(2, 14.1, 1)
    cf = ax.contourf(lon, lat, div, levels=levels, cmap="Purples", extend="max", transform=PC, zorder=2)
    ax.contour(lon, lat, div, levels=-levels[::-1], colors="#b35806", linewidths=0.5, linestyles="dashed", transform=PC, zorder=3)
    contour_labeled(ax, lon, lat, smooth(pick(f, "gh200") / 10), np.arange(1140, 1290, 6), "black", 0.6)
    barbs(ax, lon, lat, u * 1.944, v * 1.944, color="#333")
    add_basemap(ax)
    colorbar(fig, cf, "200 mb divergence (10⁻⁵ s⁻¹) — shaded positive; dashed brown = convergence", ticks=levels[::2])
    title(fig, ax, meta, "200 mb divergence, wind (kt) & height (dam)")
    return fig


def plot_rh700(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    rh = pick(f, "r700")
    levels = np.arange(0, 101, 10)
    cf = ax.contourf(lon, lat, rh, levels=levels, cmap="BrBG", transform=PC, zorder=2)
    if "gh700" in f:
        contour_labeled(ax, lon, lat, smooth(f["gh700"] / 10), np.arange(270, 330, 3), "black", 0.6)
    barbs(ax, lon, lat, pick(f, "u700") * 1.944, pick(f, "v700") * 1.944, color="#333")
    add_basemap(ax)
    colorbar(fig, cf, "700 mb relative humidity (%) — browns = dry air intrusion", ticks=levels)
    title(fig, ax, meta, "700 mb relative humidity (%), wind (kt) & height (dam)")
    return fig


def plot_sst(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    t = pick(f, "t_sfc", "skt", "t") - 273.15
    land = pick(f, "lsm", "land")
    sst = np.ma.masked_where(land > 0.5, t)
    levels = np.arange(18, 33.1, 0.5)
    cf = ax.contourf(lon, lat, sst, levels=levels, cmap="turbo", extend="both", transform=PC, zorder=2)
    ax.contour(lon, lat, sst, levels=[26.5], colors="white", linewidths=1.4, transform=PC, zorder=4)
    ax.contour(lon, lat, sst, levels=[28, 30], colors="black", linewidths=0.6, transform=PC, zorder=4)
    land50 = cfeature.LAND.with_scale("50m")
    try:
        next(iter(land50.geometries()))            # force the (cached) download now, not at save time
        ax.add_feature(land50, facecolor="#d9d9d9", zorder=3)
    except Exception:  # noqa: BLE001
        pass
    mslp_contours(ax, f, color="#333", lw=0.6, labels=False)
    add_basemap(ax)
    colorbar(fig, cf, "Sea surface temperature (°C) — white line 26.5 °C, black 28 & 30 °C", ticks=levels[::4])
    title(fig, ax, meta, "Sea surface temperature (°C) & MSLP (mb)")
    return fig


def plot_vort_layer(f, meta):
    """850–500 mb layer-mean relative vorticity: tracks mid-level spins before a surface low forms."""
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    vort = np.mean([rel_vort(pick(f, f"u{p}"), pick(f, f"v{p}"), lon, lat) for p in (850, 700, 500)], axis=0)
    vort = smooth(vort, 1.2) * 1e5
    levels = np.arange(2, 30.1, 2)
    cf = ax.contourf(lon, lat, vort, levels=levels, cmap=VORT_CMAP, extend="max", transform=PC, zorder=2)
    barbs(ax, lon, lat, pick(f, "u700") * 1.944, pick(f, "v700") * 1.944, color="#333")
    mslp_contours(ax, f, lw=0.7)
    add_basemap(ax)
    colorbar(fig, cf, "850–500 mb mean relative vorticity (10⁻⁵ s⁻¹)", ticks=levels[::2])
    title(fig, ax, meta, "850–500 mb layer-mean vorticity, 700 mb wind (kt) & MSLP (mb)")
    return fig


# ---------------------------------------------------- mesoscale extras ------

def plot_precip6(f, meta):
    """6-hr precipitation on its own (models without MSLP, e.g. the National Blend)."""
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    if "tp_6" in f:
        cmap, norm, bounds = _precip_cmap()
        cf = mesh(ax, lon, lat, f["tp_6"] / 25.4, 0.01, cmap=cmap, norm=norm, transform=PC, zorder=2)
        colorbar(fig, cf, "6-hr precipitation (in)", ticks=bounds)
    else:
        ax.text(0.5, 0.5, "Not available at this hour", transform=ax.transAxes, ha="center", fontsize=11, color="#666", zorder=9)
    mslp_contours(ax, f, lw=0.7)
    add_basemap(ax)
    title(fig, ax, meta, "6-hr precipitation (in)")
    return fig


def plot_gust(f, meta):
    fig, ax = new_map(meta)
    lon, lat = f.lon, f.lat
    g = pick(f, "gust") * 1.944
    bounds = [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100]
    colors = ["#cfe8ff", "#9dcbff", "#5aa7f5", "#2e7ad8", "#2bb673", "#7ed321", "#f8e71c", "#f5a623", "#f05a28", "#d0021b", "#9b0c3d"]
    cmap = mcolors.ListedColormap(colors); norm = mcolors.BoundaryNorm(bounds, cmap.N)
    cf = mesh(ax, lon, lat, g, 15, cmap=cmap, norm=norm, transform=PC, zorder=2)
    if "u10" in f:
        barbs(ax, lon, lat, pick(f, "u10") * 1.944, pick(f, "v10") * 1.944, color="#333")
    mslp_contours(ax, f, lw=0.7)
    fg = dict(f); fg["gkt"] = g
    ff = type("F", (dict,), {})(fg); ff.lon, ff.lat = lon, lat
    city_values(ax, ff, "gkt", meta, fmt="{:.0f} kt")
    add_basemap(ax)
    colorbar(fig, cf, "10 m wind gust (kt)", ticks=bounds)
    title(fig, ax, meta, "10 m wind gust (kt)")
    return fig
