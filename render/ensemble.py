"""
Ensemble products. Each function takes (stack, meta) where `stack` is a dict of
field -> 3-D array (member, lat, lon) with .lon/.lat and .members attributes,
and returns a matplotlib Figure. Built on the helpers in plots.py.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter

from config import MODEL
from plots import PC, add_basemap, colorbar, contour_labeled, mesh, new_map, title

MEMBER_COLORS = plt.get_cmap("tab20")


class Stack(dict):
    lon: np.ndarray
    lat: np.ndarray
    members: list


def _s(a, s=1.0):
    return gaussian_filter(a, s)


def subtitle(meta, left):
    return f"{left}  ·  {len(meta['members'])} members"


# ------------------------------------------------------------ mean/spread ---

PROB_BOUNDS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PROB_CMAP = mcolors.ListedColormap(["#e8f1fb", "#c9dff5", "#a2c8ec", "#6faee0", "#3f8fd2", "#2c6fb5",
                                    "#3ec24d", "#f7e530", "#f5a623", "#e0341b"])


def _mean_spread(stack, meta, key, scale, unit, mean_levels, spread_bounds, label, title_txt, cmap="YlOrRd", contour_fmt="%d"):
    fig, ax = new_map(meta)
    lon, lat = stack.lon, stack.lat
    data = stack[key] * scale
    mean, spread = _s(np.nanmean(data, axis=0)), _s(np.nanstd(data, axis=0))
    cm = plt.get_cmap(cmap, len(spread_bounds) - 1)
    norm = mcolors.BoundaryNorm(spread_bounds, cm.N)
    cf = mesh(ax, lon, lat, spread, spread_bounds[0], cmap=cm, norm=norm, transform=PC, zorder=2)
    contour_labeled(ax, lon, lat, mean, mean_levels, "black", 1.1, fmt=contour_fmt)
    add_basemap(ax)
    colorbar(fig, cf, f"Ensemble spread — standard deviation ({unit})", ticks=spread_bounds)
    title(fig, ax, meta, subtitle(meta, title_txt))
    return fig


def ens_mslp(stack, meta):
    return _mean_spread(stack, meta, "prmsl", 0.01, "mb", np.arange(940, 1060, 4), [1, 2, 3, 4, 6, 8, 10, 12, 16, 20],
                        "MSLP", "MSLP ensemble mean (mb, contours) & spread")


def ens_z500(stack, meta):
    return _mean_spread(stack, meta, "gh500", 0.1, "dam", np.arange(480, 620, 6), [1, 2, 3, 4, 6, 8, 10, 12, 16, 20],
                        "500 mb", "500 mb height ensemble mean (dam) & spread", cmap="PuRd")


def ens_t850(stack, meta):
    fig, ax = new_map(meta)
    lon, lat = stack.lon, stack.lat
    data = stack["t850"] - 273.15
    mean, spread = _s(np.nanmean(data, axis=0)), _s(np.nanstd(data, axis=0))
    levels = np.arange(-30, 36, 2)
    cf = ax.contourf(lon, lat, mean, levels=levels, cmap="RdYlBu_r", extend="both", transform=PC, zorder=2)
    ax.contour(lon, lat, mean, levels=[0], colors="k", linewidths=1.2, linestyles="dashed", transform=PC, zorder=4)
    ax.contour(lon, lat, spread, levels=[2, 3, 4, 6], colors="#333", linewidths=[0.6, 0.8, 1.0, 1.2], transform=PC, zorder=5)
    add_basemap(ax)
    colorbar(fig, cf, "850 mb temperature ensemble mean (°C); grey contours = spread 2, 3, 4, 6 °C", ticks=levels[::3])
    title(fig, ax, meta, subtitle(meta, "850 mb temperature ensemble mean & spread"))
    return fig


def ens_t2m(stack, meta):
    fig, ax = new_map(meta)
    lon, lat = stack.lon, stack.lat
    data = (stack["t2m"] - 273.15) * 9 / 5 + 32
    mean, spread = _s(np.nanmean(data, axis=0)), _s(np.nanstd(data, axis=0))
    levels = np.arange(-30, 121, 5)
    cf = ax.contourf(lon, lat, mean, levels=levels, cmap="turbo", extend="both", transform=PC, zorder=2)
    ax.contour(lon, lat, spread, levels=[4, 6, 8, 12], colors="#333", linewidths=[0.6, 0.8, 1.0, 1.2], transform=PC, zorder=5)
    add_basemap(ax)
    colorbar(fig, cf, "2 m temperature ensemble mean (°F); grey contours = spread 4, 6, 8, 12 °F", ticks=levels[::2])
    title(fig, ax, meta, subtitle(meta, "2 m temperature ensemble mean & spread"))
    return fig


def ens_precip6(stack, meta):
    from plots import _precip_cmap, mslp_contours
    fig, ax = new_map(meta)
    lon, lat = stack.lon, stack.lat
    if "tp_6" in stack:
        mean = np.nanmean(stack["tp_6"], axis=0) / 25.4
        cmap, norm, bounds = _precip_cmap()
        cf = mesh(ax, lon, lat, mean, 0.01, cmap=cmap, norm=norm, transform=PC, zorder=2)
        colorbar(fig, cf, "6-hr precipitation, ensemble mean (in)", ticks=bounds)
    else:
        ax.text(0.5, 0.5, "No accumulated precipitation at hour 000", transform=ax.transAxes, ha="center", fontsize=11, color="#666", zorder=9)
    f = {"prmsl": np.nanmean(stack["prmsl"], axis=0)}
    f_obj = type("F", (dict,), {})(f); f_obj.lon, f_obj.lat = lon, lat
    mslp_contours(ax, f_obj, lw=0.8)
    add_basemap(ax)
    title(fig, ax, meta, subtitle(meta, "6-hr precipitation ensemble mean (in) & mean MSLP (mb)"))
    return fig


def ens_lows(stack, meta):
    """Every member's surface low centres, labelled with the member number and
    coloured by central pressure. Where does each member put the storm?"""
    fig, ax = new_map(meta)
    lon, lat = stack.lon, stack.lat
    LON, LAT = np.meshgrid(lon, lat)
    mean = _s(np.nanmean(stack["prmsl"], axis=0) / 100)
    ax.contour(lon, lat, mean, levels=np.arange(940, 1060, 4), colors="#555", linewidths=0.8, transform=PC, zorder=3)
    bounds = [940, 960, 970, 980, 990, 996, 1000, 1004, 1008, 1012]
    cmap = mcolors.ListedColormap(["#5e0a5e", "#9b0c3d", "#d0021b", "#f05a28", "#f5a623", "#7ed321", "#1f8f3a", "#2b8cbe", "#7fb3d5"])
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    size = max(9, len(lon) // 40)
    for i, m in enumerate(stack.members):
        p = _s(stack["prmsl"][i] / 100, 1.0)
        mn = minimum_filter(p, size)
        ys, xs = np.where((p == mn) & (p < 1012))
        label = m.replace("p", "").replace("c", "")          # c00 -> 00, p07 -> 07
        for y, x in zip(ys, xs):
            if 2 < y < len(lat) - 3 and 2 < x < len(lon) - 3:
                ax.text(LON[y, x], LAT[y, x], label, fontsize=8.5, fontweight="bold", ha="center", va="center",
                        color=cmap(norm(p[y, x])), transform=PC, zorder=7,
                        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    add_basemap(ax)
    colorbar(fig, sm, "Low centre pressure (mb) · number = member, 00 = control", ticks=bounds)
    title(fig, ax, meta, subtitle(meta, "Member surface low centres & ensemble-mean MSLP (mb)"))
    return fig


# --------------------------------------------------------------- spaghetti ---

def _spaghetti(stack, meta, key, scale, levels, title_txt, unit):
    fig, ax = new_map(meta)
    lon, lat = stack.lon, stack.lat
    styles = ["solid", "dashed"]
    for i, m in enumerate(stack.members):
        for lev, ls in zip(levels, styles):
            ax.contour(lon, lat, _s(stack[key][i] * scale, 1.0), levels=[lev], colors=[MEMBER_COLORS(i % 20)],
                       linewidths=0.8, linestyles=ls, transform=PC, zorder=4)
    mean = _s(np.nanmean(stack[key], axis=0) * scale)
    for lev, ls in zip(levels, styles):
        cs = ax.contour(lon, lat, mean, levels=[lev], colors="black", linewidths=2.4, linestyles=ls, transform=PC, zorder=6)
        ax.clabel(cs, fmt="%d", fontsize=8)
    add_basemap(ax)
    fig.text(0.5, 0.045, f"Thin lines: each member's {levels[0]} (solid) and {levels[1]} (dashed) {unit} contour. Black: ensemble mean.",
             ha="center", fontsize=9, color="#444")
    title(fig, ax, meta, subtitle(meta, title_txt))
    return fig


def spag_z500(stack, meta):
    return _spaghetti(stack, meta, "gh500", 0.1, [564, 582], "500 mb height spaghetti", "dam")


def spag_mslp(stack, meta):
    return _spaghetti(stack, meta, "prmsl", 0.01, [1000, 1012], "MSLP spaghetti", "mb")


# ------------------------------------------------------------- probability ---

def _prob(stack, meta, mask_stack, title_txt, extra=None):
    fig, ax = new_map(meta)
    lon, lat = stack.lon, stack.lat
    prob = 100.0 * np.nanmean(mask_stack.astype(float), axis=0)
    norm = mcolors.BoundaryNorm(PROB_BOUNDS, PROB_CMAP.N)
    cf = mesh(ax, lon, lat, _s(prob, 0.7), PROB_BOUNDS[0], cmap=PROB_CMAP, norm=norm, transform=PC, zorder=2)
    if extra:
        extra(ax)
    mean = _s(np.nanmean(stack["prmsl"], axis=0) / 100)
    ax.contour(lon, lat, mean, levels=np.arange(940, 1060, 4), colors="#555", linewidths=0.6, transform=PC, zorder=4)
    add_basemap(ax)
    colorbar(fig, cf, "Probability (% of members)", ticks=PROB_BOUNDS)
    title(fig, ax, meta, subtitle(meta, title_txt + " & mean MSLP (mb)"))
    return fig


def _wind_kt(stack):
    return np.hypot(stack["u10"], stack["v10"]) * 1.944


def prob_wind34(stack, meta):
    return _prob(stack, meta, _wind_kt(stack) >= 34, "Probability of 10 m wind ≥ 34 kt (tropical storm force)")


def prob_wind50(stack, meta):
    return _prob(stack, meta, _wind_kt(stack) >= 50, "Probability of 10 m wind ≥ 50 kt")


def prob_wind64(stack, meta):
    return _prob(stack, meta, _wind_kt(stack) >= 64, "Probability of 10 m wind ≥ 64 kt (hurricane force)")


def _precip_prob(stack, meta, inches):
    if "tp_6" not in stack:
        fig, ax = new_map(meta)
        ax.text(0.5, 0.5, "No accumulated precipitation at hour 000", transform=ax.transAxes, ha="center", fontsize=11, color="#666", zorder=9)
        add_basemap(ax); title(fig, ax, meta, subtitle(meta, f"Probability of 6-hr precip ≥ {inches} in"))
        return fig
    return _prob(stack, meta, stack["tp_6"] / 25.4 >= inches, f"Probability of 6-hr precipitation ≥ {inches} in")


def prob_precip05(stack, meta):
    return _precip_prob(stack, meta, 0.5)


def prob_precip1(stack, meta):
    return _precip_prob(stack, meta, 1.0)


def prob_mslp1000(stack, meta):
    return _prob(stack, meta, stack["prmsl"] / 100 <= 1000, "Probability of MSLP ≤ 1000 mb")


def prob_t850frz(stack, meta):
    return _prob(stack, meta, stack["t850"] <= 273.15, "Probability of 850 mb temperature ≤ 0 °C")
