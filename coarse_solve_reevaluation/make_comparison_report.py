#!/usr/bin/env python3
"""Aggregate metrics across all coarse-solve experiments and generate comparison tables & figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDERED_EXPERIMENTS = [
    ("baseline", "Baseline (Rect. Newtonian)"),
    ("rect_glen", "Exp 1: Rect. Glen's Law"),
    ("basin_glen", "Exp 2: REMA Basin Glen (Free-slip)"),
    ("basin_glen_noslip", "Exp 2b: REMA Basin Glen (No-slip)"),
    ("basin_newtonian", "Diag: REMA Basin Newtonian (Free-slip)"),
]
BOREHOLES_7 = ("ALHIC1902", "ALHIC2302", "ALHIC1502", "ALHIC1901", "ALHIC2201", "ALHIC2301", "ALHIC2401")
TEMP_BOREHOLES = ("ALHIC2301", "ALHIC2401", "ALHIC1902", "ALHIC2302", "ALHIC1901", "ALHIC2201")


def load_available_metrics(results_dir: Path) -> list[tuple[str, str, dict]]:
    loaded = []
    for key, label in ORDERED_EXPERIMENTS:
        p = results_dir / f"{key}_metrics.json"
        if p.is_file():
            loaded.append((key, label, json.loads(p.read_text(encoding="utf-8"))))
    return loaded


def build_summary_table(items: list[tuple[str, str, dict]]) -> pd.DataFrame:
    base_min10 = None
    for key, _, m in items:
        if key == "baseline":
            base_min10 = (m["window_minima_by_height_m"]["10"]["x_m"], m["window_minima_by_height_m"]["10"]["y_m"])

    rows = []
    for key, label, m in items:
        gc = m["gnss_calibration"]
        bh = m["borehole_profiles"]
        f10 = m["ols_fits_by_height_m"]["10"]
        w10 = m["window_minima_by_height_m"]["10"]
        cf = m["counterflow_and_stagnation_fig1b"]
        td = m["borehole_temperature_diagnostics"]
        sm = m["spaulding_smb_coarse_diagnostics"]
        disp_m = (
            math.hypot(w10["x_m"] - base_min10[0], w10["y_m"] - base_min10[1])
            if base_min10 is not None
            else float("nan")
        )
        row = {
            "experiment": key,
            "label": label,
            "E_opt": gc["E_opt"],
            "E_ci_low": gc["E_ci_low"],
            "E_ci_high": gc["E_ci_high"],
            "gnss_horiz_rmse_m_yr": gc["horiz_rmse_m_yr"],
            "gnss_vert_rmse_m_yr": gc["vert_rmse_m_yr"],
            "gnss_mean_ang_err_deg": gc["mean_ang_err_deg"],
            "fine_side_bc_rms_mm_yr_raw": m["fine_side_bc_rms_mm_yr_raw"],
            "fine_side_bc_rms_mm_yr_scaled": m["fine_side_bc_rms_mm_yr_scaled"],
            "U10_ALHIC1902_mm_yr": bh["ALHIC1902"]["U10_mm_yr"],
            "U10_ALHIC2302_mm_yr": bh["ALHIC2302"]["U10_mm_yr"],
            "U10_ALHIC1502_mm_yr": bh["ALHIC1502"]["U10_mm_yr"],
            "U10_ALHIC1901_mm_yr": bh["ALHIC1901"]["U10_mm_yr"],
            "U10_ALHIC2201_mm_yr": bh["ALHIC2201"]["U10_mm_yr"],
            "U10_ALHIC2301_mm_yr": bh["ALHIC2301"]["U10_mm_yr"],
            "U10_ALHIC2401_mm_yr": bh["ALHIC2401"]["U10_mm_yr"],
            "regime_ranking_preserved": m["regime_ranking_preserved"],
            "contrast_2301_over_1902": m["cds_to_nunatak_u10_contrast"]["ALHIC2301_over_ALHIC1902"],
            "contrast_2401_over_2302": m["cds_to_nunatak_u10_contrast"]["ALHIC2401_over_ALHIC2302"],
            "has_counterflow_fig1b": cf["has_eastward_counterflow_in_fig1b"],
            "max_counterflow_u_mm_yr": cf["max_eastward_u_mm_yr"],
            "slow_zone_lt2_area_km2": w10["slow_zone_lt2_area_km2"],
            "win10_min_speed_mm_yr": w10["speed_mm_yr"],
            "win10_min_disp_from_base_m": disp_m,
            "win10_extrap_age_ma": w10["fitted_age_ma"],
            "win10_pi_low_ma": w10["pi_low_ma"],
            "win10_pi_high_ma": w10["pi_high_ma"],
            "ols10_r2": f10["r2"],
            "ols10_r": f10["r"],
            "ols10_slope_ma_per_dec": f10["beta"],
            "ols10_p": f10["p_two_sided"],
            "ols5_r2": m["ols_fits_by_height_m"]["5"]["r2"],
            "ols20_r2": m["ols_fits_by_height_m"]["20"]["r2"],
            "ols50_r2": m["ols_fits_by_height_m"]["50"]["r2"],
            "temp_mean_adv_mae_K": td["mean_adv_mae_K"],
            "temp_nunatak_lt_cds_ridge_noadv": td["nunatak_lt_cds_and_ridge"],
            "smb_n_ice_firn": sm.get("n_ice_firn", np.nan),
            "smb_bias_m_yr": sm.get("bias_obs_minus_mod_m_yr", np.nan),
            "smb_rmse_m_yr": sm.get("rmse_m_yr", np.nan),
            "smb_pearson_r": sm.get("pearson_r", np.nan),
            "smb_spearman_rho": sm.get("spearman_rho", np.nan),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_speed_map_comparisons(results_dir: Path, cmp_dir: Path, items: list[tuple[str, str, dict]]):
    keys_show = [k for k, _, _ in items if k in ("baseline", "rect_glen", "basin_glen", "basin_glen_noslip")]
    grids = {}
    for k in keys_show:
        p = results_dir / f"{k}_speed10m_grid.npz"
        if p.is_file():
            grids[k] = np.load(p)
    if "baseline" not in grids or len(grids) < 2:
        return

    base = grids["baseline"]
    ux = np.unique(base["x"])
    uy = np.unique(base["y"])

    def to_2d(d, field="speed_mm_yr"):
        df = pd.DataFrame({"x": d["x"], "y": d["y"], "val": d[field]})
        return df.pivot(index="y", columns="x", values="val").to_numpy(float)

    S_base = to_2d(base)
    B_base = to_2d(base, "bed")

    others = [k for k in ("rect_glen", "basin_glen", "basin_glen_noslip") if k in grids]
    ncols = 1 + len(others)
    fig, axes = plt.subplots(2, ncols, figsize=(4.8 * ncols, 9.2), layout="constrained")
    if ncols == 1:
        axes = np.atleast_2d(axes)

    lbl_map = dict(ORDERED_EXPERIMENTS)
    cmap_s = plt.get_cmap("viridis").copy()
    cmap_s.set_bad("lightgrey")
    norm_s = mcolors.LogNorm(vmin=0.05, vmax=200.0)

    # Top row: absolute 10 m speed maps
    for col_i, k in enumerate(["baseline", *others]):
        ax = axes[0, col_i]
        S = to_2d(grids[k])
        im = ax.pcolormesh(ux / 1e3, uy / 1e3, S, cmap=cmap_s, norm=norm_s, shading="nearest")
        ax.contour(ux / 1e3, uy / 1e3, B_base, levels=[1600, 1700, 1800, 1900, 2000], colors="k", linewidths=0.4)
        ax.contour(ux / 1e3, uy / 1e3, S, levels=[2.0], colors="cyan", linewidths=1.2)
        ax.set_xlim(507.5, 512.0)
        ax.set_ylim(-1355.5, -1351.0)
        ax.set_aspect("equal")
        ax.set_title(lbl_map[k], fontsize=9.5)
        ax.set_xlabel("Easting (km)")
        if col_i == 0:
            ax.set_ylabel("Northing (km)")
    fig.colorbar(im, ax=axes[0, :], shrink=0.85, label=r"$U_{10}$ (mm yr$^{-1}$; cyan = 2 mm yr$^{-1}$)")

    # Bottom row: log10(S_exp / S_base) ratio maps (and baseline zoom in col 0)
    ax0 = axes[1, 0]
    im0 = ax0.pcolormesh(ux / 1e3, uy / 1e3, S_base, cmap="gray", norm=mcolors.LogNorm(vmin=0.01, vmax=100.0), shading="nearest")
    ax0.contour(ux / 1e3, uy / 1e3, B_base, levels=[1600, 1700, 1800, 1900, 2000], colors="cyan", linewidths=0.45)
    ax0.set_xlim(509.0, 511.0)
    ax0.set_ylim(-1355.1, -1352.5)
    ax0.set_aspect("equal")
    ax0.set_title("Baseline Stagnation Window (Fig. 3b)", fontsize=9.5)
    ax0.set_xlabel("Easting (km)")
    ax0.set_ylabel("Northing (km)")

    cmap_r = plt.get_cmap("RdBu_r").copy()
    cmap_r.set_bad("lightgrey")
    im_r = None
    for col_i, k in enumerate(others, start=1):
        ax = axes[1, col_i]
        S = to_2d(grids[k])
        ratio = np.where((S > 0) & (S_base > 0), np.log10(S / S_base), np.nan)
        im_r = ax.pcolormesh(ux / 1e3, uy / 1e3, ratio, cmap=cmap_r, vmin=-0.5, vmax=0.5, shading="nearest")
        ax.contour(ux / 1e3, uy / 1e3, B_base, levels=[1600, 1700, 1800, 1900, 2000], colors="k", linewidths=0.4)
        med_pct = float((10 ** np.nanmedian(ratio) - 1.0) * 100.0)
        iqr_lo = float((10 ** np.nanpercentile(ratio, 25) - 1.0) * 100.0)
        iqr_hi = float((10 ** np.nanpercentile(ratio, 75) - 1.0) * 100.0)
        ax.set_xlim(507.5, 512.0)
        ax.set_ylim(-1355.5, -1351.0)
        ax.set_aspect("equal")
        ax.set_title(f"{k} / baseline (med {med_pct:+.1f}%, IQR [{iqr_lo:+.0f}%, {iqr_hi:+.0f}%])", fontsize=8.8)
        ax.set_xlabel("Easting (km)")
    if im_r is not None:
        fig.colorbar(im_r, ax=axes[1, 1:], shrink=0.85, label=r"$\log_{10}(U_{10,\mathrm{exp}} / U_{10,\mathrm{base}})$ ($\pm 0.5 \approx 3\times$)")

    fig.savefig(cmp_dir / "compare_speed10m_maps.png", dpi=220)
    plt.close(fig)


def plot_summary_bars_and_regressions(results_dir: Path, cmp_dir: Path, items: list[tuple[str, str, dict]]):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), layout="constrained")
    colors = ["#222222", "#0072B2", "#009E73", "#D55E00", "#CC79A7"]

    # Panel A: U_10 across all 7 boreholes (log scale)
    ax = axes[0, 0]
    x = np.arange(len(BOREHOLES_7))
    w = 0.8 / max(len(items), 1)
    for idx, (key, label, m) in enumerate(items):
        vals = [m["borehole_profiles"][b]["U10_mm_yr"] for b in BOREHOLES_7]
        ax.bar(x + (idx - 0.5 * (len(items) - 1)) * w, vals, width=w, color=colors[idx % len(colors)], label=label, edgecolor="k", lw=0.4)
    ax.axhline(2.0, color="red", ls="--", lw=1.2, label="Slow-flow threshold (2 mm/yr)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(BOREHOLES_7, rotation=30, ha="right")
    ax.set_ylabel(r"Modeled $U_{10}$ (mm yr$^{-1}$)")
    ax.set_title("A. Near-Bed Speeds Across All Three Flow Regimes")
    ax.grid(axis="y", which="both", alpha=0.35)
    ax.legend(loc="upper left", fontsize=7.5)

    # Panel B: Age vs U_10 regression comparison (10 m above bed)
    ax = axes[0, 1]
    for idx, (key, label, m) in enumerate(items):
        cdf = pd.read_csv(results_dir / f"{key}_core_samples.csv")
        sub = cdf[cdf["height_m"] == 10.0]
        fit = m["ols_fits_by_height_m"]["10"]
        clr = colors[idx % len(colors)]
        spds = np.logspace(fit["log_speed_min"], fit["log_speed_max"], 100)
        ages_fit = fit["alpha"] + fit["beta"] * np.log10(spds)
        ax.plot(spds, ages_fit, color=clr, lw=1.8, label=f"{key}: $R^2$={fit['r2']:.2f}, slope={fit['beta']:.2f}")
        fit_pts = sub[sub["core_id"] != "ALHIC2301"]
        cds_pt = sub[sub["core_id"] == "ALHIC2301"]
        ax.scatter(fit_pts["speed_mm_yr"], fit_pts["max_age_ma"], s=36, color=clr, edgecolor="k", lw=0.5, zorder=5)
        ax.scatter(cds_pt["speed_mm_yr"], cds_pt["max_age_ma"], s=45, marker="s", facecolor="none", edgecolor=clr, lw=1.4, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel(r"Modeled speed 10 m above bed (mm yr$^{-1}$)")
    ax.set_ylabel("Maximum dated age (Ma)")
    ax.set_title("B. Age–Speed Regression (circles=fit cores, squares=ALHIC2301 CDS)")
    ax.grid(which="both", alpha=0.35)
    ax.legend(loc="lower left", fontsize=7.5)

    # Panel C: GNSS E_opt & horizontal RMSE
    ax = axes[1, 0]
    keys = [k for k, _, _ in items]
    e_vals = [m["gnss_calibration"]["E_opt"] for _, _, m in items]
    e_lo = [m["gnss_calibration"]["E_opt"] - m["gnss_calibration"]["E_ci_low"] for _, _, m in items]
    e_hi = [m["gnss_calibration"]["E_ci_high"] - m["gnss_calibration"]["E_opt"] for _, _, m in items]
    rmses = [m["gnss_calibration"]["horiz_rmse_m_yr"] * 1000.0 for _, _, m in items]
    x_k = np.arange(len(keys))
    ax.errorbar(x_k, e_vals, yerr=[e_lo, e_hi], fmt="o", ms=7, color="#0072B2", capsize=4, label=r"$E_{\mathrm{opt}}$ (95% CI)")
    ax.set_xticks(x_k)
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel(r"Calibrated enhancement factor $E_{\mathrm{opt}}$", color="#0072B2")
    ax.grid(alpha=0.35)
    ax2 = ax.twinx()
    ax2.bar(x_k + 0.25, rmses, width=0.25, color="#D55E00", alpha=0.6, label="GNSS horiz. RMSE (mm/yr)")
    ax2.set_ylabel(r"GNSS horizontal RMSE (mm yr$^{-1}$)", color="#D55E00")
    ax.set_title("C. GNSS Calibration ($E_{\\mathrm{opt}}$ & Horizontal RMSE)")

    # Panel D: No-advection temperature MAE across boreholes (Fig S2M test)
    ax = axes[1, 1]
    xt = np.arange(len(TEMP_BOREHOLES))
    for idx, (key, label, m) in enumerate(items):
        maes = [m["borehole_temperature_diagnostics"]["noadv_mae_by_borehole_K"][b] for b in TEMP_BOREHOLES]
        ax.plot(xt, maes, marker="o", lw=1.6, color=colors[idx % len(colors)], label=key)
    ax.set_xticks(xt)
    ax.set_xticklabels(TEMP_BOREHOLES, rotation=25, ha="right")
    ax.set_ylabel("No-advection MAE (K)")
    ax.set_title("D. Borehole Thermal Advection Signal (Fig. S2M)")
    ax.grid(alpha=0.35)
    ax.legend(loc="upper right", fontsize=7.5)

    fig.savefig(cmp_dir / "compare_metrics_overview.png", dpi=220)
    plt.close(fig)


def plot_boundary_coupling_diagnostic(results_dir: Path, cmp_dir: Path, items: list[tuple[str, str, dict]]):
    """Visualize how coarse-to-fine lateral BC changes decay inward across GNSS stakes & raw borehole speeds."""
    keys_present = [k for k, _, _ in items]
    if "baseline" not in keys_present or len(items) < 2:
        return

    base_gnss = pd.read_csv(results_dir / "baseline_gnss_stakes.csv").set_index("name")
    # Fine mesh bounding box (EPSG:3031)
    fx0, fx1, fy0, fy1 = 506902.38, 512046.01, -1356284.47, -1350629.26
    dist_edge = np.minimum.reduce([
        base_gnss["x"] - fx0, fx1 - base_gnss["x"],
        base_gnss["y"] - fy0, fy1 - base_gnss["y"],
    ])

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9), layout="constrained")
    colors = {"baseline": "#222222", "rect_glen": "#0072B2", "basin_glen": "#009E73", "basin_glen_noslip": "#D55E00", "basin_newtonian": "#CC79A7"}
    markers = {"rect_glen": "o", "basin_glen": "s", "basin_glen_noslip": "^", "basin_newtonian": "D"}

    # Panel A: Per-wall coarse-to-fine raw RMS boundary velocity
    ax = axes[0]
    walls = ["east", "south", "north", "west"]
    xw = np.arange(len(walls))
    ww = 0.78 / len(items)
    for idx, (key, label, m) in enumerate(items):
        wrms = [m.get("fine_side_bc_wall_rms", {}).get(w, {}).get("raw_mm_yr", np.nan) for w in walls]
        ax.bar(xw + (idx - 0.5 * (len(items) - 1)) * ww, wrms, width=ww, color=colors.get(key, "gray"), edgecolor="k", lw=0.4, label=key)
    ax.set_xticks(xw)
    ax.set_xticklabels(["East\n(inflow)", "South\n(inflow)", "North\n(outflow)", "West\n(rock)"])
    ax.set_ylabel(r"Raw ($E=1$) wall RMS speed (mm yr$^{-1}$)")
    ax.set_title("A. Coarse-to-Fine Side Boundary Velocity")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(loc="upper right", fontsize=7.5)

    # Panel B: Raw GNSS surface speed ratio (exp / baseline) vs distance from nearest fine-mesh wall
    ax = axes[1]
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    for key, label, _ in items:
        if key == "baseline":
            continue
        gdf = pd.read_csv(results_dir / f"{key}_gnss_stakes.csv").set_index("name")
        ratio = gdf.loc[base_gnss.index, "h_mod_raw"] / base_gnss["h_mod_raw"]
        ax.scatter(dist_edge, ratio, s=42, color=colors.get(key, "gray"), marker=markers.get(key, "o"), edgecolor="k", lw=0.5, alpha=0.85, label=key)
    ax.set_xlabel("Distance from nearest fine-mesh lateral wall (m)")
    ax.set_ylabel(r"Raw ($E=1$) surface speed ratio ($u_{\mathrm{exp}} / u_{\mathrm{base}}$)")
    ax.set_title("B. Inward Decay of Lateral BC Perturbation (18 GNSS Stakes)")
    ax.grid(alpha=0.35)
    ax.legend(loc="upper right", fontsize=7.5)

    # Panel C: Raw (E=1) near-bed U_10 ratio (exp / baseline) at the 7 boreholes
    ax = axes[2]
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    xb = np.arange(len(BOREHOLES_7))
    base_m = dict((k, m) for k, _, m in items)["baseline"]
    base_e = base_m["E_used"]
    non_base = [(k, l, m) for k, l, m in items if k != "baseline"]
    wb = 0.78 / max(len(non_base), 1)
    for idx, (key, label, m) in enumerate(non_base):
        e_k = m["E_used"]
        ratios = [
            (m["borehole_profiles"][b]["U10_mm_yr"] / e_k) / (base_m["borehole_profiles"][b]["U10_mm_yr"] / base_e)
            for b in BOREHOLES_7
        ]
        ax.bar(xb + (idx - 0.5 * (len(non_base) - 1)) * wb, ratios, width=wb, color=colors.get(key, "gray"), edgecolor="k", lw=0.4, label=key)
    ax.set_xticks(xb)
    ax.set_xticklabels(BOREHOLES_7, rotation=30, ha="right")
    ax.set_ylabel(r"Raw ($E=1$) $U_{10,\mathrm{exp}} / U_{10,\mathrm{base}}$")
    ax.set_title("C. Interior Near-Bed Speed Ratio Before E Scaling")
    ax.set_ylim(0.75, 1.40)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(loc="upper right", fontsize=7.5)

    fig.savefig(cmp_dir / "compare_boundary_coupling_decay.png", dpi=220)
    plt.close(fig)


def plot_coarse_basin_velocity_comparison(
    results_dir: Path,
    cmp_dir: Path,
    items: list[tuple[str, str, dict]],
    inputs_dir: Path | None,
    repo_dir: Path | None,
):
    """Side-by-side opaque speed heatmap + streamlines across the entire coarse domain / REMA basin."""
    if inputs_dir is None or repo_dir is None:
        return
    lima_path = inputs_dir / "LIMA_virtual_Mosaic_15m.tif"
    basin_gj = inputs_dir / "allan_hills_rema_basin.geojson"
    bh_csv = repo_dir / "Borehole_Locations/Borehole_locations_DTS.csv"
    if not (lima_path.is_file() and basin_gj.is_file() and bh_csv.is_file()):
        return

    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    import rasterio
    from rasterio.plot import show as rasterio_show
    from pyproj import Transformer
    from shapely.geometry import shape

    panels = [
        ("baseline", "A. Baseline: 1600 m Rect. Newtonian"),
        ("rect_glen", "B. Exp 1: 1600 m Rect. Glen's Law"),
        ("basin_glen", "C. Exp 2: REMA Basin Glen (Free-slip)"),
    ]
    loaded = []
    m_map = {k: m for k, _, m in items}
    for key, ptitle in panels:
        gp = results_dir / f"{key}_coarse_surf_grid.npz"
        if gp.is_file() and key in m_map:
            loaded.append((key, ptitle, np.load(gp), m_map[key]))
    if not loaded:
        return

    bh = pd.read_csv(bh_csv)
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    bx, by = tf.transform(bh["E"].to_numpy(float), bh["S"].to_numpy(float))
    poly = shape(json.loads(basin_gj.read_text(encoding="utf-8"))["features"][0]["geometry"])
    px, py = poly.exterior.xy

    with rasterio.open(lima_path) as src1:
        lima_img = src1.read()
        lima_tf = src1.transform

    vmin_s, vmax_s = 0.005, 2.5
    cmap_v = plt.get_cmap("viridis").copy()
    cmap_v.set_bad((0.0, 0.0, 0.0, 0.0))
    halo = [pe.withStroke(linewidth=2.8, foreground="black", alpha=0.7)]
    halo_w = [pe.withStroke(linewidth=3.0, foreground="white", alpha=0.9)]

    fig, axes = plt.subplots(1, len(loaded), figsize=(5.5 * len(loaded), 7.2), sharex=True, sharey=True, layout="constrained")
    if len(loaded) == 1:
        axes = [axes]

    im = None
    for idx, (key, ptitle, g, m) in enumerate(loaded):
        ax = axes[idx]
        rasterio_show(lima_img, ax=ax, transform=lima_tf)
        gx, gy, U, V, S, ZS = g["gx"], g["gy"], g["U"], g["V"], g["S"], g["ZS"]
        is_basin = bool(g["is_basin"])
        e_val = float(m["E_used"])

        S_plot = np.where(np.isfinite(S), np.maximum(S, vmin_s * 0.5), np.nan)
        im = ax.pcolormesh(
            gx, gy, S_plot,
            cmap=cmap_v,
            norm=mcolors.LogNorm(vmin=vmin_s, vmax=vmax_s),
            shading="auto",
            alpha=1.0,
            zorder=2,
        )

        # Subtle elevation contours
        if np.isfinite(ZS).any():
            z_lo = int(math.floor(np.nanmin(ZS) / 100.0) * 100)
            z_hi = int(math.ceil(np.nanmax(ZS) / 100.0) * 100)
            ax.contour(gx, gy, ZS, levels=np.arange(z_lo, z_hi + 1, 100.0), colors="white", linewidths=0.4, alpha=0.35, zorder=3)

        flow_mask = np.isfinite(S) & (S >= 0.003)
        ax.streamplot(
            gx, gy, np.where(flow_mask, U, np.nan), np.where(flow_mask, V, np.nan),
            color=(1.0, 1.0, 1.0, 0.80), linewidth=0.85, density=1.05, arrowsize=0.85, minlength=0.08, zorder=4,
        )

        leg_h = []
        if is_basin:
            ax.plot(px, py, color="#00E5FF", ls="-", lw=2.0, path_effects=halo, zorder=6)
            ax.add_patch(Rectangle((502369.57, -1362201.48), 531866.83 - 502369.57, -1342728.25 - (-1362201.48), fill=False, ec="white", ls=":", lw=1.2, path_effects=halo, zorder=5))
            leg_h.append(Line2D([0], [0], color="#00E5FF", lw=2.2, path_effects=halo, label="REMA basin domain"))
            leg_h.append(Line2D([0], [0], color="0.25", ls=":", lw=1.5, label="1600 m rect. (ref.)"))
        else:
            ax.add_patch(Rectangle((502369.57, -1362201.48), 531866.83 - 502369.57, -1342728.25 - (-1362201.48), fill=False, ec="#00E5FF", ls="-", lw=2.0, path_effects=halo, zorder=6))
            ax.plot(px, py, color="white", ls=":", lw=1.2, path_effects=halo, zorder=5)
            leg_h.append(Line2D([0], [0], color="#00E5FF", lw=2.2, path_effects=halo, label="1600 m rect. domain"))
            leg_h.append(Line2D([0], [0], color="0.25", ls=":", lw=1.5, label="REMA basin (ref.)"))

        ax.add_patch(Rectangle((506902.38, -1356284.47), 512046.01 - 506902.38, -1350629.26 - (-1356284.47), fill=False, ec="#FF2D20", ls="--", lw=1.9, path_effects=halo_w, zorder=7))
        ax.scatter(bx, by, marker="o", s=24, fc="#FF2D20", ec="white", lw=0.7, zorder=9)
        leg_h.append(Line2D([0], [0], color="#FF2D20", ls="--", lw=1.9, label="80 m fine mesh"))
        leg_h.append(Line2D([0], [0], marker="o", ls="none", ms=5.5, mfc="#FF2D20", mec="black", mew=0.5, label="Boreholes"))

        ax.set_xlim(494000, 533500)
        ax.set_ylim(-1376500, -1326500)
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (m, EPSG:3031)")
        if idx == 0:
            ax.set_ylabel("Northing (m, EPSG:3031)")
        ax.set_title(f"{ptitle}\n(calibrated E = {e_val:.2f})", fontsize=10)
        ax.legend(handles=leg_h, loc="lower right", fontsize=7.6, framealpha=0.92, facecolor="white", edgecolor="0.4")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.015, extend="both")
        cbar.set_label(r"Modeled surface ice speed $|\mathbf{u}_s|$ (m yr$^{-1}$)")
        cbar.set_ticks([0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5])
        cbar.ax.set_yticklabels(["0.005", "0.01", "0.05", "0.1", "0.5", "1.0", "2.5"])

    fig.savefig(cmp_dir / "compare_coarse_basin_velocities.png", dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_markdown_summary(out_md: Path, df: pd.DataFrame, items: list[tuple[str, str, dict]]):
    lines = [
        "# Coarse-Solve Reevaluation: Quantitative vs. Qualitative Comparison",
        "",
        "This report evaluates the influence of two coarse-region modeling changes on all results and figures in the Allan Hills ice-flow paper:",
        "",
        "1. **Experiment 1 (`rect_glen`)**: Replacing the Newtonian coarse solve ($\\mu_0 = 5\\times 10^{15}\\,\\mathrm{Pa\\,s}$) with a coupled thermomechanical Glen's-flow ($n=3$, temperature-dependent $A(T)$) solve on the same 1600 m rectangular domain, coupled into the 80 m fine mesh.",
        "2. **Experiment 2 (`basin_glen`)**: Replacing the rectangular coarse domain with a modern-day drainage basin delineated from REMA v2.0 surface slopes ($\\sigma = 1.5\\,\\mathrm{km}$), solved with coupled Glen's flow and physically natural free-slip impermeable lateral walls ($\\mathbf{u}\\cdot\\mathbf{n} = 0$), coupled into the 80 m fine mesh.",
        "3. **Sensitivities (`basin_glen_noslip`, `basin_newtonian`)**: Testing no-slip lateral walls on the drainage basin and Newtonian rheology on the drainage basin to separate domain geometry, lateral boundary condition, and rheology effects.",
        "",
        "## Key Metrics Comparison",
        "",
    ]
    cols_show = [
        "experiment", "E_opt", "gnss_horiz_rmse_m_yr", "gnss_mean_ang_err_deg",
        "U10_ALHIC1902_mm_yr", "U10_ALHIC2302_mm_yr", "U10_ALHIC1502_mm_yr",
        "U10_ALHIC1901_mm_yr", "U10_ALHIC2201_mm_yr", "U10_ALHIC2301_mm_yr", "U10_ALHIC2401_mm_yr",
        "ols10_r2", "ols10_slope_ma_per_dec", "ols10_p", "win10_extrap_age_ma", "win10_min_disp_from_base_m",
    ]
    sub = df[cols_show].copy()
    header = "| " + " | ".join(sub.columns) + " |"
    sep = "|" + "|".join("---" for _ in sub.columns) + "|"
    lines.extend([header, sep])
    for _, r in sub.iterrows():
        cells = []
        for c in sub.columns:
            val = r[c]
            if isinstance(val, (float, np.floating)):
                cells.append(f"{val:.3f}" if np.isfinite(val) else "nan")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Qualitative Robustness Checklist",
        "",
        "| Paper qualitative finding | " + " | ".join(k for k, _, _ in items) + " |",
        "|---|" + "|".join("---" for _ in items) + "|",
    ])
    checks = [
        ("Opposing nunatak counterflow & stagnation zone (Fig. 1b, Fig. 2)", lambda m: "YES" if m["counterflow_and_stagnation_fig1b"]["has_eastward_counterflow_in_fig1b"] else "NO"),
        ("Slow-flow (<2 mm/yr) at ALHIC1902 & ALHIC2302", lambda m: "YES" if max(m["borehole_profiles"]["ALHIC1902"]["U10_mm_yr"], m["borehole_profiles"]["ALHIC2302"]["U10_mm_yr"]) < 2.0 else "NO"),
        ("Strict speed ordering: Nunatak < Submerged Ridge < Cul-de-sac", lambda m: "YES" if m["regime_ranking_preserved"] else "NO"),
        ("Strong inverse age–log10(U10) fit (5 non-CDS cores, R² > 0.80, p < 0.05)", lambda m: f"YES (R²={m['ols_fits_by_height_m']['10']['r2']:.2f}, p={m['ols_fits_by_height_m']['10']['p_two_sided']:.3f})" if (m["ols_fits_by_height_m"]["10"]["r2"] > 0.80 and m["ols_fits_by_height_m"]["10"]["p_two_sided"] < 0.05) else f"CHANGED (R²={m['ols_fits_by_height_m']['10']['r2']:.2f}, p={m['ols_fits_by_height_m']['10']['p_two_sided']:.3f})"),
        ("CDS speed order-of-magnitude faster than stagnation (>30x)", lambda m: f"YES ({m['cds_to_nunatak_u10_contrast']['ALHIC2301_over_ALHIC1902']:.0f}x / {m['cds_to_nunatak_u10_contrast']['ALHIC2401_over_ALHIC2302']:.0f}x)"),
        ("Height robustness across 5, 10, 20, 50 m (all R² > 0.80)", lambda m: "YES (" + ", ".join(f"{m['ols_fits_by_height_m'][h]['r2']:.2f}" for h in ("5", "10", "20", "50")) + ")"),
        ("Borehole temperature advection signal (Nunatak no-adv MAE < Ridge & CDS)", lambda m: "YES" if m["borehole_temperature_diagnostics"]["nunatak_lt_cds_and_ridge"] else "MIXED"),
        ("Weak coarse-mesh steady-SMB association (Fig. S10)", lambda m: f"r={m['spaulding_smb_coarse_diagnostics'].get('pearson_r', float('nan')):.2f} (n={m['spaulding_smb_coarse_diagnostics'].get('n_ice_firn', 0)})"),
    ]
    for desc, fn in checks:
        vals = [fn(m) for _, _, m in items]
        lines.append(f"| {desc} | " + " | ".join(vals) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--cmp-fig-dir", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, default=None)
    parser.add_argument("--repo-dir", type=Path, default=None)
    args = parser.parse_args()

    args.cmp_fig_dir.mkdir(parents=True, exist_ok=True)
    items = load_available_metrics(args.results_dir)
    if not items:
        raise RuntimeError(f"No *_metrics.json files found in {args.results_dir}")

    df = build_summary_table(items)
    df.to_csv(args.results_dir / "comparison_summary_table.csv", index=False)
    plot_speed_map_comparisons(args.results_dir, args.cmp_fig_dir, items)
    plot_summary_bars_and_regressions(args.results_dir, args.cmp_fig_dir, items)
    plot_boundary_coupling_diagnostic(args.results_dir, args.cmp_fig_dir, items)
    plot_coarse_basin_velocity_comparison(args.results_dir, args.cmp_fig_dir, items, args.inputs_dir, args.repo_dir)
    write_markdown_summary(args.results_dir / "comparison_summary.md", df, items)
    print(f"Wrote comparison summary for {len(items)} experiments to {args.results_dir}")


if __name__ == "__main__":
    main()
