#!/usr/bin/env python3
"""Evaluate one coarse-to-fine experiment and regenerate EVERY paper & SI figure + metrics.

Loads either:
  - Saved CheckpointFile outputs (`coarse_solution.h5`, `fine_solution.h5`, `run_diagnostics.json`)
    produced by `run_coarse_and_fine_experiments.py`, OR
  - Legacy baseline DumbCheckpoints (`--legacy-baseline`) to reproduce the exact published baseline
    through the identical evaluation and plotting pipeline for apples-to-apples comparison.

Must be run on 1 MPI rank (serial) so Firedrake point evaluation `.at(...)` operates on the full mesh.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import string
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.plot import show as rasterio_show
import scipy.interpolate
import scipy.signal
import scipy.stats
from pyproj import Transformer
from shapely.geometry import Point, shape

import firedrake as fd
from firedrake import (
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    SpatialCoordinate,
    VectorFunctionSpace,
    ds_b,
    grad,
    inner,
)

# Import mesh & physical helpers from sibling runner
from run_coarse_and_fine_experiments import (
    CP,
    K_THERM,
    N_LAYERS,
    RHO,
    SPY,
    T_LAPSE_RATE,
    T_SURFACE_MEAN,
    build_basin_extruded_mesh,
    build_rect_extruded_mesh,
    load_resampled_raster_dem,
)

HEIGHTS_M = (5.0, 10.0, 20.0, 50.0)
FIT_CORES = ("ALHIC1502", "ALHIC1901", "ALHIC1902", "ALHIC2201", "ALHIC2302")
DISPLAY_ONLY_CORES = ("ALHIC2301",)
ALL_AGE_CORES = FIT_CORES + DISPLAY_ONLY_CORES
RELEASED_NUNATAK_WINDOW = (509000.0, 511000.0, -1355100.0, -1352500.0)
MANUSCRIPT_WINDOW = (507500.0, 512000.0, -1355500.0, -1351000.0)
CORE_SURFACE_Z = {
    "ALHIC1901": 1937.88,
    "ALHIC1902": 1940.76,
    "ALHIC1903": 1884.84,
    "ALHIC2201": 1934.54,
    "ALHIC2301": 1865.68,
    "ALHIC2302": 1940.76,
    "ALHIC2401": 1865.68,
}
FLOW_GROUP = {
    "ALHIC1902": "nunatak_stagnation",
    "ALHIC2302": "nunatak_stagnation",
    "ALHIC1502": "collocated_ridge",
    "ALHIC1901": "collocated_ridge",
    "ALHIC2201": "distinct_ridge",
}
FLOW_COLORS = {
    "nunatak_stagnation": "#0072B2",
    "collocated_ridge": "#D55E00",
    "distinct_ridge": "#009E73",
}


def configure_matplotlib():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def eval_points_safe(fn, pts: np.ndarray, dim: int = 3, chunk: int = 2000) -> np.ndarray:
    """Batch-evaluate a Firedrake Function at 3D (or 2D) points, returning NaN where outside domain."""
    pts = np.asarray(pts, dtype=float)
    out = np.full((len(pts), dim) if dim > 1 else (len(pts),), np.nan, dtype=float)
    for start in range(0, len(pts), chunk):
        stop = min(start + chunk, len(pts))
        sub = pts[start:stop]
        try:
            vals = np.asarray(fn.at(sub, dont_raise=True), dtype=float)
            # dont_raise returns None entries for points outside domain
            if vals.dtype == object:
                raise ValueError("object array from dont_raise")
            out[start:stop] = vals.reshape(out[start:stop].shape)
            continue
        except Exception:
            pass
        for k, pt in enumerate(sub, start=start):
            try:
                v = fn.at(tuple(pt))
                if v is not None:
                    arr = np.asarray(v, dtype=float)
                    out[k] = arr if dim > 1 else float(arr.reshape(-1)[0])
            except Exception:
                continue
    return out


def eval_scalar_2d(fn, pts2d: np.ndarray) -> np.ndarray:
    return eval_points_safe(fn, pts2d, dim=1, chunk=2000)


def load_models(args, inp: Path):
    """Load coarse & fine meshes, flow functions, steady temperatures, and base bed/surf fields."""
    # Always reconstruct fine base bed/surface for exact elevation queries
    fine_dem = load_resampled_raster_dem(
        inp / "main_boreholes_aoi_fix_bed.tif",
        inp / "main_boreholes_aoi_fix_surf.tif",
        upscale_factor=0.1,
    )
    mesh_fine_ref, _, base_fine, base_bed_f, base_surf_f = build_rect_extruded_mesh(
        fine_dem, layers=N_LAYERS, name="fine_80m"
    )
    base_bed_f.ufl_domain().tolerance = 1e-8
    base_surf_f.ufl_domain().tolerance = 1e-8

    if args.legacy_baseline:
        from firedrake.checkpointing import DumbCheckpoint, FILE_READ

        coarse_dem = load_resampled_raster_dem(
            inp / "bedmachine_ian_merged_bed_large_aoi_8m_aligned_fixed.tif",
            inp / "rema_large_aoi_align_fixed.tif",
            upscale_factor=1.0 / 200.0,
        )
        mesh_coarse, _, base_coarse, base_bed_c, base_surf_c = build_rect_extruded_mesh(
            coarse_dem, layers=N_LAYERS, name="coarse_rect_1600m"
        )
        mesh_coarse.tolerance = 1e-8
        Yc = VectorFunctionSpace(mesh_coarse, "CG", 3) * FunctionSpace(mesh_coarse, "CG", 1)
        flow_coarse = Function(Yc, name="coarse_flow")
        chk_c = DumbCheckpoint(str(inp / "coarse_mesh_flow_final_model"), mode=FILE_READ)
        chk_c.load(flow_coarse, name="coarse_mesh_flow")
        chk_c.close()

        mesh_fine = mesh_fine_ref
        mesh_fine.tolerance = 1e-8
        Yf = VectorFunctionSpace(mesh_fine, "CG", 3) * FunctionSpace(mesh_fine, "CG", 1)
        flow_fine = Function(Yf, name="fine_flow")
        chk_f = DumbCheckpoint(str(inp / "fine_mesh_flow_80.0m_updated_viscosity"), mode=FILE_READ)
        chk_f.load(flow_fine, name="fine_mesh_flow")
        chk_f.close()

        Vf = FunctionSpace(mesh_fine, "CG", 1)
        T_fine = Function(Vf, name="fine_temp")
        chk_t = DumbCheckpoint(str(inp / "fine_mesh_temperature_80.0m_updated_viscosity"), mode=FILE_READ)
        chk_t.load(T_fine, name="fine_mesh_temp")
        chk_t.close()

        # Load legacy viscosity residuals pickle for Fig S7
        import pickle as pkl

        res_pkl = args.repo_dir / "scripts" / "residuals_full_model.pkl"
        with res_pkl.open("rb") as fh:
            abs_r, rel_r = pkl.load(fh)
        run_diag = {
            "experiment": "baseline",
            "coarse": {"lateral_mode": "noslip", "glen": False, "Y_dim": int(Yc.dim())},
            "fine": {
                "Y_dim": int(Yf.dim()),
                "theta_abs": [float(np.asarray(x).reshape(-1)[0]) for x in abs_r],
                "theta_rel": [float(np.asarray(x).reshape(-1)[0]) for x in rel_r],
            },
        }
        is_basin = False
    else:
        exp_dir = args.exp_dir
        run_diag = json.loads((exp_dir / "run_diagnostics.json").read_text(encoding="utf-8"))
        is_basin = args.experiment.startswith("basin")

        if is_basin:
            _, _, base_coarse, base_bed_c, base_surf_c = build_basin_extruded_mesh(
                inp / "allan_hills_basin_2d.msh",
                inp / "allan_hills_basin_dems.npz",
                layers=N_LAYERS,
                name="coarse_basin",
            )
            coarse_mesh_name = "coarse_basin"
        else:
            coarse_dem = load_resampled_raster_dem(
                inp / "bedmachine_ian_merged_bed_large_aoi_8m_aligned_fixed.tif",
                inp / "rema_large_aoi_align_fixed.tif",
                upscale_factor=1.0 / 200.0,
            )
            _, _, base_coarse, base_bed_c, base_surf_c = build_rect_extruded_mesh(
                coarse_dem, layers=N_LAYERS, name="coarse_rect_1600m"
            )
            coarse_mesh_name = "coarse_rect_1600m"

        with fd.CheckpointFile(str(exp_dir / "coarse_solution.h5"), "r") as chk:
            mesh_coarse = chk.load_mesh(coarse_mesh_name)
            flow_coarse = chk.load_function(mesh_coarse, "coarse_flow")
        mesh_coarse.tolerance = 1e-8

        with fd.CheckpointFile(str(exp_dir / "fine_solution.h5"), "r") as chk:
            mesh_fine = chk.load_mesh("fine_80m")
            flow_fine = chk.load_function(mesh_fine, "fine_flow")
            T_fine = chk.load_function(mesh_fine, "fine_temp")
        mesh_fine.tolerance = 1e-8

    base_bed_c.ufl_domain().tolerance = 1e-8
    base_surf_c.ufl_domain().tolerance = 1e-8
    return {
        "mesh_coarse": mesh_coarse,
        "flow_coarse": flow_coarse,
        "u_coarse": flow_coarse.subfunctions[0],
        "base_coarse": base_coarse,
        "base_bed_c": base_bed_c,
        "base_surf_c": base_surf_c,
        "mesh_fine": mesh_fine,
        "flow_fine": flow_fine,
        "u_fine": flow_fine.subfunctions[0],
        "T_fine": T_fine,
        "base_fine": base_fine,
        "base_bed_f": base_bed_f,
        "base_surf_f": base_surf_f,
        "fine_dem": fine_dem,
        "run_diag": run_diag,
        "is_basin": is_basin,
    }


def calibrate_enhancement_factor(u_fine, base_surf_f, gps_df: pd.DataFrame):
    """Calibrate single uniform E against the 18 fine-mesh GNSS stakes (exact match to figs5.ipynb & Methods)."""
    xs = gps_df["x"].to_numpy(float)
    ys = gps_df["y"].to_numpy(float)
    surf_z = eval_scalar_2d(base_surf_f, np.column_stack([xs, ys]))

    rows = []
    for idx, r in gps_df.iterrows():
        sz = surf_z[idx]
        if not np.isfinite(sz):
            continue
        # Evaluate just inside top surface (0.05 m below surface to avoid boundary roundoff)
        v = eval_points_safe(u_fine, np.array([[r["x"], r["y"], sz - 0.05]]), dim=3)[0]
        if not np.isfinite(v).all():
            v = eval_points_safe(u_fine, np.array([[r["x"], r["y"], sz - 0.5]]), dim=3)[0]
        if not np.isfinite(v).all():
            continue
        rows.append(
            {
                "name": str(r["Name"]),
                "x": float(r["x"]),
                "y": float(r["y"]),
                "z_surf": float(sz),
                "u_obs": float(r["u"]),
                "v_obs": float(r["v"]),
                "w_obs": float(r["w"]),
                "h_obs": float(math.hypot(r["u"], r["v"])),
                "h_unc": float(r["Horizontal uncertainty"]),
                "v_unc": float(r["Vertical uncertainty"]),
                "u_mod_raw": float(v[0] * SPY),
                "v_mod_raw": float(v[1] * SPY),
                "w_mod_raw": float(v[2] * SPY),
                "h_mod_raw": float(math.hypot(v[0] * SPY, v[1] * SPY)),
            }
        )
    df = pd.DataFrame(rows)
    h_obs = df["h_obs"].to_numpy(float)
    h_mod = df["h_mod_raw"].to_numpy(float)
    n = len(df)

    # Exact closed-form OLS optimum for E minimizing sum((E*h_mod - h_obs)^2)
    e_exact = float(np.dot(h_mod, h_obs) / np.dot(h_mod, h_mod))
    resid_opt = e_exact * h_mod - h_obs
    sse_opt = float(np.sum(resid_opt ** 2))
    rmse_opt = float(math.sqrt(sse_opt / n))
    # 95% CI for one-parameter fit (n-1 dof, matching Methods line 338)
    s2 = sse_opt / (n - 1)
    se_e = float(math.sqrt(s2 / np.dot(h_mod, h_mod)))
    t_crit = float(scipy.stats.t.ppf(0.975, df=n - 1))
    e_ci_low = e_exact - t_crit * se_e
    e_ci_high = e_exact + t_crit * se_e

    # Also compute curve over dense E grid for Fig S1 inset
    e_max_plot = max(3.5, e_exact * 1.8)
    e_min_plot = max(0.05, min(0.5, e_exact * 0.3))
    e_grid = np.linspace(e_min_plot, e_max_plot, 250)
    rmse_grid = np.array([math.sqrt(float(np.mean((e * h_mod - h_obs) ** 2))) for e in e_grid])

    # Angular error (direction) independent of E
    ang_obs = np.arctan2(df["v_obs"], df["u_obs"])
    ang_mod = np.arctan2(df["v_mod_raw"], df["u_mod_raw"])
    ang_err = np.degrees(np.abs(np.arctan2(np.sin(ang_mod - ang_obs), np.cos(ang_mod - ang_obs))))

    # Vertical RMSE at E_opt
    w_resid = e_exact * df["w_mod_raw"].to_numpy(float) - df["w_obs"].to_numpy(float)
    w_rmse = float(math.sqrt(float(np.mean(w_resid ** 2))))

    # Also evaluate RMSE at fixed E=1.97 for reference
    rmse_at_197 = float(math.sqrt(float(np.mean((1.97 * h_mod - h_obs) ** 2))))

    df["h_resid_opt"] = e_exact * df["h_mod_raw"] - df["h_obs"]
    df["w_resid_opt"] = e_exact * df["w_mod_raw"] - df["w_obs"]
    df["ang_err_deg"] = ang_err

    return {
        "n_stakes": n,
        "E_opt": e_exact,
        "E_ci_low": e_ci_low,
        "E_ci_high": e_ci_high,
        "E_se": se_e,
        "horiz_rmse_m_yr": rmse_opt,
        "vert_rmse_m_yr": w_rmse,
        "mean_ang_err_deg": float(np.mean(ang_err)),
        "median_ang_err_deg": float(np.median(ang_err)),
        "horiz_rmse_at_E197_m_yr": rmse_at_197,
        "stakes_df": df,
        "e_grid": e_grid,
        "rmse_grid": rmse_grid,
    }


def run_transient_temperatures(mesh_fine, u_fine, T_steady, e_scale: float):
    """Run 150-yr transient temperature (2 K linear warming, 15 x 10 yr Backward Euler, geo_flux=0.04)
    both with advection (E * u) and without advection (u = 0), matching fig2_v3.ipynb.
    """
    V = T_steady.function_space()
    _, _, z = SpatialCoordinate(mesh_fine)
    dX = fd.Measure("dx", domain=mesh_fine)
    dSb = fd.Measure("ds_b", domain=mesh_fine)
    phi = fd.TestFunction(V)

    dt = Constant(10.0 * SPY)
    n_steps = 15
    geo_flux = Constant(0.04)
    T_mean = Constant(T_SURFACE_MEAN)
    T_surf_expr = T_mean - (z - Constant(2000.0)) * Constant(T_LAPSE_RATE)
    bc_top = DirichletBC(V, T_surf_expr, "top")

    def integrate(with_advection: bool):
        T_curr = T_steady.copy(deepcopy=True)
        T_prev = T_steady.copy(deepcopy=True)
        T_trial = Function(V)
        F_mass = Constant(RHO * CP) * ((T_trial - T_prev) / dt) * phi * dX
        F_diff = Constant(K_THERM) * inner(grad(T_trial), grad(phi)) * dX
        F_geo = -geo_flux * phi * dSb
        if with_advection:
            u_scaled = Constant(e_scale) * u_fine
            # Exact grad(T).n = 0 advection weak form (matches steady solver)
            F_adv = Constant(RHO * CP) * inner(u_scaled, grad(T_trial)) * phi * dX
            F = F_mass + F_diff + F_adv + F_geo
        else:
            F = F_mass + F_diff + F_geo
        for step in range(n_steps):
            # Matching fig2_v3.ipynb: t = ts[step] = step * 10 yr, warming = (t / 150) * 2 K
            T_mean.assign(T_SURFACE_MEAN + (step / float(n_steps)) * 2.0)
            fd.solve(
                F == 0,
                T_trial,
                bcs=[bc_top],
                solver_parameters={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
            )
            T_prev.assign(T_trial)
            T_curr.assign(T_trial)
        return T_curr

    T_clim = integrate(with_advection=True)
    T_no_adv = integrate(with_advection=False)
    return T_clim, T_no_adv


def load_geochem_and_ages(geochem_xlsx: Path, age_recon_csv: Path):
    xl = pd.ExcelFile(geochem_xlsx)
    recon = pd.read_csv(age_recon_csv).set_index("core_id")
    ages = {}
    d38_means = {}
    d18_means = {}
    for core_id in ALL_AGE_CORES:
        df = xl.parse(core_id)
        ages_ka = pd.to_numeric(df["Argon age (ka)"], errors="coerce")
        raw_max_ma = float(ages_ka.max() / 1000.0)
        ages[core_id] = float(recon.loc[core_id, "reconciled_max_age_ma"]) if core_id in recon.index else raw_max_ma
        young = df[ages_ka < 1500.0]
        d38_means[core_id] = float(pd.to_numeric(young["δ38/36Ar (‰)"], errors="coerce").dropna().mean())
        d18_means[core_id] = float(pd.to_numeric(young["del18OSMOW"], errors="coerce").dropna().mean())
    return ages, d38_means, d18_means


def fit_ols_age_on_log10_speed(speeds_mm_yr: np.ndarray, ages_ma: np.ndarray):
    x = np.log10(np.asarray(speeds_mm_yr, dtype=float))
    y = np.asarray(ages_ma, dtype=float)
    n = len(x)
    dof = n - 2
    xbar = float(np.mean(x))
    ybar = float(np.mean(y))
    sxx = float(np.sum((x - xbar) ** 2))
    sxy = float(np.sum((x - xbar) * (y - ybar)))
    syy = float(np.sum((y - ybar) ** 2))
    beta = sxy / sxx
    alpha = ybar - beta * xbar
    resid = y - (alpha + beta * x)
    sse = float(np.sum(resid ** 2))
    rse = math.sqrt(sse / dof)
    r2 = 1.0 - sse / syy
    r = sxy / math.sqrt(sxx * syy)
    se_beta = rse / math.sqrt(sxx)
    t_stat = beta / se_beta
    p_val = float(2.0 * scipy.stats.t.sf(abs(t_stat), df=dof))
    t_crit = float(scipy.stats.t.ppf(0.975, df=dof))
    return {
        "n": n,
        "dof": dof,
        "alpha": alpha,
        "beta": beta,
        "beta_ci_low": beta - t_crit * se_beta,
        "beta_ci_high": beta + t_crit * se_beta,
        "rse": rse,
        "r2": r2,
        "r": r,
        "p_two_sided": p_val,
        "xbar": xbar,
        "sxx": sxx,
        "t_crit": t_crit,
        "log_speed_min": float(x.min()),
        "log_speed_max": float(x.max()),
    }


def predict_age_ols(fit: dict, speeds_mm_yr: np.ndarray):
    x0 = np.log10(np.asarray(speeds_mm_yr, dtype=float))
    mean = fit["alpha"] + fit["beta"] * x0
    se_mean = fit["rse"] * np.sqrt(1.0 / fit["n"] + (x0 - fit["xbar"]) ** 2 / fit["sxx"])
    se_pred = fit["rse"] * np.sqrt(1.0 + 1.0 / fit["n"] + (x0 - fit["xbar"]) ** 2 / fit["sxx"])
    tc = fit["t_crit"]
    return {
        "mean": mean,
        "ci_low": mean - tc * se_mean,
        "ci_high": mean + tc * se_mean,
        "pi_low": mean - tc * se_pred,
        "pi_high": mean + tc * se_pred,
    }


def evaluate_age_speed_and_grid(u_fine, base_bed_f, base_surf_f, fine_dem, boreholes_df, ages, e_scale: float):
    """Exact-height sampling at 5, 10, 20, 50 m for all 6 cores and full fine-mesh grid."""
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    bh = boreholes_df.copy()
    bh["x_m"], bh["y_m"] = tf.transform(bh["E"].to_numpy(float), bh["S"].to_numpy(float))
    bh_idx = bh.set_index("Core")

    # 1. Core samples at 5, 10, 20, 50 m
    core_rows = []
    for core_id in ALL_AGE_CORES:
        x_m = float(bh_idx.loc[core_id, "x_m"])
        y_m = float(bh_idx.loc[core_id, "y_m"])
        z_bed = float(eval_scalar_2d(base_bed_f, np.array([[x_m, y_m]]))[0])
        z_surf = float(eval_scalar_2d(base_surf_f, np.array([[x_m, y_m]]))[0])
        for h in HEIGHTS_M:
            v = eval_points_safe(u_fine, np.array([[x_m, y_m, z_bed + h]]), dim=3)[0]
            spd_mm = float(np.linalg.norm(v) * SPY * 1000.0 * e_scale)
            core_rows.append(
                {
                    "core_id": core_id,
                    "height_m": h,
                    "x_m": x_m,
                    "y_m": y_m,
                    "bed_elevation_m": z_bed,
                    "surface_elevation_m": z_surf,
                    "speed_mm_yr": spd_mm,
                    "max_age_ma": ages[core_id],
                }
            )
    core_df = pd.DataFrame(core_rows)

    # Also compute bottom-20 m average and full-column average horizontal/vertical speeds for all 7 boreholes in fine mesh
    borehole_profiles = {}
    for core_id in ("ALHIC1502", "ALHIC1901", "ALHIC1902", "ALHIC2201", "ALHIC2301", "ALHIC2302", "ALHIC2401"):
        x_m = float(bh_idx.loc[core_id, "x_m"])
        y_m = float(bh_idx.loc[core_id, "y_m"])
        z_bed = float(eval_scalar_2d(base_bed_f, np.array([[x_m, y_m]]))[0])
        z_surf = float(eval_scalar_2d(base_surf_f, np.array([[x_m, y_m]]))[0])
        zs = np.arange(z_bed + 0.5, z_surf - 0.5, 1.0)
        pts = np.column_stack([np.full_like(zs, x_m), np.full_like(zs, y_m), zs])
        vels = eval_points_safe(u_fine, pts, dim=3) * SPY * 1000.0 * e_scale  # mm/yr
        valid = np.isfinite(vels).all(axis=1)
        vels = vels[valid]
        zs = zs[valid]
        spd3 = np.linalg.norm(vels, axis=1)
        spdh = np.hypot(vels[:, 0], vels[:, 1])
        bot20 = (zs - z_bed) <= 20.0
        v10 = eval_points_safe(u_fine, np.array([[x_m, y_m, z_bed + 10.0]]), dim=3)[0] * SPY * 1000.0 * e_scale
        borehole_profiles[core_id] = {
            "x_m": x_m,
            "y_m": y_m,
            "z_bed": z_bed,
            "z_surf": z_surf,
            "thickness_m": z_surf - z_bed,
            "U10_mm_yr": float(np.linalg.norm(v10)),
            "bottom_20m_mean_speed_mm_yr": float(np.mean(spd3[bot20])) if bot20.any() else float("nan"),
            "col_mean_horiz_mm_yr": float(np.mean(spdh)),
            "col_mean_vert_mm_yr": float(np.mean(vels[:, 2])),
            "col_mean_abs_vert_mm_yr": float(np.mean(np.abs(vels[:, 2]))),
        }

    # 2. Full fine-mesh grid at 5, 10, 20, 50 m (exact coordinates of the 71 x 65 mesh nodes)
    base_coords = np.asarray(base_bed_f.ufl_domain().coordinates.dat.data_ro[:, :2], dtype=float)
    z_beds = np.asarray(base_bed_f.dat.data_ro, dtype=float)
    z_surfs = np.asarray(base_surf_f.dat.data_ro, dtype=float)
    grid_rows = []
    vel_10m_vectors = None
    for h in HEIGHTS_M:
        valid_h = (z_beds + h) < (z_surfs - 1e-6)
        pts = np.column_stack([base_coords[valid_h, 0], base_coords[valid_h, 1], z_beds[valid_h] + h])
        v_valid = eval_points_safe(u_fine, pts, dim=3) * SPY * 1000.0 * e_scale  # mm/yr
        spd_full = np.full(len(base_coords), np.nan, dtype=float)
        u_full = np.full((len(base_coords), 3), np.nan, dtype=float)
        ok = np.isfinite(v_valid).all(axis=1)
        idx_v = np.where(valid_h)[0]
        u_full[idx_v[ok]] = v_valid[ok]
        spd_full[idx_v[ok]] = np.linalg.norm(v_valid[ok], axis=1)
        if h == 10.0:
            vel_10m_vectors = {
                "x": base_coords[:, 0],
                "y": base_coords[:, 1],
                "bed": z_beds,
                "surf": z_surfs,
                "uvw_mm_yr": u_full,
                "speed_mm_yr": spd_full,
            }
        for i in range(len(base_coords)):
            grid_rows.append(
                {
                    "x_m": float(base_coords[i, 0]),
                    "y_m": float(base_coords[i, 1]),
                    "height_m": h,
                    "bed_elevation_m": float(z_beds[i]),
                    "surface_elevation_m": float(z_surfs[i]),
                    "speed_mm_yr": float(spd_full[i]) if np.isfinite(spd_full[i]) else np.nan,
                    "valid_domain": bool(np.isfinite(spd_full[i])),
                }
            )
    grid_df = pd.DataFrame(grid_rows)

    # 3. OLS fits & released-window minima at all 4 heights
    fits = {}
    window_minima = {}
    wxmin, wxmax, wymin, wymax = RELEASED_NUNATAK_WINDOW
    for h in HEIGHTS_M:
        sub_c = core_df[(core_df["height_m"] == h) & (core_df["core_id"].isin(FIT_CORES))]
        fit = fit_ols_age_on_log10_speed(sub_c["speed_mm_yr"].to_numpy(float), sub_c["max_age_ma"].to_numpy(float))
        fits[h] = fit
        sub_g = grid_df[
            (grid_df["height_m"] == h)
            & grid_df["valid_domain"]
            & grid_df["x_m"].between(wxmin, wxmax)
            & grid_df["y_m"].between(wymin, wymax)
        ].sort_values(["speed_mm_yr", "x_m", "y_m"])
        best = sub_g.iloc[0]
        pred = predict_age_ols(fit, np.array([float(best["speed_mm_yr"])]))
        window_minima[h] = {
            "x_m": float(best["x_m"]),
            "y_m": float(best["y_m"]),
            "speed_mm_yr": float(best["speed_mm_yr"]),
            "fitted_age_ma": float(pred["mean"][0]),
            "pi_low_ma": float(pred["pi_low"][0]),
            "pi_high_ma": float(pred["pi_high"][0]),
            "slow_zone_lt2_area_km2": float((sub_g["speed_mm_yr"] < 2.0).sum() * (80.0 * 80.0) / 1e6),
        }

    # Counterflow check on east flank of nunatak (zoom box Fig 1b: x in [508650, 509800], y in [-1353900, -1352800])
    v10 = vel_10m_vectors
    in_zoom = (
        (v10["x"] >= 508650.0)
        & (v10["x"] <= 509800.0)
        & (v10["y"] >= -1353900.0)
        & (v10["y"] <= -1352800.0)
        & np.isfinite(v10["speed_mm_yr"])
    )
    # Regional incoming flow has u < 0 (grid west); nunatak downslope counterflow has u > 0 (grid east)!
    u_zoom = v10["uvw_mm_yr"][in_zoom, 0]
    v_zoom = v10["uvw_mm_yr"][in_zoom, 1]
    w_zoom = v10["uvw_mm_yr"][in_zoom, 2]
    counterflow_mask = u_zoom > 0.0
    counterflow_diag = {
        "has_eastward_counterflow_in_fig1b": bool(counterflow_mask.any()),
        "counterflow_node_count": int(counterflow_mask.sum()),
        "max_eastward_u_mm_yr": float(np.max(u_zoom)) if len(u_zoom) else 0.0,
        "max_counterflow_horiz_mm_yr": float(np.max(np.hypot(u_zoom[counterflow_mask], v_zoom[counterflow_mask])))
        if counterflow_mask.any()
        else 0.0,
        "min_w_in_counterflow_mm_yr": float(np.min(w_zoom[counterflow_mask])) if counterflow_mask.any() else 0.0,
        "slow_lt2_nodes_in_fig1b": int((v10["speed_mm_yr"][in_zoom] < 2.0).sum()),
    }

    return core_df, grid_df, fits, window_minima, borehole_profiles, vel_10m_vectors, counterflow_diag


def evaluate_spaulding_smb(u_coarse, base_surf_c, spaulding_df: pd.DataFrame, e_scale: float):
    """Evaluate coarse-mesh steady-geometry SMB: a_modeled = E * (u_s . grad_h h - w_s) at Spaulding sites."""
    V_grad = VectorFunctionSpace(base_surf_c.ufl_domain(), "CG", 1)
    grad_s = fd.project(grad(base_surf_c), V_grad)

    rows = []
    for _, r in spaulding_df.iterrows():
        if not np.isfinite(r["observed_smb_m_ice_eq_yr"]):
            continue
        x_m, y_m = float(r["x_m"]), float(r["y_m"])
        sz = float(eval_scalar_2d(base_surf_c, np.array([[x_m, y_m]]))[0])
        if not np.isfinite(sz):
            continue
        gs = eval_points_safe(grad_s, np.array([[x_m, y_m]]), dim=2)[0]
        vel = eval_points_safe(u_coarse, np.array([[x_m, y_m, sz - 0.1]]), dim=3)[0]
        if not (np.isfinite(gs).all() and np.isfinite(vel).all()):
            vel = eval_points_safe(u_coarse, np.array([[x_m, y_m, sz - 1.0]]), dim=3)[0]
        if not (np.isfinite(gs).all() and np.isfinite(vel).all()):
            continue
        u_yr = vel * SPY
        a_unscaled = float(u_yr[0] * gs[0] + u_yr[1] * gs[1] - u_yr[2])
        a_scaled = e_scale * a_unscaled
        is_ice_firn = float(r["surface_density_Mg_m3"]) >= 0.8
        rows.append(
            {
                "pole_id": str(r["pole_id"]),
                "x_m": x_m,
                "y_m": y_m,
                "observed_smb": float(r["observed_smb_m_ice_eq_yr"]),
                "obs_sigma": float(r["observed_smb_one_sigma_m_yr"]),
                "modeled_smb_Eopt": a_scaled,
                "modeled_smb_E1": a_unscaled,
                "is_ice_firn": is_ice_firn,
            }
        )
    df = pd.DataFrame(rows)
    ice = df[df["is_ice_firn"]]
    if len(ice) >= 3:
        obs = ice["observed_smb"].to_numpy(float)
        mod = ice["modeled_smb_Eopt"].to_numpy(float)
        diff = obs - mod
        pr, pp = scipy.stats.pearsonr(obs, mod)
        sr, sp = scipy.stats.spearmanr(obs, mod)
        stats = {
            "n_all": int(len(df)),
            "n_ice_firn": int(len(ice)),
            "bias_obs_minus_mod_m_yr": float(np.mean(diff)),
            "mae_m_yr": float(np.mean(np.abs(diff))),
            "rmse_m_yr": float(math.sqrt(float(np.mean(diff ** 2)))),
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_rho": float(sr),
            "spearman_p": float(sp),
        }
    else:
        stats = {"n_all": int(len(df)), "n_ice_firn": int(len(ice))}
    return df, stats


def sample_transect(u_fine, csv_path: Path, slice_end: int | None, e_scale: float):
    df = pd.read_csv(csv_path)
    xs_raw = df["E"].to_numpy(float)[:slice_end]
    ys_raw = df["N"].to_numpy(float)[:slice_end]
    up_x, up_y = [], []
    for n in range(len(xs_raw) - 1):
        dist = math.hypot(xs_raw[n + 1] - xs_raw[n], ys_raw[n + 1] - ys_raw[n])
        if dist < 1e-6:
            continue
        n_pts = max(int(dist / 80.0) + 1, 2)
        seg_x = np.linspace(xs_raw[n], xs_raw[n + 1], n_pts)
        seg_y = np.linspace(ys_raw[n], ys_raw[n + 1], n_pts)
        if up_x:
            seg_x = seg_x[1:]
            seg_y = seg_y[1:]
        up_x.append(seg_x)
        up_y.append(seg_y)
    ux = np.concatenate(up_x)
    uy = np.concatenate(up_y)
    # Strictly monotonic cumulative along-transect arclength
    dists = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(ux), np.diff(uy)))])
    z_coords = np.arange(1400.0, 2120.0, 2.0)

    # Batch evaluate entire (n_x * n_z) slice in one fast pass
    XX, ZZ = np.meshgrid(ux, z_coords, indexing="ij")
    YY, _ = np.meshgrid(uy, z_coords, indexing="ij")
    DD, _ = np.meshgrid(dists, z_coords, indexing="ij")
    pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    vels = eval_points_safe(u_fine, pts, dim=3, chunk=4000) * SPY * 1000.0 * e_scale
    U = vels[:, 0].reshape(XX.shape)
    V = vels[:, 1].reshape(XX.shape)
    W = vels[:, 2].reshape(XX.shape)
    return DD.T, ZZ.T, U.T, V.T, W.T, ux, uy


# ===========================================================================
# FIGURE GENERATORS (matching manuscript Figs 1, 2, 3 & SI Figs S1-S3, S5-S10)
# ===========================================================================

def plot_fig1(out_dir: Path, vel10, fine_dem, boreholes_df, gps_df, lima_path: Path, good_dem_path: Path, repo_dir: Path, title_tag: str):
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    bx, by = tf.transform(boreholes_df["E"].to_numpy(float), boreholes_df["S"].to_numpy(float))

    with rasterio.open(lima_path) as src1:
        lima_img = src1.read()
        lima_tf = src1.transform
    with rasterio.open(good_dem_path) as src_g:
        good_filter = src_g.read(2)
        good_tf = src_g.transform

    fl1 = pd.read_csv(repo_dir / "flowlines/old_ice_flowlines/ALHIC1902_R_surface_nunatak_test_100m_test_case_4.csv")
    fl2 = pd.read_csv(repo_dir / "flowlines/manual_flowlines/csv/ALHIC1901_surface_rema_100m.csv")
    fl3 = pd.read_csv(repo_dir / "flowlines/manual_flowlines/csv/ALHIC2401_bed_merged_100m.csv")

    ok = np.isfinite(vel10["speed_mm_yr"])
    vx = vel10["x"][ok]
    vy = vel10["y"][ok]
    u_myr = vel10["uvw_mm_yr"][ok, 0] / 1000.0
    v_myr = vel10["uvw_mm_yr"][ok, 1] / 1000.0
    w_myr = vel10["uvw_mm_yr"][ok, 2] / 1000.0

    norm_w = mcolors.Normalize(vmin=-0.001, vmax=0.001)
    cmap_w = plt.cm.seismic
    colors_w = cmap_w(norm_w(w_myr))

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(13.5, 5.8), layout="constrained")
    for a in (ax, axz):
        rasterio_show(lima_img, ax=a, transform=lima_tf)

    Q = ax.quiver(vx, vy, u_myr, v_myr, angles="xy", scale_units="xy", scale=None, color=colors_w)
    ax.quiver(gps_df["x"], gps_df["y"], gps_df["u"], gps_df["v"], angles="xy", scale_units="xy", scale=None, color="black")
    rasterio_show(good_filter, ax=ax, transform=good_tf, cmap="gray", alpha=0.35)

    # Bed contours
    bed_g = fine_dem["bed"]
    lons = fine_dem["xs"]
    lats = fine_dem["ys"]
    levels = [1500, 1600, 1700, 1800, 1900, 2000, 2100]
    for a in (ax, axz):
        cs = a.contour(lons, lats, bed_g, colors="k", levels=levels, linewidths=0.5)
        a.clabel(cs, fontsize=7, fmt="%d")

    # Transects
    ax.plot(fl1["E"], fl1["N"], ls="--", c="black", lw=1.4, label="Slow Transect (Fig. 2)")
    ax.plot(fl2["E"][25:46], fl2["N"][25:46], ls="dotted", c="black", lw=1.6, label="Fast Transect (Fig. S5)")
    ax.plot(fl3["E"][:19], fl3["N"][:19], ls="dashdot", c="black", lw=1.4, label="CDS Transect (Fig. S6)")
    axz.plot(fl1["E"], fl1["N"], ls="--", c="black", lw=1.4)

    # Boreholes
    styles = [
        (5, "d", "cyan"),
        (1, "*", "yellow"),
        (3, "s", "green"),
        (4, "^", "pink"),
        (7, "X", "blue"),
        (0, "v", "red"),
    ]
    for idx_b, mk, fc in styles:
        lbl = boreholes_df.iloc[idx_b]["Core"]
        if idx_b == 4:
            lbl = "ALHIC2301/ALHIC2401"
        ax.scatter(bx[idx_b], by[idx_b], marker=mk, ec="black", fc=fc, s=55, zorder=8, label=lbl)
    axz.scatter(bx[5], by[5], marker="d", ec="black", fc="cyan", s=110, zorder=8)
    axz.scatter(bx[1], by[1], marker="*", ec="black", fc="yellow", s=140, zorder=8)

    # Zoomed quiver (exaggerated to emphasize flow convergence)
    axz.quiver(
        vx, vy, u_myr, v_myr,
        angles="xy", scale_units="xy", scale=2e-5,
        width=0.005, color=colors_w, headwidth=5, headlength=5, minshaft=2,
    )

    ax.set_xlim(507500, 512000)
    ax.set_ylim(-1355500, -1351000)
    ax.set_aspect("equal")
    axz.set_xlim(508650, 509800)
    axz.set_ylim(-1353900, -1352800)
    axz.set_aspect("equal")
    ax.add_patch(Rectangle((508650, -1353900), 1150, 1100, fill=False, ec="black", lw=1.5))
    ax.add_patch(Rectangle((509000, -1355100), 2000, 2600, fill=False, ec="purple", ls="--", lw=1.3))

    ax.set_xlabel("Easting (m, EPSG:3031)")
    ax.set_ylabel("Northing (m, EPSG:3031)")
    axz.set_xlabel("Easting (m, EPSG:3031)")
    ax.set_title(f"A. Modeled Flow 10 m Above Bed ({title_tag})")
    axz.set_title("B. Zoomed Stagnation Zone Upstream of Nunatak")
    ax.legend(loc="lower right", fontsize=7, ncol=2, framealpha=0.88)

    sm = cm.ScalarMappable(cmap=cmap_w, norm=norm_w)
    cbar = fig.colorbar(sm, ax=axz, shrink=0.85, extend="both")
    cbar.set_label(r"Vertical velocity $w$ (m yr$^{-1}$)")
    fig.savefig(out_dir / "C2_Fig01.png", dpi=260, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_transect_fig(out_base: Path, DD, ZZ, U, V, W, xlim, ylim, vms, end_labels, title_str, shift_x: float = 0.0):
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.2), sharex=True, layout="constrained")
    D_plot = DD - shift_x
    specs = [
        (U, vms[0], r"u (mm yr$^{-1}$)", "A. East-west velocity"),
        (V, vms[1], r"v (mm yr$^{-1}$)", "B. North-south velocity"),
        (W, vms[2], r"w (mm yr$^{-1}$)", "C. Vertical velocity"),
    ]
    for ax, (arr, vm, clbl, ptitle) in zip(axes, specs):
        im = ax.pcolormesh(D_plot, ZZ, arr, cmap="coolwarm", vmin=-vm, vmax=vm, shading="auto")
        fig.colorbar(im, ax=ax, label=clbl, extend="both")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_ylabel("Elevation (m)")
        ax.text(0.01, 0.04, ptitle, transform=ax.transAxes, fontsize=9.5, weight="bold")
        ax.text(0.0, 1.02, end_labels[0], transform=ax.transAxes, fontsize=11)
        ax.text(0.98, 1.02, end_labels[1], transform=ax.transAxes, fontsize=11)
    axes[-1].set_xlabel("Distance (m)")
    fig.suptitle(title_str, fontsize=11)
    fig.savefig(out_base.with_suffix(".png"), dpi=240)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_fig3_and_s3_and_s9(out_dir: Path, core_df, grid_df, fits, window_minima, d38_means, d18_means, title_tag: str):
    # --- Main Fig 3 (10 m above bed, d38Ar/36Ar) ---
    h = 10.0
    fit = fits[h]
    tgt = window_minima[h]
    cal_spd = np.logspace(fit["log_speed_min"], fit["log_speed_max"], 200)
    cal_p = predict_age_ols(fit, cal_spd)
    ext_spd = np.logspace(math.log10(tgt["speed_mm_yr"]), fit["log_speed_min"], 150)
    ext_p = predict_age_ols(fit, ext_spd)

    fig, (ax_r, ax_m) = plt.subplots(1, 2, figsize=(12.2, 5.8), layout="constrained")
    ax_r.fill_betweenx(cal_spd, cal_p["pi_low"], cal_p["pi_high"], color="#56B4E9", alpha=0.22, label="95% prediction interval")
    ax_r.fill_betweenx(cal_spd, cal_p["ci_low"], cal_p["ci_high"], color="#0072B2", alpha=0.28, label="95% mean CI")
    ax_r.plot(cal_p["mean"], cal_spd, color="#0072B2", lw=1.8, label=rf"OLS fit ($R^2$={fit['r2']:.2f}, p={fit['p_two_sided']:.3f})")
    ax_r.plot(ext_p["mean"], ext_spd, color="#0072B2", lw=1.5, ls="--", label="Extrapolation in Fig. 3b window")
    ax_r.fill_betweenx(ext_spd, ext_p["pi_low"], ext_p["pi_high"], color="#56B4E9", alpha=0.22)

    norm_iso = mcolors.Normalize(0.09, 0.19)
    cmap_iso = plt.get_cmap("Reds_r")
    sub_c = core_df[core_df["height_m"] == h]
    for row in sub_c.itertuples(index=False):
        mk = "s" if row.core_id in DISPLAY_ONLY_CORES else "o"
        ax_r.scatter(
            row.max_age_ma, row.speed_mm_yr, s=55, marker=mk,
            facecolor=cmap_iso(norm_iso(d38_means[row.core_id])), edgecolor="black", lw=0.7, zorder=6,
        )
        ax_r.annotate(row.core_id, (row.max_age_ma, row.speed_mm_yr), xytext=(6, 4), textcoords="offset points", fontsize=8)

    ax_r.errorbar(
        tgt["fitted_age_ma"], tgt["speed_mm_yr"],
        xerr=[[tgt["fitted_age_ma"] - tgt["pi_low_ma"]], [tgt["pi_high_ma"] - tgt["fitted_age_ma"]]],
        fmt="o", ms=7, mfc="#F0E442", mec="black", mew=0.8, ecolor="black", elinewidth=1.2, capsize=4, zorder=8,
        label=f"Window min ({tgt['fitted_age_ma']:.1f} Ma [{tgt['pi_low_ma']:.1f}, {tgt['pi_high_ma']:.1f}])",
    )
    ax_r.set_yscale("log")
    ax_r.set_xlim(0.0, max(16.0, tgt["pi_high_ma"] + 0.8))
    ax_r.set_ylim(max(1e-4, tgt["speed_mm_yr"] / 2.5), 120.0)
    ax_r.set_xlabel("Maximum dated age (Ma)")
    ax_r.set_ylabel(r"Modeled speed at bed + 10 m (mm yr$^{-1}$)")
    ax_r.set_title(f"A. Age vs Speed at 10 m ({title_tag})")
    ax_r.grid(which="major", color="0.55", lw=0.6)
    ax_r.grid(which="minor", color="0.85", lw=0.4)
    ax_r.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    sm_iso = plt.cm.ScalarMappable(norm=norm_iso, cmap=cmap_iso)
    fig.colorbar(sm_iso, ax=ax_r, fraction=0.046, shrink=0.6, pad=0.03, label=r"mean $\delta^{38}$Ar/$^{36}$Ar (<1.5 Ma, ‰)")

    # Panel B: 10 m speed map in Fig 3b window
    wxmin, wxmax, wymin, wymax = RELEASED_NUNATAK_WINDOW
    sub_g = grid_df[(grid_df["height_m"] == h) & grid_df["x_m"].between(wxmin, wxmax) & grid_df["y_m"].between(wymin, wymax)]
    piv_s = sub_g.pivot(index="y_m", columns="x_m", values="speed_mm_yr")
    piv_b = sub_g.pivot(index="y_m", columns="x_m", values="bed_elevation_m")
    s_arr = piv_s.to_numpy(float)
    b_arr = piv_b.to_numpy(float)
    cmap_g = plt.get_cmap("gray").copy()
    cmap_g.set_bad("lightgrey")
    vmin_s = float(np.nanmin(s_arr))
    vmax_s = float(np.nanmax(s_arr))
    im_m = ax_m.pcolormesh(piv_s.columns, piv_s.index, s_arr, cmap=cmap_g, norm=mcolors.LogNorm(vmin=vmin_s, vmax=vmax_s), shading="nearest")
    cs = ax_m.contour(piv_b.columns, piv_b.index, b_arr, levels=[1500, 1600, 1700, 1800, 1900, 2000, 2100], colors="black", linewidths=0.5)
    ax_m.clabel(cs, fontsize=7, inline=True)
    for row in sub_c.itertuples(index=False):
        if wxmin <= row.x_m <= wxmax and wymin <= row.y_m <= wymax:
            ax_m.scatter(row.x_m, row.y_m, s=30, facecolor="white", edgecolor="black", lw=0.7, zorder=6)
    ax_m.scatter(tgt["x_m"], tgt["y_m"], s=75, facecolor="#F0E442", edgecolor="black", lw=0.8, zorder=7)
    ax_m.set_xlim(wxmin, wxmax)
    ax_m.set_ylim(wymin, wymax)
    ax_m.set_aspect("equal")
    ax_m.set_xlabel("Easting (m, EPSG:3031)")
    ax_m.set_ylabel("Northing (m, EPSG:3031)")
    ax_m.set_title("B. Modeled Speed 10 m Above Bed")
    fig.colorbar(im_m, ax=ax_m, fraction=0.046, shrink=0.82, pad=0.03, label=r"Speed 10 m above bed (mm yr$^{-1}$)")
    fig.savefig(out_dir / "C2_Fig03.png", dpi=240)
    plt.close(fig)

    # --- Fig S3 (d18O version of Fig 3a) ---
    fig, ax = plt.subplots(figsize=(6.8, 5.4), layout="constrained")
    ax.fill_betweenx(cal_spd, cal_p["pi_low"], cal_p["pi_high"], color="#56B4E9", alpha=0.22)
    ax.fill_betweenx(cal_spd, cal_p["ci_low"], cal_p["ci_high"], color="#0072B2", alpha=0.28)
    ax.plot(cal_p["mean"], cal_spd, color="#0072B2", lw=1.8, label=rf"5-core OLS fit ($R^2$={fit['r2']:.2f})")
    norm18 = mcolors.Normalize(-40.0, -36.5)
    cmap18 = plt.get_cmap("Blues_r")
    for row in sub_c.itertuples(index=False):
        mk = "s" if row.core_id in DISPLAY_ONLY_CORES else "o"
        ax.scatter(row.max_age_ma, row.speed_mm_yr, s=65, marker=mk, facecolor=cmap18(norm18(d18_means[row.core_id])), edgecolor="black", lw=0.7, zorder=6)
        ax.annotate(row.core_id, (row.max_age_ma, row.speed_mm_yr), xytext=(6, 4), textcoords="offset points", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xlabel("Maximum dated age (Ma)")
    ax.set_ylabel(r"Modeled speed at bed + 10 m (mm yr$^{-1}$)")
    ax.set_title(f"Supplementary Fig. 3: Age vs Speed colored by $\\delta^{{18}}$O\n({title_tag})", fontsize=10)
    ax.grid(which="both", alpha=0.4)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm18, cmap=cmap18), ax=ax, label=r"mean $\delta^{18}$O (<1.5 Ma, ‰)")
    fig.savefig(out_dir / "C2_FigS03.png", dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    # --- Fig S9 (4-height sensitivity: 5, 10, 20, 50 m) ---
    fig, axes = plt.subplots(4, 2, figsize=(11.2, 13.8), layout="constrained")
    mx0, mx1, my0, my1 = MANUSCRIPT_WINDOW
    all_v = grid_df[grid_df["valid_domain"] & grid_df["x_m"].between(mx0, mx1) & grid_df["y_m"].between(my0, my1)]["speed_mm_yr"]
    norm_m = mcolors.LogNorm(vmin=float(all_v.min()), vmax=float(all_v.max()))
    cmap_v = plt.get_cmap("viridis").copy()
    cmap_v.set_bad("lightgrey")
    shared_xmin = min(window_minima[hh]["speed_mm_yr"] for hh in HEIGHTS_M) / 1.8

    for r_i, hh in enumerate(HEIGHTS_M):
        axr, axm = axes[r_i]
        ff = fits[hh]
        tt = window_minima[hh]
        cs_p = np.logspace(ff["log_speed_min"], ff["log_speed_max"], 200)
        cp = predict_age_ols(ff, cs_p)
        es_p = np.logspace(math.log10(tt["speed_mm_yr"]), ff["log_speed_min"], 150)
        ep = predict_age_ols(ff, es_p)

        axr.fill_between(cs_p, cp["pi_low"], cp["pi_high"], color="#56B4E9", alpha=0.22)
        axr.fill_between(cs_p, cp["ci_low"], cp["ci_high"], color="#0072B2", alpha=0.27)
        axr.plot(cs_p, cp["mean"], color="#0072B2", lw=1.8)
        axr.plot(es_p, ep["mean"], color="#0072B2", lw=1.4, ls="--")
        axr.fill_between(es_p, ep["pi_low"], ep["pi_high"], color="#56B4E9", alpha=0.22)
        sc = core_df[core_df["height_m"] == hh]
        for row in sc.itertuples(index=False):
            if row.core_id in DISPLAY_ONLY_CORES:
                axr.scatter(row.speed_mm_yr, row.max_age_ma, s=46, marker="x", color="0.25", lw=1.3, zorder=5)
            else:
                axr.scatter(row.speed_mm_yr, row.max_age_ma, s=42, marker="o", facecolor=FLOW_COLORS[FLOW_GROUP[row.core_id]], edgecolor="black", lw=0.6, zorder=5)
            if r_i == 0:
                axr.annotate(row.core_id, (row.speed_mm_yr, row.max_age_ma), xytext=(5, 4), textcoords="offset points", fontsize=7)
        axr.errorbar(
            tt["speed_mm_yr"], tt["fitted_age_ma"],
            yerr=[[tt["fitted_age_ma"] - tt["pi_low_ma"]], [tt["pi_high_ma"] - tt["fitted_age_ma"]]],
            fmt="D", ms=5.5, mfc="#F0E442", mec="black", ecolor="black", capsize=3, zorder=8,
        )
        axr.set_xscale("log")
        axr.set_xlim(shared_xmin, 250.0)
        axr.set_ylim(-1.0, max(20.0, tt["pi_high_ma"] + 1.5))
        axr.set_xlabel(r"Modeled speed (mm yr$^{-1}$)")
        axr.set_ylabel("Maximum dated age (Ma)")
        axr.grid(which="both", color="0.88", lw=0.5)
        axr.set_title(f"{hh:g} m above bed: min-speed fit {tt['fitted_age_ma']:.1f} Ma [{tt['pi_low_ma']:.1f}, {tt['pi_high_ma']:.1f}]; $R^2$={ff['r2']:.2f}", fontsize=8.5)

        sg = grid_df[(grid_df["height_m"] == hh) & grid_df["x_m"].between(mx0, mx1) & grid_df["y_m"].between(my0, my1)]
        ps = sg.pivot(index="y_m", columns="x_m", values="speed_mm_yr")
        pb = sg.pivot(index="y_m", columns="x_m", values="bed_elevation_m")
        im_v = axm.pcolormesh(ps.columns, ps.index, ps.to_numpy(float), cmap=cmap_v, norm=norm_m, shading="nearest")
        axm.contour(pb.columns, pb.index, pb.to_numpy(float), levels=[1500, 1600, 1700, 1800, 1900, 2000, 2100], colors="black", linewidths=0.45)
        for row in sc.itertuples(index=False):
            axm.scatter(row.x_m, row.y_m, s=26, facecolor="white", edgecolor="black", lw=0.7, zorder=5)
        axm.scatter(tt["x_m"], tt["y_m"], marker="x", s=65, color="black", lw=2.4, zorder=6)
        axm.scatter(tt["x_m"], tt["y_m"], marker="x", s=45, color="white", lw=1.2, zorder=7)
        axm.add_patch(Rectangle((wxmin, wymin), wxmax - wxmin, wymax - wymin, fill=False, ec="black", ls="--", lw=0.9))
        axm.set_xlim(mx0, mx1)
        axm.set_ylim(my0, my1)
        axm.set_aspect("equal")
        axm.set_title(f"{hh:g} m speed map", fontsize=8.5)

    fig.colorbar(im_v, ax=axes[:, 1], shrink=0.4, label=r"Modeled speed (mm yr$^{-1}$)")
    fig.suptitle(f"Supplementary Fig. 9: Height Sensitivity ({title_tag})", fontsize=11)
    fig.savefig(out_dir / "C2_FigS09.png", dpi=220)
    fig.savefig(out_dir / "C2_FigS09.pdf")
    plt.close(fig)


def plot_fig_s1(out_dir: Path, gnss_cal: dict, title_tag: str):
    df = gnss_cal["stakes_df"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    ax.errorbar(df["name"], df["h_resid_opt"], yerr=df["h_unc"], fmt="o", ms=5, capsize=2, label="Horizontal speed residual (model − obs)")
    ax.errorbar(df["name"], df["w_resid_opt"], yerr=df["v_unc"], fmt="o", ms=5, capsize=2, label="Vertical velocity residual (model − obs)")
    ax.axhline(0.0, color="k", lw=0.7, ls="--")
    ax.set_ylabel(r"Residual (m yr$^{-1}$)")
    ax.set_xlabel("GNSS stake")
    ax.set_ylim(-0.16, 0.30)
    ax.grid(alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"Supplementary Fig. 1: GNSS Residuals ({title_tag})\n"
        f"E = {gnss_cal['E_opt']:.2f} [{gnss_cal['E_ci_low']:.2f}, {gnss_cal['E_ci_high']:.2f}], "
        f"RMSE = {gnss_cal['horiz_rmse_m_yr']:.4f} m yr$^{{-1}}$",
        fontsize=10,
    )

    axins = ax.inset_axes([0.10, 0.58, 0.36, 0.34])
    axins.plot(gnss_cal["e_grid"], gnss_cal["rmse_grid"], color="navy")
    axins.scatter([gnss_cal["E_opt"]], [gnss_cal["horiz_rmse_m_yr"]], color="red", zorder=5)
    axins.text(gnss_cal["E_opt"], gnss_cal["horiz_rmse_m_yr"] + 0.005, f"E={gnss_cal['E_opt']:.2f}", fontsize=8, ha="center")
    axins.set_xlabel("Enhancement factor E", fontsize=8)
    axins.set_ylabel("RMSE (m/yr)", fontsize=8)
    axins.tick_params(labelsize=7.5)
    axins.grid(alpha=0.35)

    fig.savefig(out_dir / "C2_FigS01.png", dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def evaluate_and_plot_fig_s2(out_dir: Path, T_steady, T_clim, T_no_adv, base_bed_f, base_surf_f, boreholes_df, repo_dir: Path, title_tag: str):
    avg_temps = pd.read_pickle(repo_dir / "Borehole_Temps/avg_temps.pkl")
    residual_temps = pd.read_pickle(repo_dir / "Borehole_Temps/residual_temps.pkl")
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    bh = boreholes_df.copy()
    bh["x_m"], bh["y_m"] = tf.transform(bh["E"].to_numpy(float), bh["S"].to_numpy(float))
    bh_idx = bh.set_index("Core")

    key_order = ("ALHIC2301", "ALHIC2401", "ALHIC1902", "ALHIC2302", "ALHIC1901", "ALHIC2201")
    offsets = {}
    adv_maes = {}
    noadv_maes = {}
    noadv_ses = {}
    profiles = {}

    for name in key_order:
        x_m = float(bh_idx.loc[name, "x_m"])
        y_m = float(bh_idx.loc[name, "y_m"])
        zb = float(eval_scalar_2d(base_bed_f, np.array([[x_m, y_m]]))[0])
        zs = float(eval_scalar_2d(base_surf_f, np.array([[x_m, y_m]]))[0])
        z_prof = np.arange(zb + 1.0, zs - 0.5, 5.0)
        pts = np.column_stack([np.full_like(z_prof, x_m), np.full_like(z_prof, y_m), z_prof])
        t_st = eval_points_safe(T_steady, pts, dim=1)
        t_cl = eval_points_safe(T_clim, pts, dim=1)
        t_na = eval_points_safe(T_no_adv, pts, dim=1)
        ok = np.isfinite(t_st) & np.isfinite(t_cl) & np.isfinite(t_na)
        z_prof, t_st, t_cl, t_na = z_prof[ok], t_st[ok], t_cl[ok], t_na[ok]
        z_off = z_prof[-1] - CORE_SURFACE_Z[name]
        z_aligned = z_prof - z_off

        off_list = []
        adv_res_list = []
        na_res_list = []
        for k_obs in avg_temps:
            if not k_obs.startswith(name):
                continue
            obs_z = avg_temps[k_obs][0].values
            obs_t = avg_temps[k_obs][2].values
            # Compare below shallow seasonal signal (exclude top 5 points = ~25 m, matching fig2_v3.ipynb!)
            z_cmp = z_aligned[:-5]
            in_rng = (z_cmp >= obs_z.min()) & (z_cmp <= obs_z.max())
            if in_rng.sum() < 5:
                continue
            f_obs = scipy.interpolate.interp1d(obs_z, obs_t)
            obs_interp = f_obs(z_cmp[in_rng])
            off_val = float(np.mean(t_cl[:-5][in_rng] - obs_interp))
            off_list.append(off_val)
            adv_res_list.append(np.abs((t_cl[:-5][in_rng] - off_val) - obs_interp))
            # No-advection MAE (detrended residual vs observed linear residual OR shifted profile)
            na_off = float(np.mean(t_na[:-5][in_rng] - obs_interp))
            na_res_list.append(np.abs((t_na[:-5][in_rng] - na_off) - obs_interp))

        offsets[name] = float(np.mean(off_list)) if off_list else float("nan")
        adv_all = np.concatenate(adv_res_list) if adv_res_list else np.array([np.nan])
        na_all = np.concatenate(na_res_list) if na_res_list else np.array([np.nan])
        adv_maes[name] = float(np.mean(adv_all))
        noadv_maes[name] = float(np.mean(na_all))
        noadv_ses[name] = float(np.std(na_all) / math.sqrt(len(na_all)))
        profiles[name] = (z_aligned, t_st, t_cl, t_na)

    fig = plt.figure(figsize=(20, 7.6), layout="constrained")
    gs = GridSpec(2, 6, figure=fig, height_ratios=[5, 2.8])
    plot_labels = string.ascii_uppercase[:6]
    inset_labels = string.ascii_uppercase[6:12]

    for idx_p, name in enumerate(key_order):
        ax = fig.add_subplot(gs[0, idx_p])
        z_al, t_st, t_cl, _ = profiles[name]
        off = offsets[name]
        ax.plot(t_cl - off, z_al, c="black", label="Model (with climate)")
        ax.plot(t_st - off, z_al, c="black", ls="--", label="Model (steady state)")

        axins = inset_axes(ax, width="45%", height="58%", loc="lower right", borderpad=0.9)
        axins.plot(scipy.signal.detrend(t_cl[:-5]), z_al[:-5], c="black", lw=1.1)
        axins.plot(scipy.signal.detrend(t_st[:-5]), z_al[:-5], c="black", ls="--", lw=1.0)
        axins.set_xlim(-0.25, 0.30)
        axins.tick_params(labelsize=6.5)

        min_t = float(np.min(t_st - off))
        for k_obs in avg_temps:
            if k_obs.startswith(name):
                clr = {"22": "steelblue", "23": "indianred", "24": "forestgreen"}.get(k_obs[-2:], "purple")
                ax.plot(avg_temps[k_obs][2], avg_temps[k_obs][0], c=clr, label=k_obs[-2:] + " Season")
                axins.plot(residual_temps[k_obs][2], residual_temps[k_obs][0], c=clr, lw=1.0)
                min_t = min(min_t, float(avg_temps[k_obs][2].min()))
        ax.set_xlim(min_t - 2.0, -10.0)
        ax.set_title(name)
        ax.text(0.02, 0.94, plot_labels[idx_p], transform=ax.transAxes, weight="bold", fontsize=11)
        axins.text(0.05, 0.90, inset_labels[idx_p], transform=axins.transAxes, weight="bold", fontsize=9)
        ax.text(0.02, 0.02, f"offset: {off:.2f} °C", transform=ax.transAxes, fontsize=8)
        if idx_p == 0:
            ax.set_ylabel("Elevation (m)")
            ax.set_xlabel("Temperature (°C)")
            ax.legend(loc="upper right", fontsize=6.5)

    ax_m = fig.add_subplot(gs[1, :])
    groups = [
        (["ALHIC2301", "ALHIC2401"], "red", "Cul-de-sac"),
        (["ALHIC1902", "ALHIC2302"], "blue", "Nunatak (stagnation)"),
        (["ALHIC1901", "ALHIC2201"], "green", "Submerged Ridge"),
    ]
    for keys, clr, lbl in groups:
        ax_m.errorbar(
            keys, [noadv_maes[k] for k in keys], yerr=[noadv_ses[k] for k in keys],
            fmt="o", capsize=3, color=clr, label=lbl, ms=6,
        )
    ax_m.text(0.01, 0.88, "M", transform=ax_m.transAxes, weight="bold", fontsize=11)
    ax_m.set_xlabel("Borehole")
    ax_m.set_ylabel("No-advection MAE (K)")
    ax_m.grid(alpha=0.4)
    ax_m.legend(loc="upper right", fontsize=8.5)
    fig.suptitle(f"Supplementary Fig. 2: Borehole Temperatures ({title_tag})", fontsize=12)
    fig.savefig(out_dir / "C2_FigS02.png", dpi=220)
    plt.close(fig)

    return {
        "offsets_C": offsets,
        "mean_adv_mae_K": float(np.mean(list(adv_maes.values()))),
        "adv_mae_by_borehole_K": adv_maes,
        "noadv_mae_by_borehole_K": noadv_maes,
        "nunatak_lt_cds_and_ridge": bool(
            max(noadv_maes["ALHIC1902"], noadv_maes["ALHIC2302"])
            < min(noadv_maes["ALHIC2301"], noadv_maes["ALHIC2401"], noadv_maes["ALHIC1901"], noadv_maes["ALHIC2201"])
        ),
    }


def plot_fig_s7(out_dir: Path, run_diag: dict, title_tag: str):
    t_abs = run_diag["fine"].get("theta_abs", [])
    t_rel = run_diag["fine"].get("theta_rel", [])
    if not t_abs:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    ln1 = ax.plot(np.arange(len(t_abs)), t_abs, c="indianred", lw=1.8, label=r"Normalized viscosity update $\theta_{\mathrm{abs},t}$")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\theta_{\mathrm{abs},t}$", color="indianred")
    ax.set_xlabel("Iteration Number t")
    ax.grid(alpha=0.35)
    ax2 = ax.twinx()
    ln2 = ax2.plot(np.arange(1, 1 + len(t_rel)), t_rel, c="steelblue", lw=1.8, label=r"Fractional change $\theta_{\mathrm{rel},t}$")
    ax2.set_ylabel(r"$\theta_{\mathrm{rel},t}$", color="steelblue")
    lns = ln1 + ln2
    ax.legend(lns, [l.get_label() for l in lns], loc="upper right", fontsize=8)
    ax.set_title(f"Supplementary Fig. 7: Fine-Mesh Viscosity Convergence ({title_tag})")
    fig.savefig(out_dir / "C2_FigS07.png", dpi=240)
    fig.savefig(out_dir / "C2_FigS07.pdf")
    plt.close(fig)


def sample_coarse_surface_grid(u_coarse, base_bed_c, base_surf_c, is_basin: bool, inp: Path, e_scale: float, grid_step_m: float = 200.0):
    """Evaluate calibrated coarse surface velocity, surface slope, and ice thickness on a regular grid."""
    from shapely import contains_xy
    from shapely.geometry import box as shapely_box

    if is_basin:
        gj = json.loads((inp / "allan_hills_rema_basin.geojson").read_text(encoding="utf-8"))
        poly = shape(gj["features"][0]["geometry"])
        xmin, ymin, xmax, ymax = poly.bounds
    else:
        xmin, xmax = 502369.57, 531866.83
        ymin, ymax = -1362201.48, -1342728.25
        poly = shapely_box(xmin, ymin, xmax, ymax)

    gx = np.arange(math.floor(xmin / grid_step_m) * grid_step_m, math.ceil(xmax / grid_step_m) * grid_step_m + 1.0, grid_step_m)
    gy = np.arange(math.floor(ymin / grid_step_m) * grid_step_m, math.ceil(ymax / grid_step_m) * grid_step_m + 1.0, grid_step_m)
    XX, YY = np.meshgrid(gx, gy)

    # Project surface gradient onto CG1 on the coarse 2D mesh for smooth boundary-complete slopes
    V_grad = VectorFunctionSpace(base_surf_c.ufl_domain(), "CG", 1)
    grad_s_fn = fd.project(grad(base_surf_c), V_grad)

    in_poly = contains_xy(poly.buffer(grid_step_m * 0.5), XX, YY)
    idx_r, idx_c = np.where(in_poly)
    pts2 = np.column_stack([XX[idx_r, idx_c], YY[idx_r, idx_c]])

    zs_val = eval_scalar_2d(base_surf_c, pts2)
    zb_val = eval_scalar_2d(base_bed_c, pts2)
    gs_val = eval_points_safe(grad_s_fn, pts2, dim=2, chunk=2500)
    valid2 = np.isfinite(zs_val) & np.isfinite(zb_val) & np.isfinite(gs_val).all(axis=1)

    U = np.full(XX.shape, np.nan, dtype=float)
    V = np.full(XX.shape, np.nan, dtype=float)
    S = np.full(XX.shape, np.nan, dtype=float)
    ZS = np.full(XX.shape, np.nan, dtype=float)
    H = np.full(XX.shape, np.nan, dtype=float)
    SLOPE = np.full(XX.shape, np.nan, dtype=float)

    if valid2.any():
        vr = idx_r[valid2]
        vc = idx_c[valid2]
        zs_ok = zs_val[valid2]
        zb_ok = zb_val[valid2]
        slp_ok = np.hypot(gs_val[valid2, 0], gs_val[valid2, 1])
        # Strictly inside top layer even where thickness hits 1 m rock-outcrop guard
        z_eval = np.maximum(zs_ok - 0.25, 0.5 * (zb_ok + zs_ok))
        pts3 = np.column_stack([pts2[valid2, 0], pts2[valid2, 1], z_eval])
        vel3 = eval_points_safe(u_coarse, pts3, dim=3, chunk=2500) * SPY * e_scale
        ok3 = np.isfinite(vel3).all(axis=1)
        if ok3.any():
            U[vr[ok3], vc[ok3]] = vel3[ok3, 0]
            V[vr[ok3], vc[ok3]] = vel3[ok3, 1]
            S[vr[ok3], vc[ok3]] = np.hypot(vel3[ok3, 0], vel3[ok3, 1])
            ZS[vr[ok3], vc[ok3]] = zs_ok[ok3]
            H[vr[ok3], vc[ok3]] = np.maximum(zs_ok[ok3] - zb_ok[ok3], 0.0)
            SLOPE[vr[ok3], vc[ok3]] = slp_ok[ok3]

    return {"gx": gx, "gy": gy, "U": U, "V": V, "S": S, "ZS": ZS, "H": H, "SLOPE": SLOPE, "is_basin": is_basin}


def plot_fig_s8_domain(
    out_dir: Path,
    u_coarse,
    base_bed_c,
    base_surf_c,
    is_basin: bool,
    inp: Path,
    boreholes_df,
    lima_path: Path,
    e_scale: float,
    title_tag: str,
    grid_save_path: Path | None = None,
):
    """3-panel Supplementary Fig. 8: (A) Ice velocity vectors over LIMA, (B) Surface slope, (C) Ice thickness."""
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    bx, by = tf.transform(boreholes_df["E"].to_numpy(float), boreholes_df["S"].to_numpy(float))

    grid = sample_coarse_surface_grid(u_coarse, base_bed_c, base_surf_c, is_basin, inp, e_scale, grid_step_m=200.0)
    if grid_save_path is not None:
        np.savez_compressed(
            grid_save_path,
            gx=grid["gx"], gy=grid["gy"], U=grid["U"], V=grid["V"], S=grid["S"],
            ZS=grid["ZS"], H=grid["H"], SLOPE=grid["SLOPE"],
            is_basin=np.array(is_basin), e_scale=np.array(e_scale),
        )

    gx, gy, U, V, S, ZS, H, SLOPE = (
        grid["gx"], grid["gy"], grid["U"], grid["V"], grid["S"], grid["ZS"], grid["H"], grid["SLOPE"]
    )
    with rasterio.open(lima_path) as src1:
        lima_img = src1.read()
        lima_tf = src1.transform

    poly = None
    px = py = None
    if is_basin:
        gj = json.loads((inp / "allan_hills_rema_basin.geojson").read_text(encoding="utf-8"))
        poly = shape(gj["features"][0]["geometry"])
        px, py = poly.exterior.xy
        xlim = (494000, 533000)
        ylim = (-1376500, -1326500)
        leg_loc = "lower right"
        qk_xy = (0.16, 0.05)
    else:
        xlim = (497500, 537500)
        ylim = (-1370000, -1335000)
        leg_loc = "upper right"
        qk_xy = (0.16, 0.06)

    fig_h = 6.8 if is_basin else 5.4
    cb_shrink = 0.78 if is_basin else 0.76
    fig, axes = plt.subplots(1, 3, figsize=(17.5, fig_h), sharex=True, sharey=True, layout="constrained")
    halo_k = [pe.withStroke(linewidth=2.6, foreground="black", alpha=0.65)]
    halo_w = [pe.withStroke(linewidth=2.8, foreground="white", alpha=0.9)]

    def draw_domain_overlays(ax, add_legend: bool = False):
        leg_h = []
        if is_basin:
            ax.plot(px, py, color="steelblue", ls="-", lw=2.0, path_effects=halo_w, zorder=6)
            ax.add_patch(
                Rectangle(
                    (502369.57, -1362201.48), 531866.83 - 502369.57, -1342728.25 - (-1362201.48),
                    fill=False, ec="0.25", ls=":", lw=1.3, path_effects=halo_w, zorder=5,
                )
            )
            leg_h.append(Line2D([0], [0], color="steelblue", lw=2.0, label="REMA basin domain"))
            leg_h.append(Line2D([0], [0], color="0.25", ls=":", lw=1.4, label="Original 1600 m rect."))
        else:
            ax.add_patch(
                Rectangle(
                    (502369.57, -1362201.48), 531866.83 - 502369.57, -1342728.25 - (-1362201.48),
                    fill=False, ec="steelblue", ls="-", lw=2.0, path_effects=halo_w, zorder=6,
                )
            )
            leg_h.append(Line2D([0], [0], color="steelblue", lw=2.0, label="Coarse domain (1600 m)"))

        ax.add_patch(
            Rectangle(
                (506902.38, -1356284.47), 512046.01 - 506902.38, -1350629.26 - (-1356284.47),
                fill=False, ec="#D62728", ls="--", lw=1.9, path_effects=halo_w, zorder=7,
            )
        )
        ax.scatter(bx, by, marker="o", s=26, fc="#D62728", ec="white", lw=0.7, zorder=9)
        leg_h.append(Line2D([0], [0], color="#D62728", ls="--", lw=1.9, label="Fine mesh (80 m)"))
        leg_h.append(Line2D([0], [0], marker="o", ls="none", ms=5.5, mfc="#D62728", mec="black", mew=0.5, label="Boreholes"))
        if add_legend:
            ax.legend(handles=leg_h, loc=leg_loc, fontsize=7.8, framealpha=0.92, facecolor="white", edgecolor="0.4")

    # --- Panel A: Surface ice velocity vectors over unobstructed LIMA imagery ---
    ax0 = axes[0]
    rasterio_show(lima_img, ax=ax0, transform=lima_tf)
    # Sample uniform 1600 m vector grid (every 8th cell of 200 m grid) so spacing is clean & even everywhere
    step_q = 8
    XX_q, YY_q = np.meshgrid(gx[::step_q], gy[::step_q])
    U_q = U[::step_q, ::step_q]
    V_q = V[::step_q, ::step_q]
    S_q = S[::step_q, ::step_q]
    ok_q = np.isfinite(U_q) & np.isfinite(V_q)
    p95 = float(np.percentile(S_q[ok_q], 95)) if ok_q.any() else 1.0
    q_scale = max(p95 * 11.5, 0.5)
    q_c = ax0.quiver(
        XX_q[ok_q], YY_q[ok_q], U_q[ok_q], V_q[ok_q],
        color="navy", alpha=0.88, scale=q_scale, width=0.0042,
        headwidth=3.8, headlength=4.5, zorder=5,
    )
    key_val = round(max(p95, 0.1), 1)
    # Crisp white backdrop box for quiverkey so it's readable over LIMA
    qk_box_x = xlim[0] + 0.03 * (xlim[1] - xlim[0])
    qk_box_y = ylim[0] + 0.02 * (ylim[1] - ylim[0])
    ax0.add_patch(
        Rectangle(
            (qk_box_x, qk_box_y), 0.30 * (xlim[1] - xlim[0]), 0.065 * (ylim[1] - ylim[0]),
            fc="white", ec="0.4", lw=0.7, alpha=0.92, zorder=8,
        )
    )
    qk = ax0.quiverkey(q_c, qk_xy[0], qk_xy[1], key_val, f"{key_val:.1f} m yr$^{{-1}}$", labelpos="E", coordinates="axes", fontproperties={"size": 8.2})
    qk.set_zorder(10)
    draw_domain_overlays(ax0, add_legend=True)
    ax0.set_title(f"A. Modeled Surface Velocity (E = {e_scale:.2f})", fontsize=10.5)

    # --- Panel B: Ice surface slope |grad s| ---
    ax1 = axes[1]
    rasterio_show(lima_img, ax=ax1, transform=lima_tf)
    cmap_slp = plt.get_cmap("YlOrRd").copy()
    cmap_slp.set_bad((0.0, 0.0, 0.0, 0.0))
    slp_plot = np.where(np.isfinite(SLOPE), np.maximum(SLOPE, 0.001), np.nan)
    im_s = ax1.pcolormesh(
        gx, gy, slp_plot,
        cmap=cmap_slp,
        norm=mcolors.LogNorm(vmin=0.002, vmax=0.10),
        shading="auto",
        alpha=1.0,
        zorder=2,
    )
    if np.isfinite(ZS).any():
        z_lo = int(math.floor(np.nanmin(ZS) / 100.0) * 100)
        z_hi = int(math.ceil(np.nanmax(ZS) / 100.0) * 100)
        cs = ax1.contour(gx, gy, ZS, levels=np.arange(z_lo, z_hi + 1, 100.0), colors="0.2", linewidths=0.45, alpha=0.55, zorder=3)
        ax1.clabel(cs, fmt="%d m", fontsize=6.5, inline=True, inline_spacing=4)
    draw_domain_overlays(ax1, add_legend=False)
    ax1.set_title("B. Ice Surface Slope |∇s| & Elevation", fontsize=10.5)
    cb_s = fig.colorbar(im_s, ax=ax1, shrink=cb_shrink, pad=0.02, extend="both")
    cb_s.set_label(r"Surface slope $|\nabla_h s|$ (m m$^{-1}$)")
    cb_s.set_ticks([0.002, 0.005, 0.01, 0.02, 0.05, 0.10])
    cb_s.ax.set_yticklabels(["0.002", "0.005", "0.01", "0.02", "0.05", "0.10"])

    # --- Panel C: Ice thickness H = s - b ---
    ax2 = axes[2]
    rasterio_show(lima_img, ax=ax2, transform=lima_tf)
    cmap_h = plt.get_cmap("viridis").copy()
    cmap_h.set_bad((0.0, 0.0, 0.0, 0.0))
    im_h = ax2.pcolormesh(
        gx, gy, H,
        cmap=cmap_h,
        vmin=0.0,
        vmax=1050.0,
        shading="auto",
        alpha=1.0,
        zorder=2,
    )
    draw_domain_overlays(ax2, add_legend=False)
    ax2.set_title("C. Ice Thickness H", fontsize=10.5)
    cb_h = fig.colorbar(im_h, ax=ax2, shrink=cb_shrink, pad=0.02, extend="max")
    cb_h.set_label("Ice thickness H (m)")

    for i_ax, ax in enumerate(axes):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (m, EPSG:3031)")
        if i_ax == 0:
            ax.set_ylabel("Northing (m, EPSG:3031)")

    fig.suptitle(f"Supplementary Fig. 8: Coarse Domain Flow, Surface Slope & Ice Thickness ({title_tag})", fontsize=12)
    fig.savefig(out_dir / "C2_FigS06.png", dpi=240, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_dir / "C2_FigS06.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_fig_s10(out_dir: Path, smb_df: pd.DataFrame, smb_stats: dict, title_tag: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2), layout="constrained")
    ice = smb_df[smb_df["is_ice_firn"]]
    snow = smb_df[~smb_df["is_ice_firn"]]

    for ax, sub_all, title in [
        (ax1, True, f"A. All in-domain Spaulding sites (n={len(smb_df)})"),
        (ax2, False, f"B. Ice & firn sites (n={len(ice)}; r={smb_stats.get('pearson_r', float('nan')):.2f}, RMSE={smb_stats.get('rmse_m_yr', float('nan')):.3f})"),
    ]:
        if sub_all:
            vals = np.concatenate([smb_df["observed_smb"], smb_df["modeled_smb_Eopt"]])
        else:
            vals = np.concatenate([ice["observed_smb"], ice["modeled_smb_Eopt"]])
        lo = min(-0.08, float(np.nanmin(vals)) - 0.02)
        hi = max(0.08, float(np.nanmax(vals)) + 0.02)
        g = np.linspace(lo, hi, 100)
        ax.fill_between(g, g, hi, color="#d9f0d3", alpha=0.35, label=r"$\partial h/\partial t > 0$")
        ax.fill_between(g, lo, g, color="#fddbc7", alpha=0.35, label=r"$\partial h/\partial t < 0$")
        ax.plot([lo, hi], [lo, hi], "k-", lw=1.2, label="1:1 steady geometry")
        if sub_all and len(snow):
            ax.scatter(snow["modeled_smb_Eopt"], snow["observed_smb"], s=36, fc="none", ec="0.35", lw=1.0, label="Excluded snow sites")
        ax.errorbar(
            ice["modeled_smb_Eopt"], ice["observed_smb"], yerr=ice["obs_sigma"],
            fmt="o", ms=5, mfc="#0072B2", mec="black", ecolor="0.3", capsize=2, label="Ice/firn sites",
        )
        if not sub_all:
            for r in ice.itertuples(index=False):
                ax.annotate(r.pole_id, (r.modeled_smb_Eopt, r.observed_smb), xytext=(4, 3), textcoords="offset points", fontsize=7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(r"Modeled steady-geometry SMB $\dot{a}_{\mathrm{modeled}}$ (m ice eq. yr$^{-1}$)")
        ax.set_ylabel(r"Observed SMB $\dot{a}_{\mathrm{observed}}$ (m ice eq. yr$^{-1}$)")
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=0.35)
        ax.legend(loc="upper left", fontsize=7.5)

    fig.suptitle(f"Supplementary Fig. 10: Surface Mass Balance Check ({title_tag})", fontsize=11)
    fig.savefig(out_dir / "C2_FigS10.png", dpi=220)
    fig.savefig(out_dir / "C2_FigS10.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--title-tag", default=None)
    parser.add_argument("--legacy-baseline", action="store_true")
    parser.add_argument("--exp-dir", type=Path, default=None)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--fixed-E", type=float, default=None, help="Optional override of E instead of GNSS calibration")
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if metrics JSON already exists")
    parser.add_argument("--only-fig-s8", action="store_true", help="Fast mode: only regenerate C2_FigS06 (full-basin velocity map) + coarse surface grid NPZ")
    args = parser.parse_args()

    if not args.force and not args.only_fig_s8 and args.metrics_out.is_file():
        print(f"=== Skipping evaluation for {args.experiment}: {args.metrics_out} already exists ===", flush=True)
        return
    if not args.legacy_baseline and not (args.exp_dir / "run_diagnostics.json").is_file():
        print(f"=== Skipping evaluation for {args.experiment}: {args.exp_dir / 'run_diagnostics.json'} not ready yet ===", flush=True)
        return

    configure_matplotlib()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    title_tag = args.title_tag or args.experiment
    inp = args.inputs_dir

    print(f"=== Evaluating experiment: {args.experiment} ===", flush=True)
    bundle = load_models(args, inp)
    u_coarse = bundle["u_coarse"]
    u_fine = bundle["u_fine"]
    T_fine = bundle["T_fine"]
    base_bed_f = bundle["base_bed_f"]
    base_surf_f = bundle["base_surf_f"]
    base_bed_c = bundle["base_bed_c"]
    base_surf_c = bundle["base_surf_c"]

    boreholes_df = pd.read_csv(args.repo_dir / "Borehole_Locations/Borehole_locations_DTS.csv")
    gps_df = pd.read_csv(args.repo_dir / "GPS_velocities/24-25_velocities-good.csv")
    lima_path = inp / "LIMA_virtual_Mosaic_15m.tif"

    if args.only_fig_s8:
        if args.fixed_E is not None:
            e_use = args.fixed_E
        elif args.metrics_out.is_file():
            e_use = float(json.loads(args.metrics_out.read_text(encoding="utf-8"))["E_used"])
        else:
            e_use = calibrate_enhancement_factor(u_fine, base_surf_f, gps_df)["E_opt"]
        plot_fig_s8_domain(
            args.fig_dir, u_coarse, base_bed_c, base_surf_c, bundle["is_basin"],
            inp, boreholes_df, lima_path, e_use, title_tag,
            grid_save_path=args.metrics_out.parent / f"{args.experiment}_coarse_surf_grid.npz",
        )
        print(f"=== Updated C2_FigS06 & coarse surface grid for {args.experiment} (E={e_use:.3f}) ===", flush=True)
        return

    spaulding_df = pd.read_csv(inp / "spaulding_table_s4.csv")
    ages, d38_means, d18_means = load_geochem_and_ages(inp / "argon_and_water_iso_for_JM.xlsx", inp / "age_reconciliation.csv")

    # 1. Calibrate E against 18 fine-mesh GNSS stakes
    gnss_cal = calibrate_enhancement_factor(u_fine, base_surf_f, gps_df)
    e_use = args.fixed_E if args.fixed_E is not None else gnss_cal["E_opt"]
    print(
        f"  GNSS calibration: E_opt={gnss_cal['E_opt']:.3f} [{gnss_cal['E_ci_low']:.3f}, {gnss_cal['E_ci_high']:.3f}], "
        f"RMSE_h={gnss_cal['horiz_rmse_m_yr']:.4f} m/yr, ang_err={gnss_cal['mean_ang_err_deg']:.1f} deg (using E={e_use:.3f})",
        flush=True,
    )

    # 2. Age-speed & exact-height grid evaluation
    core_df, grid_df, fits, window_minima, bh_profiles, vel10, counterflow_diag = evaluate_age_speed_and_grid(
        u_fine, base_bed_f, base_surf_f, bundle["fine_dem"], boreholes_df, ages, e_use
    )
    print(
        f"  10m OLS: R2={fits[10.0]['r2']:.4f}, slope={fits[10.0]['beta']:.3f} Ma/dec, p={fits[10.0]['p_two_sided']:.4f} | "
        f"min_speed={window_minima[10.0]['speed_mm_yr']:.4f} mm/yr -> {window_minima[10.0]['fitted_age_ma']:.2f} Ma "
        f"[{window_minima[10.0]['pi_low_ma']:.1f}, {window_minima[10.0]['pi_high_ma']:.1f}]",
        flush=True,
    )

    # 3. 150-yr transient temperatures (with advection & no advection) for Fig S2
    T_clim, T_no_adv = run_transient_temperatures(bundle["mesh_fine"], u_fine, T_fine, e_use)

    # 4. Spaulding SMB on coarse mesh (Fig S10)
    smb_df, smb_stats = evaluate_spaulding_smb(u_coarse, base_surf_c, spaulding_df, e_use)

    # 5. Side-boundary velocity summary on fine mesh (overall + per wall: 1=West, 2=East, 3=South, 4=North)
    mesh_fine = bundle["mesh_fine"]
    dSv_f = fd.Measure("ds_v", domain=mesh_fine)
    side_area = float(fd.assemble(Constant(1.0) * dSv_f))
    side_u_rms_raw = math.sqrt(float(fd.assemble(inner(u_fine, u_fine) * dSv_f)) / side_area) * SPY * 1000.0
    side_u_rms_scaled = side_u_rms_raw * e_use
    wall_u_rms = {}
    for wid, wname in ((1, "west"), (2, "east"), (3, "south"), (4, "north")):
        warea = float(fd.assemble(Constant(1.0) * dSv_f(wid)))
        wrms_raw = math.sqrt(float(fd.assemble(inner(u_fine, u_fine) * dSv_f(wid))) / max(warea, 1e-12)) * SPY * 1000.0
        wall_u_rms[wname] = {"raw_mm_yr": float(wrms_raw), "scaled_mm_yr": float(wrms_raw * e_use)}

    # 6. Regenerate ALL figures
    lima_path = inp / "LIMA_virtual_Mosaic_15m.tif"
    good_dem_path = inp / "ian_bed_outline_raster_full.tif"

    plot_fig1(args.fig_dir, vel10, bundle["fine_dem"], boreholes_df, gps_df, lima_path, good_dem_path, args.repo_dir, title_tag)

    # Fig 2 (Stagnation transect A-A')
    DD1, ZZ1, U1, V1, W1, _, _ = sample_transect(
        u_fine, args.repo_dir / "flowlines/old_ice_flowlines/ALHIC1902_R_surface_nunatak_test_100m_test_case_4.csv", None, e_use
    )
    plot_transect_fig(
        args.fig_dir / "C2_Fig02", DD1, ZZ1, U1, V1, W1,
        xlim=(0, 2000), ylim=(1600, 2000), vms=(12, 12, 4),
        end_labels=("A", "A'"), title_str=f"Fig. 2: Stagnation Transect A-A' ({title_tag})",
    )

    # Fig S5 (Fast submerged-ridge transect B-B')
    DD2, ZZ2, U2, V2, W2, _, _ = sample_transect(
        u_fine, args.repo_dir / "flowlines/manual_flowlines/csv/ALHIC1901_surface_rema_100m.csv", None, e_use
    )
    plot_transect_fig(
        args.fig_dir / "C2_FigS05", DD2, ZZ2, U2, V2, W2,
        xlim=(0, 1700), ylim=(1400, 2000), vms=(100, 100, 20),
        end_labels=("B", "B'"), title_str=f"Supplementary Fig. 5: Submerged Ridge Transect B-B' ({title_tag})",
        shift_x=2400.0,
    )

    # Fig S6 (Cul-de-sac transect C-C', saved as FigS08.png matching manuscript)
    DD3, ZZ3, U3, V3, W3, _, _ = sample_transect(
        u_fine, args.repo_dir / "flowlines/manual_flowlines/csv/ALHIC2401_bed_merged_100m.csv", 19, e_use
    )
    plot_transect_fig(
        args.fig_dir / "FigS08", DD3, ZZ3, U3, V3, W3,
        xlim=(0, float(DD3[0, -1])), ylim=(1600, 2100), vms=(200, 200, 40),
        end_labels=("C", "C'"), title_str=f"Supplementary Fig. 6: Cul-de-sac Transect C-C' ({title_tag})",
    )

    plot_fig3_and_s3_and_s9(args.fig_dir, core_df, grid_df, fits, window_minima, d38_means, d18_means, title_tag)
    plot_fig_s1(args.fig_dir, gnss_cal, title_tag)
    temp_diag = evaluate_and_plot_fig_s2(args.fig_dir, T_fine, T_clim, T_no_adv, base_bed_f, base_surf_f, boreholes_df, args.repo_dir, title_tag)
    plot_fig_s7(args.fig_dir, bundle["run_diag"], title_tag)
    plot_fig_s8_domain(
        args.fig_dir, u_coarse, base_bed_c, base_surf_c, bundle["is_basin"],
        inp, boreholes_df, lima_path, e_use, title_tag,
        grid_save_path=args.metrics_out.parent / f"{args.experiment}_coarse_surf_grid.npz",
    )
    plot_fig_s10(args.fig_dir, smb_df, smb_stats, title_tag)

    # Save compact CSV exports for downstream cross-experiment comparison
    core_df.to_csv(args.metrics_out.parent / f"{args.experiment}_core_samples.csv", index=False)
    gnss_cal["stakes_df"].to_csv(args.metrics_out.parent / f"{args.experiment}_gnss_stakes.csv", index=False)
    smb_df.to_csv(args.metrics_out.parent / f"{args.experiment}_spaulding_smb.csv", index=False)
    np.savez_compressed(
        args.metrics_out.parent / f"{args.experiment}_speed10m_grid.npz",
        x=vel10["x"], y=vel10["y"], bed=vel10["bed"], surf=vel10["surf"],
        uvw_mm_yr=vel10["uvw_mm_yr"], speed_mm_yr=vel10["speed_mm_yr"],
    )

    # Assemble complete structured metrics JSON
    cds_contrast = {
        "ALHIC2301_over_ALHIC1902": bh_profiles["ALHIC2301"]["U10_mm_yr"] / max(bh_profiles["ALHIC1902"]["U10_mm_yr"], 1e-12),
        "ALHIC2401_over_ALHIC2302": bh_profiles["ALHIC2401"]["U10_mm_yr"] / max(bh_profiles["ALHIC2302"]["U10_mm_yr"], 1e-12),
    }
    # Check monotonic regime ranking: nunatak < ridge < CDS
    u10 = {k: bh_profiles[k]["U10_mm_yr"] for k in bh_profiles}
    regime_ranking_preserved = bool(
        max(u10["ALHIC1902"], u10["ALHIC2302"])
        < min(u10["ALHIC1502"], u10["ALHIC1901"], u10["ALHIC2201"])
        and max(u10["ALHIC1502"], u10["ALHIC1901"], u10["ALHIC2201"])
        < min(u10["ALHIC2301"], u10["ALHIC2401"])
    )

    metrics = {
        "experiment": args.experiment,
        "title_tag": title_tag,
        "E_used": float(e_use),
        "gnss_calibration": {k: v for k, v in gnss_cal.items() if k not in ("stakes_df", "e_grid", "rmse_grid")},
        "fine_side_bc_rms_mm_yr_raw": float(side_u_rms_raw),
        "fine_side_bc_rms_mm_yr_scaled": float(side_u_rms_scaled),
        "fine_side_bc_wall_rms": wall_u_rms,
        "borehole_profiles": bh_profiles,
        "regime_ranking_preserved": regime_ranking_preserved,
        "cds_to_nunatak_u10_contrast": cds_contrast,
        "counterflow_and_stagnation_fig1b": counterflow_diag,
        "ols_fits_by_height_m": {str(int(k)): v for k, v in fits.items()},
        "window_minima_by_height_m": {str(int(k)): v for k, v in window_minima.items()},
        "borehole_temperature_diagnostics": temp_diag,
        "spaulding_smb_coarse_diagnostics": smb_stats,
        "solver_diagnostics": bundle["run_diag"],
    }
    args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"=== Finished {args.experiment}: metrics -> {args.metrics_out}, figures -> {args.fig_dir} ===", flush=True)


if __name__ == "__main__":
    main()
