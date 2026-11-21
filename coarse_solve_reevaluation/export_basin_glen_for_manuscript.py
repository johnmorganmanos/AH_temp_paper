#!/usr/bin/env python3
"""Export compact basin_glen arrays from sermeq Firedrake checkpoints for local
publication-quality manuscript figure rendering (Figs 1, 2, 3, S2, S5, S6, S8, S9)
and exact text velocity diagnostics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer

import evaluate_and_plot_experiment as ev


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, required=True)
    ap.add_argument("--inputs-dir", type=Path, required=True)
    ap.add_argument("--repo-dir", type=Path, required=True)
    ap.add_argument("--metrics-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    e_use = float(metrics["E_used"])

    load_args = SimpleNamespace(
        experiment="basin_glen",
        exp_dir=args.exp_dir,
        legacy_baseline=False,
    )
    bundle = ev.load_models(load_args, args.inputs_dir)
    mesh_fine = bundle["mesh_fine"]
    u_fine = bundle["u_fine"]
    T_fine = bundle["T_fine"]
    base_fine = bundle["base_fine"]
    base_bed_f = bundle["base_bed_f"]
    base_surf_f = bundle["base_surf_f"]
    fine_dem = bundle["fine_dem"]

    # ------------------------------------------------------------------
    # 1. Exact Comment 5 schema exports (core_samples.csv & grid_samples.csv)
    # ------------------------------------------------------------------
    geochem_xlsx = args.inputs_dir / "argon_and_water_iso_for_JM.xlsx"
    age_recon_csv = args.inputs_dir / "age_reconciliation.csv"
    ages, _, _ = ev.load_geochem_and_ages(geochem_xlsx, age_recon_csv)
    boreholes_df = pd.read_csv(args.repo_dir / "Borehole_Locations/Borehole_locations_DTS.csv")
    core_df, grid_df, fits, window_minima, bh_profiles, vel10, counterflow_diag = ev.evaluate_age_speed_and_grid(
        u_fine, base_bed_f, base_surf_f, fine_dem, boreholes_df, ages, e_use
    )

    ckpt_path = args.exp_dir / "fine_solution.h5"
    ckpt_sha = sha256_file(ckpt_path) if ckpt_path.exists() else ("0" * 64)
    ts_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    core_out = core_df.copy()
    core_out["height_m"] = core_out["height_m"].astype(int)
    core_out["evaluation_elevation_m"] = core_out["bed_elevation_m"] + core_out["height_m"]
    core_out["max_age_uncertainty_ma"] = np.nan
    core_out["age_source"] = "Figure 3 author workbook argon_and_water_iso_for_JM.xlsx"
    core_out["source_notebook"] = "coarse_solve_reevaluation/run_coarse_and_fine_experiments.py"
    core_out["source_repository_commit"] = "basin_glen"
    core_out["source_notebook_sha256"] = ckpt_sha
    core_out["model_checkpoint"] = "basin_glen/fine_solution.h5"
    core_out["model_checkpoint_sha256"] = ckpt_sha
    core_out["export_timestamp_utc"] = ts_utc
    core_out["velocity_scale_factor"] = e_use
    core_out["horizontal_crs"] = "EPSG:3031"
    core_out["sampling_method"] = "exact_local_bed_plus_height_m"
    core_cols = [
        "core_id", "height_m", "x_m", "y_m", "bed_elevation_m", "surface_elevation_m",
        "evaluation_elevation_m", "speed_mm_yr", "max_age_ma", "max_age_uncertainty_ma",
        "age_source", "source_notebook", "source_repository_commit", "source_notebook_sha256",
        "model_checkpoint", "model_checkpoint_sha256", "export_timestamp_utc",
        "velocity_scale_factor", "horizontal_crs", "sampling_method",
    ]
    core_out[core_cols].to_csv(args.out_dir / "basin_glen_comment5_core_samples.csv", index=False)

    grid_out = grid_df.copy()
    grid_out["height_m"] = grid_out["height_m"].astype(int)
    grid_out["evaluation_elevation_m"] = grid_out["bed_elevation_m"] + grid_out["height_m"]
    grid_out["source_notebook"] = "coarse_solve_reevaluation/run_coarse_and_fine_experiments.py"
    grid_out["source_repository_commit"] = "basin_glen"
    grid_out["source_notebook_sha256"] = ckpt_sha
    grid_out["model_checkpoint"] = "basin_glen/fine_solution.h5"
    grid_out["model_checkpoint_sha256"] = ckpt_sha
    grid_out["export_timestamp_utc"] = ts_utc
    grid_out["velocity_scale_factor"] = e_use
    grid_out["horizontal_crs"] = "EPSG:3031"
    grid_out["sampling_method"] = "exact_local_bed_plus_height_m"
    grid_cols = [
        "x_m", "y_m", "height_m", "bed_elevation_m", "surface_elevation_m",
        "evaluation_elevation_m", "speed_mm_yr", "valid_domain", "source_notebook",
        "source_repository_commit", "source_notebook_sha256", "model_checkpoint",
        "model_checkpoint_sha256", "export_timestamp_utc", "velocity_scale_factor",
        "horizontal_crs", "sampling_method",
    ]
    grid_out[grid_cols].to_csv(args.out_dir / "basin_glen_comment5_grid_samples.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Transects (Fig 2 A-A', Fig S5 B-B', Fig S6/FigS08 C-C') + exact diagnostics
    # ------------------------------------------------------------------
    DD1, ZZ1, U1, V1, W1, ux1, uy1 = ev.sample_transect(
        u_fine, args.repo_dir / "flowlines/old_ice_flowlines/ALHIC1902_R_surface_nunatak_test_100m_test_case_4.csv", None, e_use
    )
    DD2, ZZ2, U2, V2, W2, ux2, uy2 = ev.sample_transect(
        u_fine, args.repo_dir / "flowlines/manual_flowlines/csv/ALHIC1901_surface_rema_100m.csv", None, e_use
    )
    DD3, ZZ3, U3, V3, W3, ux3, uy3 = ev.sample_transect(
        u_fine, args.repo_dir / "flowlines/manual_flowlines/csv/ALHIC2401_bed_merged_100m.csv", 19, e_use
    )
    np.savez_compressed(
        args.out_dir / "basin_glen_transects.npz",
        DD1=DD1, ZZ1=ZZ1, U1=U1, V1=V1, W1=W1,
        DD2=DD2, ZZ2=ZZ2, U2=U2, V2=V2, W2=W2,
        DD3=DD3, ZZ3=ZZ3, U3=U3, V3=V3, W3=W3,
    )

    # Exact diagnostics for main text lines 197 & 199:
    xy_base = base_fine.coordinates.dat.data_ro[:, :2]
    zs_all = ev.eval_scalar_2d(base_surf_f, xy_base)
    in_1b = (xy_base[:, 0] >= 508650) & (xy_base[:, 0] <= 509800) & (xy_base[:, 1] >= -1353900) & (xy_base[:, 1] <= -1352800)
    pts_1b_surf = np.column_stack([xy_base[in_1b, 0], xy_base[in_1b, 1], zs_all[in_1b] - 0.1])
    vel_1b_surf = ev.eval_points_safe(u_fine, pts_1b_surf, dim=3) * ev.SPY * 1000.0 * e_use
    ok_1b = np.isfinite(vel_1b_surf).all(axis=1)
    cf_1b = ok_1b & (vel_1b_surf[:, 0] > 0.0)
    cf_1b_horiz = np.hypot(vel_1b_surf[cf_1b, 0], vel_1b_surf[cf_1b, 1]) if cf_1b.any() else np.array([0.0])

    # Also check eastern flank of nunatak in Fig 1b where v > 0 (downslope toward grid north-east)
    cf_1b_any = ok_1b & ((vel_1b_surf[:, 0] > 0.0) | (vel_1b_surf[:, 1] > 0.0))
    cf_1b_any_horiz = np.hypot(vel_1b_surf[cf_1b_any, 0], vel_1b_surf[cf_1b_any, 1]) if cf_1b_any.any() else np.array([0.0])

    cf_aa = np.isfinite(U1) & (DD1 <= 500.0) & ((U1 > 0.0) | (V1 > 0.0))
    cf_aa_horiz = np.hypot(U1[cf_aa], V1[cf_aa]) if cf_aa.any() else np.array([0.0])
    cf_aa_wmin = float(np.nanmin(W1[np.isfinite(W1) & (DD1 <= 500.0)]))

    tf = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    bh = boreholes_df.copy()
    bh["x_m"], bh["y_m"] = tf.transform(bh["E"].to_numpy(float), bh["S"].to_numpy(float))
    bh_idx = bh.set_index("Core")
    u10_comp = {}
    for cid in ("ALHIC2301", "ALHIC2401", "ALHIC1902", "ALHIC2302", "ALHIC1901", "ALHIC1502", "ALHIC2201"):
        xm, ym = float(bh_idx.loc[cid, "x_m"]), float(bh_idx.loc[cid, "y_m"])
        zb = float(ev.eval_scalar_2d(base_bed_f, np.array([[xm, ym]]))[0])
        v10 = ev.eval_points_safe(u_fine, np.array([[xm, ym, zb + 10.0]]), dim=3)[0] * ev.SPY * 1000.0 * e_use
        u10_comp[cid] = {
            "u_mm_yr": float(v10[0]),
            "v_mm_yr": float(v10[1]),
            "w_mm_yr": float(v10[2]),
            "horiz_mm_yr": float(math.hypot(v10[0], v10[1])),
            "speed_mm_yr": float(math.sqrt(float(np.dot(v10, v10)))),
        }

    ok_cc = np.isfinite(U3) & np.isfinite(V3) & np.isfinite(W3)
    cc_horiz = np.hypot(U3[ok_cc], V3[ok_cc])

    diag_out = {
        "E_used": e_use,
        "fig1b_10m_counterflow": counterflow_diag,
        "fig1b_surface_eastward_counterflow_max_horiz_mm_yr": float(np.max(cf_1b_horiz)),
        "fig1b_surface_eastward_counterflow_min_w_mm_yr": float(np.min(vel_1b_surf[cf_1b, 2])) if cf_1b.any() else 0.0,
        "fig1b_surface_pos_uv_max_horiz_mm_yr": float(np.max(cf_1b_any_horiz)),
        "fig1b_surface_pos_uv_min_w_mm_yr": float(np.min(vel_1b_surf[cf_1b_any, 2])) if cf_1b_any.any() else 0.0,
        "transect_AA_flank_max_horiz_mm_yr": float(np.max(cf_aa_horiz)),
        "transect_AA_flank_max_u_mm_yr": float(np.nanmax(U1[np.isfinite(U1) & (DD1 <= 500.0)])),
        "transect_AA_flank_max_v_mm_yr": float(np.nanmax(V1[np.isfinite(V1) & (DD1 <= 500.0)])),
        "transect_AA_flank_min_w_mm_yr": cf_aa_wmin,
        "transect_CC_max_horiz_mm_yr": float(np.max(cc_horiz)),
        "transect_CC_max_abs_w_mm_yr": float(np.max(np.abs(W3[ok_cc]))),
        "borehole_u10_components": u10_comp,
    }
    (args.out_dir / "basin_glen_text_diagnostics.json").write_text(json.dumps(diag_out, indent=2) + "\n")

    # ------------------------------------------------------------------
    # 3. Borehole temperature profiles for publication C2_FigS02.png
    # ------------------------------------------------------------------
    T_clim, T_no_adv = ev.run_transient_temperatures(mesh_fine, u_fine, T_fine, e_use)
    key_order = ("ALHIC2301", "ALHIC2401", "ALHIC1902", "ALHIC2302", "ALHIC1901", "ALHIC2201")
    temp_save = {}
    for name in key_order:
        xm, ym = float(bh_idx.loc[name, "x_m"]), float(bh_idx.loc[name, "y_m"])
        zb = float(ev.eval_scalar_2d(base_bed_f, np.array([[xm, ym]]))[0])
        zs = float(ev.eval_scalar_2d(base_surf_f, np.array([[xm, ym]]))[0])
        z_prof = np.arange(zb + 1.0, zs - 0.5, 5.0)
        pts = np.column_stack([np.full_like(z_prof, xm), np.full_like(z_prof, ym), z_prof])
        t_st = ev.eval_points_safe(T_fine, pts, dim=1)
        t_cl = ev.eval_points_safe(T_clim, pts, dim=1)
        t_na = ev.eval_points_safe(T_no_adv, pts, dim=1)
        ok = np.isfinite(t_st) & np.isfinite(t_cl) & np.isfinite(t_na)
        z_prof, t_st, t_cl, t_na = z_prof[ok], t_st[ok], t_cl[ok], t_na[ok]
        z_off = z_prof[-1] - ev.CORE_SURFACE_Z[name]
        z_aligned = z_prof - z_off
        temp_save[f"{name}_z"] = z_aligned
        temp_save[f"{name}_t_st"] = t_st
        temp_save[f"{name}_t_cl"] = t_cl
        temp_save[f"{name}_t_na"] = t_na
    np.savez_compressed(args.out_dir / "basin_glen_borehole_temps.npz", **temp_save)

    # ------------------------------------------------------------------
    # 4. Export cropped Fig. 1 rasters + fine_dem + vel10 for local pub rendering
    # ------------------------------------------------------------------
    b_xmin, b_xmax, b_ymin, b_ymax = 507000.0, 512500.0, -1356000.0, -1350500.0
    with rasterio.open(args.inputs_dir / "LIMA_virtual_Mosaic_15m.tif") as s_l:
        win_l = from_bounds(b_xmin, b_ymin, b_xmax, b_ymax, s_l.transform)
        lima_crop = s_l.read(window=win_l)
        tf_l = s_l.window_transform(win_l)
        lima_ext = np.array([
            tf_l.c, tf_l.c + tf_l.a * lima_crop.shape[2],
            tf_l.f + tf_l.e * lima_crop.shape[1], tf_l.f,
        ], dtype=float)
    with rasterio.open(args.inputs_dir / "ian_bed_outline_raster_full.tif") as s_g:
        win_g = from_bounds(b_xmin, b_ymin, b_xmax, b_ymax, s_g.transform)
        good_crop = s_g.read(2, window=win_g)
        tf_g = s_g.window_transform(win_g)
        good_ext = np.array([
            tf_g.c, tf_g.c + tf_g.a * good_crop.shape[1],
            tf_g.f + tf_g.e * good_crop.shape[0], tf_g.f,
        ], dtype=float)

    np.savez_compressed(
        args.out_dir / "basin_glen_fig1_data.npz",
        vel10_x=vel10["x"], vel10_y=vel10["y"],
        vel10_uvw_mm_yr=vel10["uvw_mm_yr"], vel10_speed_mm_yr=vel10["speed_mm_yr"],
        fine_xs=fine_dem["xs"], fine_ys=fine_dem["ys"], fine_bed=fine_dem["bed"],
        lima_rgb=np.moveaxis(lima_crop[:3], 0, -1), lima_extent=lima_ext,
        good_mask=good_crop, good_extent=good_ext,
    )

    print("EXPORT_BASIN_GLEN_FOR_MANUSCRIPT_OK!", flush=True)


if __name__ == "__main__":
    main()
