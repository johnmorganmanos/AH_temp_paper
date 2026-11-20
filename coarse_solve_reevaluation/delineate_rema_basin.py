#!/usr/bin/env python3
"""Delineate the modern-day Allan Hills drainage basin from REMA v2.0 surface slopes
and generate a 2D unstructured Gmsh triangular mesh for the coarse ice-flow solve.

Inputs (passed via CLI arguments or environment/defaults):
  - REMA v2.0 500 m filled mosaic GeoTIFF (rema_mosaic_500m_v2.0_filled_cop30_dem.tif)
  - Bedmap3 500 m bed elevation GeoTIFF (bm3_bed.tif)
  - Local merged BedMachine + ground-radar 8 m bed & REMA 8 m surface DEMs
  - Regional 100 m merged BedMachine + radar bed DEM (merged_AH_bed_topo_fix_smooth_100m.tif)
  - Fine-mesh 8 m bed/surface DEMs (for exact fine-domain footprint)
  - GNSS velocities CSV and Spaulding Table S4 CSV (for validation & coverage checks)

Outputs (written into --out-dir):
  - allan_hills_rema_basin.geojson       Smooth EPSG:3031 basin boundary + provenance/metrics
  - allan_hills_basin_2d.msh             Gmsh MSH v2.2 2D triangular mesh (~1200 m spacing)
  - basin_delineation_metrics.json       Quantitative geometry & flowline-alignment metrics
  - basin_delineation_diagnostics.png    3-panel diagnostic figure
  - allan_hills_basin_dems.npz           (optional, written to --dem-out if provided; kept outside git)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, distance_transform_edt, gaussian_filter
from shapely.geometry import Point, Polygon, box, mapping


# Exact extents of the published 1600 m rectangular coarse mesh and 80 m fine mesh (EPSG:3031)
OLD_COARSE_BOUNDS = (502369.5677210526, -1362201.4799846155, 531866.8308789474, -1342728.2492153845)
FINE_BOUNDS = (506902.3839153846, -1356284.4702338027, 512046.01468461537, -1350629.2589661973)


def load_regional_rema_and_bedmap3(rema_path: Path, bm3_path: Path, bounds=(460000.0, 570000.0, -1410000.0, -1300000.0)):
    xmin, xmax, ymin, ymax = bounds
    with rasterio.open(rema_path) as src:
        win = from_bounds(xmin, ymin, xmax, ymax, src.transform).round_offsets().round_lengths()
        rema = src.read(1, window=win).astype(float)
        tf = src.window_transform(win)
        if src.nodata is not None:
            rema[rema == src.nodata] = np.nan
    with rasterio.open(bm3_path) as src:
        win_b = from_bounds(xmin, ymin, xmax, ymax, src.transform).round_offsets().round_lengths()
        bed = src.read(1, window=win_b).astype(float)
        if src.nodata is not None:
            bed[bed == src.nodata] = np.nan

    h, w = rema.shape
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    xs, ys = rasterio.transform.xy(tf, rows.ravel(), cols.ravel())
    xs = np.asarray(xs, dtype=float).reshape(h, w)
    ys = np.asarray(ys, dtype=float).reshape(h, w)

    # Return ascending x1, ascending y1, and [ny, nx] grids
    x1 = xs[0, :]
    y1 = ys[::-1, 0]
    s_raw = rema[::-1, :]
    b_raw = bed[::-1, :]
    return x1, y1, s_raw, b_raw


def compute_slope_alignment(x1, y1, s_fill, gps_df, sigma_km: float):
    dx = float(x1[1] - x1[0])
    s_sm = gaussian_filter(s_fill, sigma=sigma_km * 1000.0 / dx)
    ds_dy, ds_dx = np.gradient(s_sm, dx, dx)
    iu = RegularGridInterpolator((y1, x1), -ds_dx, bounds_error=False, fill_value=np.nan)
    iv = RegularGridInterpolator((y1, x1), -ds_dy, bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([gps_df["y"].to_numpy(float), gps_df["x"].to_numpy(float)])
    mu, mv = iu(pts), iv(pts)
    ang_m = np.arctan2(mv, mu)
    ang_g = np.arctan2(gps_df["v"].to_numpy(float), gps_df["u"].to_numpy(float))
    diff = np.degrees(np.abs(np.arctan2(np.sin(ang_m - ang_g), np.cos(ang_m - ang_g))))
    fx0, fy0, fx1, fy1 = FINE_BOUNDS
    in_fine = (gps_df["x"] >= fx0) & (gps_df["x"] <= fx1) & (gps_df["y"] >= fy0) & (gps_df["y"] <= fy1)
    return {
        "sigma_km": sigma_km,
        "all_54_median_deg": float(np.nanmedian(diff)),
        "all_54_mean_deg": float(np.nanmean(diff)),
        "regional_36_median_deg": float(np.nanmedian(diff[~in_fine])),
        "regional_36_mean_deg": float(np.nanmean(diff[~in_fine])),
        "fine_18_median_deg": float(np.nanmedian(diff[in_fine])),
    }


def delineate_basin_polygon(x1, y1, s_fill, sigma_km: float = 1.5):
    dx = float(x1[1] - x1[0])
    s_sm = gaussian_filter(s_fill, sigma=sigma_km * 1000.0 / dx)
    ds_dy, ds_dx = np.gradient(s_sm, dx, dx)
    spd = np.hypot(ds_dx, ds_dy) + 1e-12
    iu = RegularGridInterpolator((y1, x1), -ds_dx / spd, bounds_error=False, fill_value=0.0)
    iv = RegularGridInterpolator((y1, x1), -ds_dy / spd, bounds_error=False, fill_value=0.0)
    ig = RegularGridInterpolator((y1, x1), spd, bounds_error=False, fill_value=0.0)
    is_sm = RegularGridInterpolator((y1, x1), s_sm, bounds_error=False, fill_value=np.nan)

    # Dense 250 m grid over Allan Hills catchment for sharp continuous RK2 watershed extraction
    gx = np.arange(490000.0, 540250.0, 250.0)
    gy = np.arange(-1380000.0, -1320000.0, 250.0)
    GX, GY = np.meshgrid(gx, gy)
    px = GX.copy()
    py = GY.copy()
    active = np.ones_like(px, dtype=bool)
    step = 200.0
    for _ in range(450):
        if not active.any():
            break
        pts = np.column_stack([py[active], px[active]])
        g = ig(pts)
        vx = iu(pts)
        vy = iv(pts)
        xm = px[active] + 0.5 * step * vx
        ym = py[active] + 0.5 * step * vy
        pm = np.column_stack([ym, xm])
        vxm = iu(pm)
        vym = iv(pm)
        nrm = np.hypot(vxm, vym)
        xn = px[active] + step * np.where(nrm > 1e-6, vxm / (nrm + 1e-12), 0.0)
        yn = py[active] + step * np.where(nrm > 1e-6, vym / (nrm + 1e-12), 0.0)
        inb = (xn > x1[2]) & (xn < x1[-3]) & (yn > y1[2]) & (yn < y1[-3])
        keep = inb & (g >= 3e-4) & (nrm > 1e-6) & (is_sm(np.column_stack([yn, xn])) <= is_sm(pts) + 0.02)
        idx = np.where(active)
        px[idx[0][keep], idx[1][keep]] = xn[keep]
        py[idx[0][keep], idx[1][keep]] = yn[keep]
        active[idx[0][~keep], idx[1][~keep]] = False

    fx0, fy0, fx1, fy1 = FINE_BOUNDS
    in_fine = (GX >= fx0) & (GX <= fx1) & (GY >= fy0) & (GY <= fy1)
    sx_bin = np.round(px / 2000.0).astype(int)
    sy_bin = np.round(py / 2000.0).astype(int)
    sid = sx_bin * 10000 + sy_bin
    fine_sids = np.unique(sid[in_fine])
    mask = np.isin(sid, fine_sids)

    # Trim narrow zero-ice rocky col west of Allan Hills escarpment crest (x < 498 km)
    mask = mask & (GX >= 498000.0)
    mask = binary_fill_holes(mask)
    mask = binary_closing(mask, iterations=2)
    mask = binary_opening(mask, iterations=2)
    mask = binary_fill_holes(mask)

    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(GX, GY, mask.astype(float), levels=[0.5])
    polys = []
    for seg in cs.allsegs[0]:
        if len(seg) >= 4:
            p = Polygon(seg).buffer(0)
            if p.area > 1e6:
                polys.append(p)
    plt.close(fig_tmp)

    raw_basin = max(polys, key=lambda p: p.area)
    # Smooth corner artifacts and simplify to clean ~500 m vertex spacing for Gmsh
    basin = raw_basin.buffer(600.0).buffer(-600.0).simplify(500.0, preserve_topology=True)
    if basin.geom_type == "MultiPolygon":
        basin = max(basin.geoms, key=lambda p: p.area)
    # Ensure CCW orientation
    if not basin.exterior.is_ccw:
        basin = Polygon(list(basin.exterior.coords)[::-1])
    return basin, s_sm, iu, iv, ig


def evaluate_boundary_alignment(poly: Polygon, iu, iv, n_samples_per_edge: int = 20):
    coords = np.asarray(poly.exterior.coords, dtype=float)
    lengths = []
    dots = []
    for i in range(len(coords) - 1):
        p0, p1 = coords[i], coords[i + 1]
        seg_len = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if seg_len < 1.0:
            continue
        t = (p1 - p0) / seg_len
        n_out = np.array([t[1], -t[0]])
        mid_seg = 0.5 * (p0 + p1)
        if poly.contains(Point(mid_seg + 50.0 * n_out)):
            n_out = -n_out
        nsub = max(1, int(math.ceil(seg_len / 250.0))) if n_samples_per_edge == 0 else n_samples_per_edge
        dl = seg_len / nsub
        for frac in np.linspace(0.5 / nsub, 1.0 - 0.5 / nsub, nsub):
            pt = p0 + frac * (p1 - p0)
            vx = float(iu([[pt[1], pt[0]]])[0])
            vy = float(iv([[pt[1], pt[0]]])[0])
            lengths.append(dl)
            dots.append(vx * n_out[0] + vy * n_out[1])
    L = np.asarray(lengths, dtype=float)
    d = np.asarray(dots, dtype=float)
    abs_d = np.abs(d)
    # Length-weighted median of |u_hat . n|
    order = np.argsort(abs_d)
    cdf = np.cumsum(L[order]) / np.sum(L)
    w_med = float(abs_d[order[np.searchsorted(cdf, 0.5)]])
    return {
        "perimeter_km": float(np.sum(L) / 1000.0),
        "mean_u_dot_n": float(np.sum(L * d) / np.sum(L)),
        "mean_abs_u_dot_n": float(np.sum(L * abs_d) / np.sum(L)),
        "weighted_median_abs_u_dot_n": w_med,
        "median_crossing_angle_deg": float(np.degrees(np.arcsin(np.clip(w_med, 0.0, 1.0)))),
        "frac_outflow_gt_0p2": float(np.sum(L[d > 0.2]) / np.sum(L)),
        "frac_inflow_lt_m0p2": float(np.sum(L[d < -0.2]) / np.sum(L)),
        "frac_tangent_within_0p2": float(np.sum(L[abs_d <= 0.2]) / np.sum(L)),
        "sample_lengths": L,
        "sample_dots": d,
    }


def _write_msh2(msh_path: Path, points: np.ndarray, triangles: np.ndarray, boundary_edges: list[tuple[int, int]]):
    """Write a standard Gmsh MSH v2.2 ASCII 2D triangular mesh with Physical Curve 10 & Physical Surface 1."""
    lines = [
        "$MeshFormat",
        "2.2 0 8",
        "$EndMeshFormat",
        "$PhysicalNames",
        "2",
        '1 10 "lateral_wall"',
        '2 1 "basin_interior"',
        "$EndPhysicalNames",
        "$Nodes",
        str(len(points)),
    ]
    for i, (x, y) in enumerate(points, start=1):
        lines.append(f"{i} {x:.6f} {y:.6f} 0.0")
    lines.append("$EndNodes")
    n_elems = len(boundary_edges) + len(triangles)
    lines.extend(["$Elements", str(n_elems)])
    eid = 1
    for a, b in boundary_edges:
        lines.append(f"{eid} 1 2 10 10 {a + 1} {b + 1}")
        eid += 1
    for a, b, c in triangles:
        lines.append(f"{eid} 2 2 1 1 {a + 1} {b + 1} {c + 1}")
        eid += 1
    lines.append("$EndElements")
    msh_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_gmsh_2d_mesh(basin: Polygon, msh_path: Path, lc: float = 1200.0, lc_fine: float = 800.0):
    """Build a clean 2D triangular Gmsh mesh (MSH v2.2) with gentle refinement near the fine domain."""
    coords = np.asarray(basin.exterior.coords, dtype=float)[:-1]
    fx0, fy0, fx1, fy1 = FINE_BOUNDS
    fine_cx, fine_cy = 0.5 * (fx0 + fx1), 0.5 * (fy0 + fy1)

    try:
        import gmsh  # type: ignore

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("allan_hills_basin")
        pt_ids = [gmsh.model.geo.addPoint(float(x), float(y), 0.0, lc) for x, y in coords]
        n = len(pt_ids)
        line_ids = [gmsh.model.geo.addLine(pt_ids[i], pt_ids[(i + 1) % n]) for i in range(n)]
        cl = gmsh.model.geo.addCurveLoop(line_ids)
        surf = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, line_ids, tag=10)
        gmsh.model.addPhysicalGroup(2, [surf], tag=1)
        f1 = gmsh.model.mesh.field.add("MathEval")
        expr = (
            f"{lc_fine:.1f} + ({lc - lc_fine:.1f}) * min(1.0, max(0.0, "
            f"(Sqrt((x - {fine_cx:.1f})^2 + (y - {fine_cy:.1f})^2) - 4500.0) / 8000.0))"
        )
        gmsh.model.mesh.field.setString(f1, "F", expr)
        gmsh.model.mesh.field.setAsBackgroundMesh(f1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(msh_path))
        gmsh.finalize()
    except (ImportError, OSError):
        import meshpy.triangle as mtri  # type: ignore

        # Resample polygon perimeter to ~lc_fine spacing near fine domain and ~lc outside
        poly_pts = []
        n = len(coords)
        for i in range(n):
            p0, p1 = coords[i], coords[(i + 1) % n]
            seg_len = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            mid = 0.5 * (p0 + p1)
            dist = float(np.hypot(mid[0] - fine_cx, mid[1] - fine_cy))
            target_h = lc_fine + (lc - lc_fine) * min(1.0, max(0.0, (dist - 4500.0) / 8000.0))
            nsub = max(1, int(round(seg_len / target_h)))
            for k in range(nsub):
                poly_pts.append(tuple(p0 + (k / nsub) * (p1 - p0)))
        n_b = len(poly_pts)
        facets = [(i, (i + 1) % n_b) for i in range(n_b)]
        info = mtri.MeshInfo()
        info.set_points(poly_pts)
        info.set_facets(facets, facet_markers=[10] * n_b)

        def needs_refinement(vertices, area):
            cx = (vertices[0][0] + vertices[1][0] + vertices[2][0]) / 3.0
            cy = (vertices[0][1] + vertices[1][1] + vertices[2][1]) / 3.0
            dist = math.hypot(cx - fine_cx, cy - fine_cy)
            h_loc = lc_fine + (lc - lc_fine) * min(1.0, max(0.0, (dist - 4500.0) / 8000.0))
            max_area = 0.5 * (h_loc ** 2) * 0.9
            return bool(area > max_area)

        mesh_out = mtri.build(info, refinement_func=needs_refinement, min_angle=28.0)
        pts_arr = np.asarray(mesh_out.points, dtype=float)
        tris_arr = np.asarray(mesh_out.elements, dtype=int)
        # Ensure positive (CCW) orientation on every triangle
        v0 = pts_arr[tris_arr[:, 0]]
        v1 = pts_arr[tris_arr[:, 1]]
        v2 = pts_arr[tris_arr[:, 2]]
        cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
        flip = cross < 0
        tris_arr[flip, 1], tris_arr[flip, 2] = tris_arr[flip, 2].copy(), tris_arr[flip, 1].copy()
        # Extract boundary edges (edges appearing in exactly 1 triangle)
        edge_counts: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
        for a, b, c in tris_arr:
            for u, v in ((a, b), (b, c), (c, a)):
                key = (min(u, v), max(u, v))
                if key in edge_counts:
                    edge_counts[key] = (edge_counts[key][0] + 1, edge_counts[key][1])
                else:
                    edge_counts[key] = (1, (u, v))
        b_edges = [orig for cnt, orig in edge_counts.values() if cnt == 1]
        _write_msh2(msh_path, pts_arr, tris_arr, b_edges)

    # Parse node and triangle counts from MSH2
    nodes = {}
    tris = []
    with msh_path.open("r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh]
    i = 0
    while i < len(lines):
        if lines[i] == "$Nodes":
            n_nodes = int(lines[i + 1])
            for k in range(n_nodes):
                parts = lines[i + 2 + k].split()
                nodes[int(parts[0])] = (float(parts[1]), float(parts[2]))
            i += 2 + n_nodes
        elif lines[i] == "$Elements":
            n_el = int(lines[i + 1])
            for k in range(n_el):
                parts = lines[i + 2 + k].split()
                etype = int(parts[1])
                if etype == 2:
                    ntags = int(parts[2])
                    v = [int(parts[3 + ntags + j]) for j in range(3)]
                    tris.append(v)
            i += 2 + n_el
        else:
            i += 1
    return nodes, tris


def build_blended_basin_dems(
    x1_rema, y1_rema, s_rema, b_bm3,
    bed_100m_path: Path,
    bed_8m_path: Path,
    surf_8m_path: Path,
    out_npz_path: Path,
):
    """Build a seamless 200 m regional surface & bed DEM covering the entire basin:
      - Inside the published large_aoi 8 m box: exact original BedMachine+radar & REMA (feathered over 1.5 km)
      - Outside large_aoi (within 100 m merged DEM box): merged_AH_bed_topo_fix_smooth_100m + REMA 500 m
      - West of x=499.9 km (tiny rocky margin): Bedmap3 bed + REMA 500 m
    """
    gx = np.arange(495000.0, 536200.0, 200.0)
    gy = np.arange(-1378000.0, -1324800.0, 200.0)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GY.ravel(), GX.ravel()])

    # Base: REMA 500 m & Bedmap3 500 m
    is_rema = RegularGridInterpolator((y1_rema, x1_rema), np.where(np.isfinite(s_rema), s_rema, np.nanmedian(s_rema)), bounds_error=False, fill_value=np.nan)
    ib_bm3 = RegularGridInterpolator((y1_rema, x1_rema), np.where(np.isfinite(b_bm3), b_bm3, np.nanmedian(b_bm3)), bounds_error=False, fill_value=np.nan)
    surf = is_rema(pts).reshape(GX.shape)
    bed = ib_bm3(pts).reshape(GX.shape)

    # Layer 2: 100 m BedMachine+radar merged bed (x in [499.9k, 550.0k], y in [-1400.1k, -1299.9k])
    with rasterio.open(bed_100m_path) as src:
        arr = src.read(1).astype(float)
        tf = src.transform
        h, w = arr.shape
        x_100 = tf.c + (np.arange(w) + 0.5) * tf.a
        y_100 = tf.f + (np.arange(h) + 0.5) * tf.e  # descending
        ib_100 = RegularGridInterpolator((y_100[::-1], x_100), arr[::-1, :], bounds_error=False, fill_value=np.nan)
        b100 = ib_100(pts).reshape(GX.shape)
        valid_100 = np.isfinite(b100)
        # Smooth 1 km feather along x=499.9k edge
        dist_in = distance_transform_edt(valid_100) * 200.0
        w100 = np.clip(dist_in / 1000.0, 0.0, 1.0)
        bed = np.where(valid_100, w100 * b100 + (1.0 - w100) * bed, bed)

    # Layer 3: Original 8 m large_aoi bed & surface (downsampled to 160 m bilinear for speed/cleanliness)
    for label, path, target in [("bed", bed_8m_path, "bed"), ("surf", surf_8m_path, "surf")]:
        with rasterio.open(path) as src:
            up = 1.0 / 20.0  # 160 m
            arr = src.read(
                1,
                out_shape=(src.count, int(src.height * up), int(src.width * up)),
                resampling=Resampling.bilinear,
            ).astype(float).squeeze()
            tf = src.transform * src.transform.scale(src.width / arr.shape[-1], src.height / arr.shape[-2])
            h, w = arr.shape
            x_8 = tf.c + (np.arange(w) + 0.5) * tf.a
            y_8 = tf.f + (np.arange(h) + 0.5) * tf.e  # descending
            interp8 = RegularGridInterpolator((y_8[::-1], x_8), arr[::-1, :], bounds_error=False, fill_value=np.nan)
            v8 = interp8(pts).reshape(GX.shape)
            valid8 = np.isfinite(v8)
            dist8 = distance_transform_edt(valid8) * 200.0
            w8 = np.clip(dist8 / 1500.0, 0.0, 1.0)
            if target == "bed":
                bed = np.where(valid8, w8 * v8 + (1.0 - w8) * bed, bed)
            else:
                surf = np.where(valid8, w8 * v8 + (1.0 - w8) * surf, surf)

    # Same negative-thickness protection as published mesh_generating.py (minimum 1 m ice)
    thick = surf - bed
    neg = np.where(thick < 0.0, thick, 0.0)
    zoff = np.where(neg < 0.0, -1.0, 0.0)
    bed_fixed = bed + neg + zoff
    # Also enforce strict positivity (>= 1.0 m) everywhere
    bed_fixed = np.minimum(bed_fixed, surf - 1.0)

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz_path, xs=gx, ys=gy, surf=surf, bed=bed_fixed)
    return gx, gy, surf, bed_fixed


def plot_diagnostics(
    out_png: Path,
    x1, y1, s_raw, s_sm,
    gx_dem, gy_dem, surf_dem, bed_dem,
    basin: Polygon,
    nodes, tris,
    basin_align, rect_align,
    gps_df, spaulding_df,
):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.8), layout="constrained")
    fx0, fy0, fx1, fy1 = FINE_BOUNDS
    rx0, ry0, rx1, ry1 = OLD_COARSE_BOUNDS
    bx, by = basin.exterior.xy

    # Panel A: REMA 500m surface, streamlines, old vs new coarse domains
    ax = axes[0]
    ds_dy, ds_dx = np.gradient(s_sm, float(y1[1] - y1[0]), float(x1[1] - x1[0]))
    im = ax.pcolormesh(x1 / 1e3, y1 / 1e3, s_raw, cmap="terrain", vmin=1500, vmax=2250, shading="auto")
    cs = ax.contour(x1 / 1e3, y1 / 1e3, s_sm, levels=np.arange(1500, 2350, 30), colors="k", linewidths=0.45, alpha=0.65)
    ax.clabel(cs, fontsize=7, fmt="%d", inline=True)
    ax.streamplot(x1 / 1e3, y1 / 1e3, -ds_dx, -ds_dy, color="navy", density=2.6, linewidth=0.65, arrowsize=0.75)
    ax.plot(np.array(bx) / 1e3, np.array(by) / 1e3, color="cyan", lw=2.6, label="REMA drainage basin")
    ax.plot(np.array(bx) / 1e3, np.array(by) / 1e3, color="blue", lw=1.2, ls="--")
    ax.add_patch(plt.Rectangle((rx0 / 1e3, ry0 / 1e3), (rx1 - rx0) / 1e3, (ry1 - ry0) / 1e3, fill=False, ec="red", lw=2.0, ls="--", label="Original rect. coarse"))
    ax.add_patch(plt.Rectangle((fx0 / 1e3, fy0 / 1e3), (fx1 - fx0) / 1e3, (fy1 - fy0) / 1e3, fill=False, ec="yellow", lw=2.2, label="Fine domain (80 m)"))
    ax.scatter(spaulding_df["x_m"] / 1e3, spaulding_df["y_m"] / 1e3, s=11, c="lime", ec="k", lw=0.35, zorder=9, label="Spaulding stakes")
    ax.quiver(gps_df["x"] / 1e3, gps_df["y"] / 1e3, gps_df["u"], gps_df["v"], color="magenta", scale=3.2, width=0.004, zorder=10, label="GNSS velocities")
    ax.set_xlim(493, 536)
    ax.set_ylim(-1377, -1325)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km, EPSG:3031)")
    ax.set_ylabel("Northing (km, EPSG:3031)")
    ax.set_title("A. REMA v2.0 Surface Slopes & Drainage Basin")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    fig.colorbar(im, ax=ax, shrink=0.78, label="REMA surface elevation (m)")

    # Panel B: Seamlessly blended ice thickness + Gmsh triangulation
    ax = axes[1]
    thick = surf_dem - bed_dem
    im2 = ax.pcolormesh(gx_dem / 1e3, gy_dem / 1e3, thick, cmap="viridis", vmin=0, vmax=1100, shading="auto")
    # Overlay triangulation edges
    xs_n = np.array([nodes[k][0] for k in sorted(nodes)])
    ys_n = np.array([nodes[k][1] for k in sorted(nodes)])
    id_to_idx = {k: i for i, k in enumerate(sorted(nodes))}
    tri_idx = np.array([[id_to_idx[v] for v in t] for t in tris])
    ax.triplot(xs_n / 1e3, ys_n / 1e3, tri_idx, color="white", lw=0.45, alpha=0.7)
    ax.add_patch(plt.Rectangle((fx0 / 1e3, fy0 / 1e3), (fx1 - fx0) / 1e3, (fy1 - fy0) / 1e3, fill=False, ec="yellow", lw=2.2, label="Fine domain (80 m)"))
    ax.set_xlim(495, 532)
    ax.set_ylim(-1377, -1325)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km, EPSG:3031)")
    ax.set_ylabel("Northing (km, EPSG:3031)")
    ax.set_title(f"B. Unstructured Basin Mesh ({len(tris)} tris, {len(nodes)} nodes)")
    fig.colorbar(im2, ax=ax, shrink=0.78, label="Merged ice thickness (m)")

    # Panel C: Boundary cross-flow comparison |u_hat . n|
    ax = axes[2]
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(
        np.abs(rect_align["sample_dots"]),
        bins=bins,
        weights=rect_align["sample_lengths"] / np.sum(rect_align["sample_lengths"]),
        alpha=0.55,
        color="indianred",
        edgecolor="darkred",
        label=f'Original rectangle (med={rect_align["weighted_median_abs_u_dot_n"]:.2f})',
    )
    ax.hist(
        np.abs(basin_align["sample_dots"]),
        bins=bins,
        weights=basin_align["sample_lengths"] / np.sum(basin_align["sample_lengths"]),
        alpha=0.65,
        color="steelblue",
        edgecolor="navy",
        label=f'REMA drainage basin (med={basin_align["weighted_median_abs_u_dot_n"]:.2f})',
    )
    ax.axvline(rect_align["weighted_median_abs_u_dot_n"], color="darkred", ls="--", lw=1.6)
    ax.axvline(basin_align["weighted_median_abs_u_dot_n"], color="navy", ls="--", lw=1.6)
    ax.set_xlabel(r"Normalized cross-boundary slope flow $|\hat{\mathbf{u}}_{\mathrm{slope}} \cdot \mathbf{n}|$")
    ax.set_ylabel("Fraction of domain perimeter")
    ax.set_title("C. Flowline Alignment Along Coarse Domain Boundary")
    ax.grid(alpha=0.35)
    ax.legend(loc="upper right", fontsize=8.5)

    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rema-500m", type=Path, required=True)
    parser.add_argument("--bm3-bed", type=Path, required=True)
    parser.add_argument("--bed-100m", type=Path, required=True)
    parser.add_argument("--bed-8m", type=Path, required=True)
    parser.add_argument("--surf-8m", type=Path, required=True)
    parser.add_argument("--gps-csv", type=Path, required=True)
    parser.add_argument("--spaulding-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dem-out", type=Path, required=True)
    parser.add_argument("--sigma-km", type=float, default=1.5)
    parser.add_argument("--lc", type=float, default=1200.0)
    parser.add_argument("--lc-fine", type=float, default=800.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gps_df = pd.read_csv(args.gps_csv)
    spaulding_df = pd.read_csv(args.spaulding_csv)

    x1, y1, s_raw, b_bm3 = load_regional_rema_and_bedmap3(args.rema_500m, args.bm3_bed)
    s_fill = np.where(np.isfinite(s_raw), s_raw, np.nanmedian(s_raw))

    # Quantify smoothing scale selection against observed GNSS velocities
    smoothing_scan = [compute_slope_alignment(x1, y1, s_fill, gps_df, sig) for sig in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]

    basin, s_sm, iu, iv, ig = delineate_basin_polygon(x1, y1, s_fill, sigma_km=args.sigma_km)
    fine_box = box(*FINE_BOUNDS)
    rect_box = box(*OLD_COARSE_BOUNDS)
    if not basin.contains(fine_box):
        raise RuntimeError("Delineated drainage basin does not fully enclose the 80 m fine domain!")

    basin_align = evaluate_boundary_alignment(basin, iu, iv)
    rect_align = evaluate_boundary_alignment(rect_box, iu, iv)

    # Coverage of GNSS and Spaulding sites
    gps_in_basin = sum(basin.contains(Point(r["x"], r["y"])) for _, r in gps_df.iterrows())
    sp_finite = spaulding_df[np.isfinite(spaulding_df["observed_smb_m_ice_eq_yr"])]
    sp_ice_firn = sp_finite[sp_finite["surface_density_Mg_m3"] >= 0.8]
    sp_all_in = sum(basin.contains(Point(r["x_m"], r["y_m"])) for _, r in sp_finite.iterrows())
    sp_ice_in = sum(basin.contains(Point(r["x_m"], r["y_m"])) for _, r in sp_ice_firn.iterrows())

    # Generate 2D Gmsh mesh
    msh_path = args.out_dir / "allan_hills_basin_2d.msh"
    nodes, tris = generate_gmsh_2d_mesh(basin, msh_path, lc=args.lc, lc_fine=args.lc_fine)

    # Build seamlessly blended regional DEMs (saved to --dem-out outside git)
    gx_dem, gy_dem, surf_dem, bed_dem = build_blended_basin_dems(
        x1, y1, s_raw, b_bm3,
        args.bed_100m, args.bed_8m, args.surf_8m,
        args.dem_out,
    )

    # Save GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3031"}},
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Allan Hills modern-day REMA v2.0 surface-slope drainage basin",
                    "sigma_km": args.sigma_km,
                    "area_km2": round(basin.area / 1e6, 3),
                    "perimeter_km": round(basin.length / 1e3, 3),
                    "gmsh_nodes_2d": len(nodes),
                    "gmsh_triangles_2d": len(tris),
                    "lc_outer_m": args.lc,
                    "lc_fine_m": args.lc_fine,
                },
                "geometry": mapping(basin),
            }
        ],
    }
    (args.out_dir / "allan_hills_rema_basin.geojson").write_text(json.dumps(geojson, indent=2) + "\n", encoding="utf-8")

    # Save metrics JSON (strip array fields from align dicts)
    def _clean(d):
        return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}

    metrics = {
        "sigma_km": args.sigma_km,
        "basin_area_km2": float(basin.area / 1e6),
        "old_rect_area_km2": float(rect_box.area / 1e6),
        "basin_perimeter_km": float(basin.length / 1e3),
        "basin_contains_fine_domain": bool(basin.contains(fine_box)),
        "min_dist_fine_to_basin_boundary_m": float(fine_box.boundary.distance(basin.boundary)),
        "gmsh_2d_nodes": len(nodes),
        "gmsh_2d_triangles": len(tris),
        "gnss_sites_inside_basin": int(gps_in_basin),
        "gnss_sites_total": int(len(gps_df)),
        "spaulding_all_finite_inside_basin": int(sp_all_in),
        "spaulding_all_finite_total": int(len(sp_finite)),
        "spaulding_ice_firn_inside_basin": int(sp_ice_in),
        "spaulding_ice_firn_total": int(len(sp_ice_firn)),
        "basin_boundary_flow_alignment": _clean(basin_align),
        "old_rect_boundary_flow_alignment": _clean(rect_align),
        "gnss_slope_smoothing_scan": smoothing_scan,
    }
    (args.out_dir / "basin_delineation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    plot_diagnostics(
        args.out_dir / "basin_delineation_diagnostics.png",
        x1, y1, s_raw, s_sm,
        gx_dem, gy_dem, surf_dem, bed_dem,
        basin, nodes, tris,
        basin_align, rect_align,
        gps_df, spaulding_df,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
