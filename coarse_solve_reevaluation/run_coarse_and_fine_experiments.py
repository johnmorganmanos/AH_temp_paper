#!/usr/bin/env python3
"""Run coarse-to-fine 3D thermomechanical full-Stokes experiments for the Allan Hills BIA.

Supported experiments (--experiment):
  - baseline          : Load & verify published rectangular Newtonian coarse + 80 m Glen fine checkpoints,
                        or run fresh with --fresh-baseline to verify solver reproduction.
  - rect_glen         : Published 1600 m rectangular coarse mesh with coupled Glen's-law rheology (n=3, A(T))
                        -> coupled as side BCs into the 80 m Glen's-law fine mesh.
  - basin_glen        : Modern-day REMA v2.0 drainage-basin coarse mesh (~800-1200 m unstructured triangles)
                        with coupled Glen's-law rheology and free-slip impermeable lateral walls (u . n = 0)
                        -> coupled as side BCs into the 80 m Glen's-law fine mesh.
  - basin_glen_noslip : Same REMA drainage-basin coarse mesh + Glen's-law rheology, but with no-slip
                        lateral walls (u = 0) to separate domain geometry from lateral BC type.
  - basin_newtonian   : Same REMA drainage-basin coarse mesh with Newtonian rheology (mu_0 = 5e15 Pa s)
                        and free-slip impermeable lateral walls -> coupled into the 80 m fine mesh
                        to isolate domain-geometry effects from coarse-rheology effects.

Run in parallel with MPI for speed, e.g.:
  OMP_NUM_THREADS=1 mpiexec -n 8 python run_coarse_and_fine_experiments.py \
      --experiment rect_glen --inputs-dir /data/fast1/bpl7/ah-coarse-quest/inputs \
      --out-dir /data/fast1/bpl7/ah-coarse-quest/outputs/rect_glen
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

from mpi4py import MPI
import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.interpolate import RegularGridInterpolator, griddata

import firedrake as fd
from firedrake import (
    CellDiameter,
    Constant,
    DirichletBC,
    ExtrudedMesh,
    FacetNormal,
    FiniteElement,
    Function,
    FunctionSpace,
    Mesh,
    MixedVectorSpaceBasis,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    RectangleMesh,
    SpatialCoordinate,
    TensorProductElement,
    VectorFunctionSpace,
    VectorSpaceBasis,
    as_vector,
    assemble,
    div,
    dot,
    ds_b,
    ds_t,
    ds_v,
    dx,
    exp,
    grad,
    inner,
    max_value,
    sqrt,
    sym,
    tr,
)

try:
    from firedrake.__future__ import interpolate as fd_interpolate
except ImportError:
    fd_interpolate = fd.interpolate

# ---------------------------------------------------------------------------
# Physical constants (exact match to scripts/AH_temp_funcs.py & model_runner.py)
# ---------------------------------------------------------------------------
SPY = 365.25 * 24.0 * 3600.0
RHO = 917.0                 # kg m^-3
CP = 2180.0                 # J kg^-1 K^-1
K_THERM = 2.22              # W m^-1 K^-1
GRAV = -9.81                # m s^-2
MU_0 = 5.0e15               # Pa s (initial / Newtonian viscosity)
A_0 = 3.5e-25               # Pa^-3 s^-1 at 263 K (Cuffey & Paterson, 2010)
N_GLEN = 3.0
Q_ACT = 6.0e4               # J mol^-1
R_GAS = 8.31446261815324    # J mol^-1 K^-1
GEO_FLUX_INIT = 0.05        # W m^-2 (model_runner.py initial thermal solve)
GEO_FLUX_COUPLED = 0.06     # W m^-2 (AH_temp_funcs.py viscosity_updater_3d)
T_SURFACE_MEAN = -33.0      # deg C at 2000 m
T_LAPSE_RATE = 0.01         # deg C m^-1
EPS_FLOOR_SQ = 1.0e-60      # (s^-1)^2, 34+ orders below physical strain rates; prevents 0**(-2/3)
N_LAYERS = 5
MAX_IT = 20                 # iterations i = 0..20 (21 coupled updates), matching AH_temp_funcs.py


def rank0() -> bool:
    return fd.COMM_WORLD.rank == 0


def log(msg: str) -> None:
    if rank0():
        print(msg, flush=True)


def mumps_solver_params(icntl14: int = 120) -> dict:
    return {
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_14": icntl14,
        "mat_mumps_icntl_24": 1,
    }


def load_resampled_raster_dem(bed_path: Path, surf_path: Path, upscale_factor: float) -> dict:
    """Exact replica of mesh_generating.py / model_runner.py DEM resampling and thickness guard."""
    with rasterio.open(bed_path) as sb, rasterio.open(surf_path) as ss:
        h = int(sb.height * upscale_factor)
        w = int(sb.width * upscale_factor)
        bed = sb.read(1, out_shape=(sb.count, h, w), resampling=Resampling.bilinear).astype(float).squeeze()
        surf = ss.read(1, out_shape=(ss.count, h, w), resampling=Resampling.bilinear).astype(float).squeeze()
        tf = ss.transform * ss.transform.scale(ss.width / surf.shape[-1], ss.height / surf.shape[-2])
    thick = surf - bed
    thick[thick > 0] = 0.0
    zero_off = copy.deepcopy(thick)
    zero_off[zero_off < 0] = -1.0
    bed = bed + thick + zero_off
    height, width = surf.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(tf, rows.ravel(), cols.ravel())
    xs = np.asarray(xs, dtype=float).reshape(height, width)
    ys = np.asarray(ys, dtype=float).reshape(height, width)
    return {"bed": bed, "surf": surf, "xs": xs, "ys": ys, "width": width, "height": height}


def build_rect_extruded_mesh(dem: dict, layers: int = N_LAYERS, name: str = "rect_mesh"):
    """Build (mesh_phys, unit_extruded_mesh, base_mesh, base_bed, base_surf) matching model_runner.py."""
    xs, ys, surf, bed = dem["xs"], dem["ys"], dem["surf"], dem["bed"]
    nx = dem["width"] - 1
    ny = dem["height"] - 1
    base_mesh = RectangleMesh(
        nx, ny, float(xs.max()), float(ys.max()),
        originX=float(xs.min()), originY=float(ys.min()),
        name=f"{name}_base",
    )
    unit_mesh = ExtrudedMesh(base_mesh, layers=layers, name=f"{name}_unit")
    base_fs = FunctionSpace(base_mesh, "CG", 1)
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    xi = base_mesh.coordinates.dat.data_ro[:, :2]
    surf_i = griddata(pts, surf.ravel(), xi, method="nearest", fill_value=np.nan, rescale=False)
    bed_i = griddata(pts, bed.ravel(), xi, method="nearest", fill_value=np.nan, rescale=False)

    base_bed = Function(base_fs, name=f"{name}_bed")
    base_surf = Function(base_fs, name=f"{name}_surf")
    base_bed.dat.data_wo[:] = bed_i
    base_surf.dat.data_wo[:] = surf_i

    ext_el = FiniteElement("R", "interval", 0)
    ext_fs = FunctionSpace(unit_mesh, TensorProductElement(base_fs.ufl_element(), ext_el))
    ext_bed = Function(ext_fs)
    ext_surf = Function(ext_fs)
    # Match base DOFs to extruded constant-in-z DOFs safely in both serial & parallel MPI
    # using base coordinates (ext_fs shares horizontal CG1 DOF layout on every rank!)
    ext_bed.dat.data_wo[:] = bed_i
    ext_surf.dat.data_wo[:] = surf_i

    x, y, z = SpatialCoordinate(unit_mesh)
    new_coords = assemble(
        fd_interpolate(
            as_vector([x, y, (1.0 - z) * ext_bed + z * ext_surf]),
            unit_mesh.coordinates.function_space(),
        )
    )
    mesh_phys = Mesh(new_coords, name=name)
    return mesh_phys, unit_mesh, base_mesh, base_bed, base_surf


def build_basin_extruded_mesh(msh_path: Path, dem_npz_path: Path, layers: int = N_LAYERS, name: str = "basin_mesh"):
    """Build terrain-following 3D extruded mesh from 2D Gmsh basin triangulation + merged DEMs."""
    base_mesh = Mesh(str(msh_path), name=f"{name}_base")
    unit_mesh = ExtrudedMesh(base_mesh, layers=layers, name=f"{name}_unit")
    base_fs = FunctionSpace(base_mesh, "CG", 1)

    d = np.load(dem_npz_path)
    gx, gy, surf_g, bed_g = d["xs"], d["ys"], d["surf"], d["bed"]
    isurf = RegularGridInterpolator((gy, gx), surf_g, method="linear", bounds_error=False, fill_value=None)
    ibed = RegularGridInterpolator((gy, gx), bed_g, method="linear", bounds_error=False, fill_value=None)

    xi = base_mesh.coordinates.dat.data_ro[:, :2]
    qpts = np.column_stack([xi[:, 1], xi[:, 0]])
    surf_i = np.asarray(isurf(qpts), dtype=float)
    bed_i = np.asarray(ibed(qpts), dtype=float)
    # Enforce >= 1 m minimum thickness (matching published convention)
    bed_i = np.minimum(bed_i, surf_i - 1.0)
    if not (np.isfinite(surf_i).all() and np.isfinite(bed_i).all()):
        raise RuntimeError("Non-finite DEM values interpolated onto basin mesh nodes!")

    base_bed = Function(base_fs, name=f"{name}_bed")
    base_surf = Function(base_fs, name=f"{name}_surf")
    base_bed.dat.data_wo[:] = bed_i
    base_surf.dat.data_wo[:] = surf_i

    ext_el = FiniteElement("R", "interval", 0)
    ext_fs = FunctionSpace(unit_mesh, TensorProductElement(base_fs.ufl_element(), ext_el))
    ext_bed = Function(ext_fs)
    ext_surf = Function(ext_fs)
    ext_bed.dat.data_wo[:] = bed_i
    ext_surf.dat.data_wo[:] = surf_i

    x, y, z = SpatialCoordinate(unit_mesh)
    new_coords = assemble(
        fd_interpolate(
            as_vector([x, y, (1.0 - z) * ext_bed + z * ext_surf]),
            unit_mesh.coordinates.function_space(),
        )
    )
    mesh_phys = Mesh(new_coords, name=name)
    return mesh_phys, unit_mesh, base_mesh, base_bed, base_surf


def eps_sym(u):
    return sym(grad(u))


def solve_stokes_step(
    mesh,
    Y,
    mu_expr,
    lateral_mode: str,
    side_ids: list,
    u_bc_func=None,
    use_nullspace: bool = False,
    nitsche_gamma: float = 100.0,
    icntl14: int = 120,
):
    """Solve linear Stokes momentum + incompressibility with prescribed viscosity expression mu_expr.

    lateral_mode:
      - 'noslip'   : u = 0 on all side_ids and bottom; traction-free top
      - 'freeslip' : u . n = 0 and zero tangential shear traction on vertical walls (ds_v)
                     via symmetric Nitsche's method; u = 0 on bottom; traction-free top
      - 'dirichlet': u = u_bc_func on all side_ids; u = 0 on bottom; traction-free top
    """
    flow = Function(Y, name="flow")
    u, p = fd.split(flow)
    v, q = fd.TestFunctions(Y)

    dX = fd.Measure("dx", domain=mesh)
    dSv = fd.Measure("ds_v", domain=mesh)

    tau = 2.0 * mu_expr * eps_sym(u)
    f = Constant(RHO) * as_vector((0.0, 0.0, Constant(GRAV)))
    F = (inner(tau, eps_sym(v)) - q * div(u) - p * div(v) - inner(f, v)) * dX

    bcs = [DirichletBC(Y.sub(0), as_vector((0.0, 0.0, 0.0)), "bottom")]
    if lateral_mode == "noslip":
        for sid in side_ids:
            bcs.append(DirichletBC(Y.sub(0), as_vector((0.0, 0.0, 0.0)), sid))
    elif lateral_mode == "dirichlet":
        assert u_bc_func is not None
        for sid in side_ids:
            bcs.append(DirichletBC(Y.sub(0), u_bc_func, sid))
    elif lateral_mode == "freeslip":
        n_vec = FacetNormal(mesh)
        h_cell = CellDiameter(mesh)
        gamma = Constant(nitsche_gamma)
        sigma_u_nn = dot(n_vec, dot(2.0 * mu_expr * eps_sym(u), n_vec)) - p
        sigma_v_nn = dot(n_vec, dot(2.0 * mu_expr * eps_sym(v), n_vec)) - q
        # Symmetric Nitsche enforcing u . n = 0 with zero tangential traction on lateral boundary ds_v
        F += (
            -sigma_u_nn * dot(v, n_vec)
            - sigma_v_nn * dot(u, n_vec)
            + (gamma * 2.0 * mu_expr / h_cell) * dot(u, n_vec) * dot(v, n_vec)
        ) * dSv
    else:
        raise ValueError(f"Unknown lateral_mode: {lateral_mode}")

    prob = NonlinearVariationalProblem(F, flow, bcs)
    kwargs = {"solver_parameters": mumps_solver_params(icntl14=icntl14)}
    if use_nullspace:
        basis = VectorSpaceBasis(constant=True, comm=mesh.comm)
        kwargs["nullspace"] = MixedVectorSpaceBasis(Y, [Y.sub(0), basis])
    solver = NonlinearVariationalSolver(prob, **kwargs)
    solver.solve()
    return flow


def solve_steady_temperature(mesh, V_T, u_expr, geo_flux_val: float, advection_form: str = "strong"):
    """Solve steady advection-diffusion temperature equation with exact sidewall grad(T).n = 0.

    Note: The original scripts/AH_temp_funcs.py wrote advection in integrated-by-parts form
    `-rho * c * T * inner(u, grad(phi)) * dx` without the lateral boundary flux
    `+int_{ds_v} rho * c * T * (u.n) * phi`, which imposes a spurious Robin condition
    `k grad(T).n - rho*c*T*(u.n) = 0` on inflow walls (`u.n < 0`). That stayed benign under
    slow Newtonian coarse BCs, but destabilizes under faster Glen's-flow coarse inflow.
    Using the standard form `+rho * c * inner(u, grad(T)) * phi * dx` genuinely enforces the
    manuscript's stated zero normal temperature gradient (`grad(T).n = 0`) on lateral walls.
    """
    _, _, z = SpatialCoordinate(mesh)
    T = Function(V_T, name="temperature")
    phi = fd.TestFunction(V_T)
    dX = fd.Measure("dx", domain=mesh)
    dSb = fd.Measure("ds_b", domain=mesh)

    F_diff = Constant(K_THERM) * inner(grad(T), grad(phi)) * dX
    if advection_form == "ibp":
        F_adv = -Constant(RHO) * Constant(CP) * T * inner(u_expr, grad(phi)) * dX
    elif advection_form == "strong":
        F_adv = Constant(RHO) * Constant(CP) * inner(u_expr, grad(T)) * phi * dX
    else:
        raise ValueError(f"Unknown advection_form: {advection_form}")
    F_geo = -Constant(geo_flux_val) * phi * dSb
    F = F_diff + F_adv + F_geo

    T_surf_expr = Constant(T_SURFACE_MEAN) - (z - Constant(2000.0)) * Constant(T_LAPSE_RATE)
    bc_top = DirichletBC(V_T, T_surf_expr, "top")
    fd.solve(
        F == 0,
        T,
        bcs=[bc_top],
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    return T


def run_thermomechanical_solve(
    mesh,
    lateral_mode: str,
    side_ids: list,
    u_bc_func=None,
    glen: bool = True,
    max_it: int = MAX_IT,
    use_nullspace: bool = False,
    nitsche_gamma: float = 100.0,
    icntl14: int = 120,
    label: str = "model",
):
    """Run initial Newtonian Stokes solve, optional steady T + coupled Glen viscosity iterations."""
    t_start = time.time()
    V_p = FunctionSpace(mesh, "CG", 1)
    V_u = VectorFunctionSpace(mesh, "CG", 3)
    Y = V_u * V_p
    V_T = FunctionSpace(mesh, "CG", 1)

    log(f"[{label}] Y.dim={Y.dim():,}, V_T.dim={V_T.dim():,}, lateral={lateral_mode}, glen={glen}")

    # Step 1: Initial Newtonian Stokes solve
    t0 = time.time()
    flow = solve_stokes_step(
        mesh, Y, Constant(MU_0),
        lateral_mode=lateral_mode, side_ids=side_ids, u_bc_func=u_bc_func,
        use_nullspace=use_nullspace, nitsche_gamma=nitsche_gamma, icntl14=icntl14,
    )
    log(f"[{label}] Initial Newtonian Stokes done in {time.time() - t0:.1f}s")

    # Initial steady temperature solve (always compute so even Newtonian coarse has T diagnostic)
    u_curr, _ = fd.split(flow)
    T = solve_steady_temperature(mesh, V_T, u_curr, GEO_FLUX_INIT)

    abs_residuals = []
    rel_residuals = []
    vel_rel_changes = []

    if glen:
        mu_prev_field = Function(V_T, name="mu_prev")
        mu_prev_field.interpolate(Constant(MU_0))
        u_prev_vec = flow.subfunctions[0].copy(deepcopy=True)

        prev_res = None
        for i in range(max_it + 1):
            ti = time.time()
            u_old, _ = fd.split(flow)
            eps_ = eps_sym(u_old)
            eps_eff = sqrt(0.5 * (inner(eps_, eps_) + tr(eps_) ** 2) + Constant(EPS_FLOOR_SQ))
            A_new = Constant(A_0) * exp(-(Constant(Q_ACT) / Constant(R_GAS)) * (1.0 / (T + Constant(273.15)) - 1.0 / Constant(263.0)))
            mu_new = 0.5 * (A_new ** (-1.0 / N_GLEN)) * (eps_eff ** ((1.0 / N_GLEN) - 1.0))

            # Project viscosity onto V_T using direct LU for exact residual tracking matching AH_temp_funcs.py
            mu_curr_field = fd.project(
                mu_new,
                V_T,
                solver_parameters={
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                },
            )

            # Solve Stokes with updated Glen viscosity expression
            flow = solve_stokes_step(
                mesh, Y, mu_new,
                lateral_mode=lateral_mode, side_ids=side_ids, u_bc_func=u_bc_func,
                use_nullspace=use_nullspace, nitsche_gamma=nitsche_gamma, icntl14=icntl14,
            )

            # Solve steady temperature with new velocity
            u_new, _ = fd.split(flow)
            T = solve_steady_temperature(mesh, V_T, u_new, GEO_FLUX_COUPLED)

            # Compute global MPI-consistent mean nodal absolute viscosity update (matching AH_temp_funcs.py)
            diff_mu = Function(V_T)
            diff_mu.assign(mu_curr_field - mu_prev_field)
            local_sum = float(np.sum(np.abs(diff_mu.dat.data_ro)))
            local_n = int(diff_mu.dat.data_ro.size)
            global_sum = mesh.comm.allreduce(local_sum)
            global_n = mesh.comm.allreduce(local_n)
            residual = global_sum / max(global_n, 1)
            theta_abs = residual / MU_0
            abs_residuals.append(float(theta_abs))

            # Also track L2 velocity relative change
            du_norm = fd.norm(flow.subfunctions[0] - u_prev_vec)
            u_norm = fd.norm(flow.subfunctions[0]) + 1e-30
            vel_rel = float(du_norm / u_norm)
            vel_rel_changes.append(vel_rel)

            if i == 0:
                prev_res = residual
                rel_str = "n/a"
            else:
                theta_rel = (residual - prev_res) / residual if residual > 0 else 0.0
                rel_residuals.append(float(theta_rel))
                prev_res = residual
                rel_str = f"{theta_rel:+.4f}"

            t_min = float(mesh.comm.allreduce(float(T.dat.data_ro.min()) if len(T.dat.data_ro) else 1e9, op=MPI.MIN))
            t_max = float(mesh.comm.allreduce(float(T.dat.data_ro.max()) if len(T.dat.data_ro) else -1e9, op=MPI.MAX))
            log(
                f"[{label}] iter {i:2d}/{max_it}: dt={time.time()-ti:5.1f}s | "
                f"theta_abs={theta_abs:.4e}, theta_rel={rel_str}, du/u={vel_rel:.4e}, T=[{t_min:.1f},{t_max:.1f}]C"
            )
            mu_prev_field.assign(mu_curr_field)
            u_prev_vec.assign(flow.subfunctions[0])

    # Flow quality diagnostics (incompressibility & lateral normal-flux leak)
    u_fn = flow.subfunctions[0]
    dX = fd.Measure("dx", domain=mesh)
    dSv = fd.Measure("ds_v", domain=mesh)
    n_vec = FacetNormal(mesh)
    div_l2 = math.sqrt(float(assemble(div(u_fn) ** 2 * dX)))
    grad_l2 = math.sqrt(float(assemble(inner(grad(u_fn), grad(u_fn)) * dX))) + 1e-30
    un_l2 = math.sqrt(float(assemble(dot(u_fn, n_vec) ** 2 * dSv)))
    u_side_l2 = math.sqrt(float(assemble(inner(u_fn, u_fn) * dSv))) + 1e-30
    local_u_max = float(np.max(np.linalg.norm(u_fn.dat.data_ro, axis=1))) if len(u_fn.dat.data_ro) else 0.0
    u_max = float(mesh.comm.allreduce(local_u_max, op=MPI.MAX))

    diag = {
        "label": label,
        "Y_dim": int(Y.dim()),
        "V_T_dim": int(V_T.dim()),
        "lateral_mode": lateral_mode,
        "glen": glen,
        "thermal_advection_form": "strong",
        "max_it": max_it if glen else 0,
        "wall_seconds": float(time.time() - t_start),
        "div_u_rel_l2": float(div_l2 / grad_l2),
        "lateral_un_rel_l2": float(un_l2 / u_side_l2),
        "u_max_m_yr_unscaled": float(u_max * SPY),
        "theta_abs": abs_residuals,
        "theta_rel": rel_residuals,
        "vel_rel_change": vel_rel_changes,
    }
    log(
        f"[{label}] DONE in {diag['wall_seconds']:.1f}s | u_max(E=1)={diag['u_max_m_yr_unscaled']:.4f} m/yr | "
        f"div(u)/|grad u|={diag['div_u_rel_l2']:.2e} | |u.n|/|u|(side)={diag['lateral_un_rel_l2']:.2e}"
    )
    return flow, T, Y, V_T, diag


def transfer_coarse_velocity_to_fine_bc(
    flow_coarse,
    unit_coarse,
    unit_fine,
    mesh_fine,
    Y_fine,
):
    """Transfer coarse velocity onto the terrain-following fine mesh via unit extruded meshes [0, 1]."""
    V_u_coarse_norm = VectorFunctionSpace(unit_coarse, "CG", 3)
    u_coarse_norm = Function(V_u_coarse_norm, name="u_coarse_norm")
    u_coarse_norm.dat.data_wo[:] = flow_coarse.subfunctions[0].dat.data_ro

    V_u_fine_norm = VectorFunctionSpace(unit_fine, "CG", 3)
    u_fine_norm = assemble(fd_interpolate(u_coarse_norm, V_u_fine_norm))

    V_trans = Y_fine.sub(0).collapse()
    u_bc_func = Function(V_trans, name="u_bc_func")
    u_bc_func.dat.data_wo[:] = u_fine_norm.dat.data_ro
    return u_bc_func


def save_checkpoint(h5_path: Path, mesh, functions: dict[str, Function]):
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with fd.CheckpointFile(str(h5_path), "w") as chk:
        chk.save_mesh(mesh)
        for name, fn in functions.items():
            chk.save_function(fn, name=name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["baseline", "baseline_verify", "rect_glen", "basin_glen", "basin_glen_noslip", "basin_newtonian"],
    )
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--coarse-max-it", type=int, default=MAX_IT)
    parser.add_argument("--fine-max-it", type=int, default=MAX_IT)
    parser.add_argument("--nitsche-gamma", type=float, default=100.0)
    parser.add_argument("--force", action="store_true", help="Re-run even if run_diagnostics.json or .in_progress exists")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    diag_file = args.out_dir / "run_diagnostics.json"
    lock_file = args.out_dir / ".in_progress"
    skip = False
    if rank0() and not args.force:
        if diag_file.is_file() or lock_file.is_file():
            skip = True
        else:
            lock_file.write_text(f"started {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n", encoding="utf-8")
    skip = fd.COMM_WORLD.bcast(skip, root=0)
    if skip:
        log(f"[{args.experiment}] Skipping (already completed or in progress in {args.out_dir}; pass --force to override).")
        return

    try:
        _run_experiment(args)
    finally:
        if rank0() and lock_file.is_file():
            lock_file.unlink(missing_ok=True)


def _run_experiment(args):
    inp = args.inputs_dir

    # Always build the 80 m fine mesh identically across all experiments
    fine_dem = load_resampled_raster_dem(
        inp / "main_boreholes_aoi_fix_bed.tif",
        inp / "main_boreholes_aoi_fix_surf.tif",
        upscale_factor=0.1,
    )
    mesh_fine, unit_fine, _, base_bed_f, base_surf_f = build_rect_extruded_mesh(fine_dem, layers=N_LAYERS, name="fine_80m")
    V_pf = FunctionSpace(mesh_fine, "CG", 1)
    V_uf = VectorFunctionSpace(mesh_fine, "CG", 3)
    Y_fine = V_uf * V_pf

    # Build coarse mesh and solve coarse problem according to experiment
    exp_name = args.experiment
    if exp_name in ("baseline", "baseline_verify", "rect_glen"):
        coarse_dem = load_resampled_raster_dem(
            inp / "bedmachine_ian_merged_bed_large_aoi_8m_aligned_fixed.tif",
            inp / "rema_large_aoi_align_fixed.tif",
            upscale_factor=1.0 / 200.0,
        )
        mesh_coarse, unit_coarse, _, base_bed_c, base_surf_c = build_rect_extruded_mesh(
            coarse_dem, layers=N_LAYERS, name="coarse_rect_1600m"
        )
        coarse_side_ids = [1, 2, 3, 4]
        coarse_lat_mode = "noslip"
        coarse_glen = (exp_name == "rect_glen")
        use_ns_coarse = True  # exact match to model_runner.py
    else:
        mesh_coarse, unit_coarse, _, base_bed_c, base_surf_c = build_basin_extruded_mesh(
            inp / "allan_hills_basin_2d.msh",
            inp / "allan_hills_basin_dems.npz",
            layers=N_LAYERS,
            name="coarse_basin",
        )
        coarse_side_ids = [10]
        coarse_lat_mode = "noslip" if exp_name == "basin_glen_noslip" else "freeslip"
        coarse_glen = (exp_name != "basin_newtonian")
        use_ns_coarse = (coarse_lat_mode == "noslip")

    flow_coarse, T_coarse, _, _, diag_coarse = run_thermomechanical_solve(
        mesh_coarse,
        lateral_mode=coarse_lat_mode,
        side_ids=coarse_side_ids,
        glen=coarse_glen,
        max_it=args.coarse_max_it,
        use_nullspace=use_ns_coarse,
        nitsche_gamma=args.nitsche_gamma,
        icntl14=120,
        label=f"{exp_name}:coarse",
    )

    save_checkpoint(
        args.out_dir / "coarse_solution.h5",
        mesh_coarse,
        {"coarse_flow": flow_coarse, "coarse_temp": T_coarse},
    )

    log(f"[{exp_name}] Transferring coarse velocity to fine-mesh side boundaries...")
    u_bc_func = transfer_coarse_velocity_to_fine_bc(flow_coarse, unit_coarse, unit_fine, mesh_fine, Y_fine)

    flow_fine, T_fine, _, _, diag_fine = run_thermomechanical_solve(
        mesh_fine,
        lateral_mode="dirichlet",
        side_ids=[1, 2, 3, 4],
        u_bc_func=u_bc_func,
        glen=True,
        max_it=args.fine_max_it,
        use_nullspace=True,  # exact match to model_runner.py
        icntl14=140,
        label=f"{exp_name}:fine",
    )

    save_checkpoint(
        args.out_dir / "fine_solution.h5",
        mesh_fine,
        {"fine_flow": flow_fine, "fine_temp": T_fine, "u_bc_func": u_bc_func},
    )

    summary = {
        "experiment": exp_name,
        "mpi_ranks": int(fd.COMM_WORLD.size),
        "coarse": diag_coarse,
        "fine": diag_fine,
    }
    if rank0():
        (args.out_dir / "run_diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        log(f"[{exp_name}] All checkpoints and diagnostics saved to {args.out_dir}")


if __name__ == "__main__":
    main()
