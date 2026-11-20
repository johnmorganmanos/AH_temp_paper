# Coarse-Region Reevaluation for the Allan Hills 3D Thermomechanical Ice-Flow Model

## Summary

We evaluate whether two approximations in the coarse outer-region solve of the Allan Hills Blue Ice Area (BIA) nested full-Stokes model alter any qualitative conclusion of the paper:

1. **Experiment 1 (`rect_glen`) — Coarse-region rheology:** Replacing the Newtonian outer solve ($\mu_0 = 5 \times 10^{15}\ \mathrm{Pa\,s}$) on the published 1600 m rectangular domain with a coupled thermomechanical Glen's-flow solve ($n = 3$, temperature-dependent $A(T)$, 21 Picard iterations), and prescribing the resulting velocity on the lateral walls of the 80 m fine mesh.
2. **Experiment 2 (`basin_glen`) — Modern-day REMA drainage-basin domain:** Replacing the rectangular coarse domain (whose walls cut across regional flowlines at a median angle of $42.0^\circ$) with an unstructured 2D–3D drainage-basin mesh delineated from REMA v2.0 500 m surface slopes ($\sigma = 1.5\ \mathrm{km}$, median boundary crossing angle $5.6^\circ$), solved with coupled Glen's flow and free-slip impermeable lateral walls ($\mathbf{u}\cdot\mathbf{n} = 0$, zero tangential traction via symmetric Nitsche's method), and coupled into the same 80 m fine mesh.
3. **Diagnostic sensitivities (`basin_glen_noslip`, `basin_newtonian`) & fresh baseline (`baseline`):** Isolating outer-wall boundary condition type (no-slip vs. free-slip on the REMA basin), domain geometry alone (Newtonian rheology on the REMA basin), and verifying exact solver reproduction of the published baseline in Firedrake 2026.10.0 / PETSc 3.26.0 / MUMPS 5.9.1.

### Central Verdict

**No qualitative conclusion of the paper changes under either modification or their combination.** Across all five experiments:

- The opposing downslope counterflow and near-bed stagnation zone upstream of the nunatak persist in the exact same location (`0 m` shift in the Fig. 3b minimum-speed node `(509233.1, -1353376.1)`).
- Near-bed speeds at `ALHIC1902` and `ALHIC2302` remain well below $2\ \mathrm{mm\,yr^{-1}}$ ($0.23\text{–}0.50\ \mathrm{mm\,yr^{-1}}$).
- The three-regime speed ordering (`Nunatak < Submerged Ridge < Cul-de-sac`) and the ~two-order-of-magnitude speed contrast between the Cul-de-sac (`ALHIC2301/2401`) and the stagnation zone (`ALHIC1902/2302`, $93\times\text{–}139\times$) are preserved.
- The inverse log-speed–age regression across the five non-Cul-de-sac cores is virtually invariant ($R^2 = 0.856\text{–}0.858$, $p = 0.024$, slope $-2.14$ to $-2.32\ \mathrm{Ma/decade}$), remains robust across $5, 10, 20, 50\ \mathrm{m}$ above bed ($R^2 = 0.85\text{–}0.87$), and yields extrapolated minimum-speed ages of $9.81\text{–}10.20\ \mathrm{Ma}$ (95% prediction intervals spanning $4.75\text{–}15.51\ \mathrm{Ma}$).
- Borehole thermal profiles and the no-advection MAE ordering (Supplementary Fig. 2) remain virtually identical ($\mathrm{MAE} = 0.040\text{–}0.041\ \mathrm{K}$ with advection; nunatak boreholes strictly lowest without advection).
- The coarse-mesh steady-geometry surface mass balance comparison (Supplementary Fig. 10) remains weakly correlated with Spaulding et al. (2012) stake observations ($r = 0.06\text{–}0.36$).

### Quantitative Mechanism: Boundary-Layer Coupling vs. Interior Topographic Control

The only notable quantitative shift arises from **coarse-region rheology (`Newtonian -> Glen`)**, not from domain geometry or outer-wall boundary condition:

- Switching the coarse solve to Glen's flow ($E = 1$) increases the raw coarse-to-fine side-boundary RMS velocity by $3.7\times$ on the rectangle (`73.1 -> 268.1 mm/yr`) and $5.7\times$ on the REMA drainage basin (`73.1 -> 417.6 mm/yr`; south wall `76 -> 738 mm/yr`, east wall `81 -> 334 mm/yr`).
- Consistent with Kamb's (1986) stress-coupling decay scale over a frozen no-slip bed ($H \approx 100\text{–}300\ \mathrm{m}$), this lateral perturbation decays rapidly inward (see [compare_boundary_coupling_decay.png](figures/comparisons/compare_boundary_coupling_decay.png)). Five GNSS stakes located $39\text{–}839\ \mathrm{m}$ from the south/east fine-mesh boundaries (`032`, `Track 5-2`, `031`, `036`, `034`) increase in raw ($E=1$) surface speed by $1.7\times\text{–}2.8\times$ and already match observed GNSS speeds near $E \approx 1$. In contrast, interior Cul-de-sac stakes ($911\text{–}1503\ \mathrm{m}$ from the edge) change by $< 0.3\%$, and raw ($E=1$) near-bed borehole speeds change by only $0.1\%$ in the Cul-de-sac (`ALHIC2301`: $24.69 \to 24.72\ \mathrm{mm\,yr^{-1}}$) and $+5\%$ in the stagnation zone (`ALHIC1902`: $0.252 \to 0.266\ \mathrm{mm\,yr^{-1}}$).
- Because the single global enhancement factor $E_{\mathrm{opt}}$ is calibrated against all 18 fine-mesh GNSS stakes, the faster near-boundary stakes pull $E_{\mathrm{opt}}$ down from **$1.97\ [1.64, 2.29]$** (`baseline`) to **$1.13\ [0.87, 1.39]$** (`rect_glen`) and **$1.14\ [0.88, 1.40]$** (`basin_glen`)—indistinguishable from $E = 1$—while slightly raising global GNSS horizontal RMSE from $0.0385$ to $0.0514\ \mathrm{m\,yr^{-1}}$ due to tension between near-edge stakes ($E \approx 1$) and interior stakes ($E \approx 2$).
- Uniform scaling by the lower $E_{\mathrm{opt}}$ reduces calibrated interior borehole speeds by $\sim 35\text{–}42\%$ relative to baseline without altering spatial ratios, flow topology, regression slope, $R^2$, or extrapolated age.
- Changing only the coarse domain geometry (`baseline` vs. `basin_newtonian`) shifts $E_{\mathrm{opt}}$ by $+0.6\%$ (`1.967 -> 1.979`) and interior stagnation/CDS speeds by $< 1\%$. Changing free-slip vs. no-slip on the REMA basin perimeter (`basin_glen` vs. `basin_glen_noslip`) alters all metrics by $< 0.4\%$.

## Quantitative Comparison Table

| Experiment | $E_{\mathrm{opt}}$ [95% CI] | GNSS RMSE$_h$ ($\mathrm{m\,yr^{-1}}$) | Side BC raw RMS ($\mathrm{mm\,yr^{-1}}$) | `ALHIC1902` $U_{10}$ | `ALHIC2302` $U_{10}$ | `ALHIC1502` $U_{10}$ | `ALHIC1901` $U_{10}$ | `ALHIC2201` $U_{10}$ | `ALHIC2301` $U_{10}$ | `ALHIC2401` $U_{10}$ | 10 m OLS $R^2$ ($p$) | Slope ($\mathrm{Ma/dec}$) | Fig. 3b extrap. age (Ma) | Min shift (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` (Rect. Newtonian) | 1.97 [1.64, 2.29] | 0.0385 | 73.1 | 0.496 | 0.376 | 9.47 | 12.93 | 15.13 | 48.55 | 52.12 | 0.856 (0.024) | −2.24 | 9.81 [4.75, 14.88] | 0.0 |
| `rect_glen` (Exp 1) | 1.13 [0.87, 1.39] | 0.0513 | 268.1 | 0.300 | 0.227 | 6.87 | 9.39 | 10.57 | 27.94 | 29.99 | 0.858 (0.024) | −2.14 | 10.02 [4.86, 15.19] | 0.0 |
| `basin_glen` (Exp 2, Free-slip) | 1.14 [0.88, 1.40] | 0.0514 | 417.6 | 0.302 | 0.229 | 6.17 | 8.49 | 9.55 | 28.09 | 30.16 | 0.856 (0.024) | −2.21 | 10.20 [4.88, 15.51] | 0.0 |
| `basin_glen_noslip` (Exp 2b) | 1.14 [0.88, 1.40] | 0.0515 | 416.8 | 0.303 | 0.229 | 6.18 | 8.51 | 9.57 | 28.21 | 30.28 | 0.856 (0.024) | −2.21 | 10.19 [4.88, 15.51] | 0.0 |
| `basin_newtonian` (Diag) | 1.98 [1.65, 2.31] | 0.0395 | 83.4 | 0.501 | 0.379 | 8.57 | 11.67 | 13.80 | 48.87 | 52.45 | 0.856 (0.024) | −2.32 | 9.97 [4.79, 15.14] | 0.0 |

*(All $U_{10}$ values are in $\mathrm{mm\,yr^{-1}}$ at calibrated $E_{\mathrm{opt}}$. Full 40-column metrics are in [results/comparison_summary_table.csv](results/comparison_summary_table.csv) and [results/comparison_summary.md](results/comparison_summary.md).)*

## REMA v2.0 Drainage-Basin Delineation

The modern-day coarse drainage basin ([delineate_rema_basin.py](delineate_rema_basin.py)) was derived from the PGC REMA v2.0 500 m filled mosaic (`rema_mosaic_500m_v2.0_filled_cop30_dem.tif`), smoothed with a Gaussian kernel of $\sigma = 1.5\ \mathrm{km}$ (selected by minimizing regional angular misfit against the 36 regional Allan Hills GNSS stakes: median agreement $9.9^\circ$). Backward steepest-descent trajectory integration from the 80 m fine domain and surrounding blue-ice accumulation zone defines a smooth $591.6\ \mathrm{km}^2$ polygon (vs. $574.4\ \mathrm{km}^2$ for the original rectangle) that:

- Encloses the entire 80 m fine mesh (`min_dist = 1,486 m` on the western Allan Hills bedrock ridge, $4\text{–}7.5\ \mathrm{km}$ on the ice-covered sides) and all 54 regional + fine-mesh GNSS stakes.
- Reduces median normalized cross-boundary slope flow $|\hat{\mathbf{u}}_{\mathrm{slope}} \cdot \mathbf{n}|$ from **$0.67$ ($42.0^\circ$ median crossing angle; $88\%$ of perimeter crossing flowlines)** on the original rectangle to **$0.10$ ($5.6^\circ$; $75.5\%$ tangent within $|\hat{\mathbf{u}}\cdot\mathbf{n}| \le 0.2$)** on the drainage basin (see [basin_geometry/basin_delineation_diagnostics.png](basin_geometry/basin_delineation_diagnostics.png)).
- Meshes cleanly into 1,925 constrained Delaunay triangles (1,017 nodes, graded $800\text{–}1200\ \mathrm{m}$, extruded to 5 terrain-following layers: `429,654` `CG3/CG1` Stokes DOFs) in [basin_geometry/allan_hills_basin_2d.msh](basin_geometry/allan_hills_basin_2d.msh), using exact published BedMachine + radar DEMs inside the original coarse domain seamlessly cosine-feathered into REMA 500 m / Bedmap3 500 m outside.
- Full-basin modeled surface speeds (opaque log-scale heatmap over LIMA) and streamlines are shown across the entire REMA drainage basin in [figures/basin_glen/C2_FigS06.png](figures/basin_glen/C2_FigS06.png) (upgraded Supplementary Fig. 8) and side-by-side against the rectangular coarse solves in [figures/comparisons/compare_coarse_basin_velocities.png](figures/comparisons/compare_coarse_basin_velocities.png).

## Numerical Finding: Thermal Sidewall Boundary Condition

During coupled Glen's-flow coarse-to-fine testing, we identified a subtle weak-form discrepancy in the legacy advection-diffusion equation (`scripts/AH_temp_funcs.py` line 389 and `scripts/model_runner.py` line 501). That code integrated the advection term by parts as $-\rho c T\,(\mathbf{u}\cdot\nabla\phi)$ without adding back the lateral boundary flux $+\int_{\partial\Omega_{\mathrm{side}}} \rho c T\,(\mathbf{u}\cdot\mathbf{n})\,\phi\,\mathrm{d}S$. That omission replaces the manuscript's stated zero-normal-gradient thermal sidewall condition ($\nabla T\cdot\mathbf{n} = 0$) with a spurious Robin condition $k\nabla T\cdot\mathbf{n} - \rho c T(\mathbf{u}\cdot\mathbf{n}) = 0$ on inflow walls ($\mathbf{u}\cdot\mathbf{n} < 0$).

- Under the slow Newtonian coarse baseline, this artifact was confined to $0.83\%$ of nodes within $323\ \mathrm{m}$ of the fine-mesh outer edge ($T \in [-45.8, -5.2]^\circ\mathrm{C}$, negligible at all interior boreholes).
- Under the $3.7\text{–}5.7\times$ faster Glen's-flow coarse inflow (`rect_glen` and `basin_glen`), the missing boundary flux destabilized the initial temperature solve ($T$ dropped to $-2217^\circ\mathrm{C}$).
- Replacing the advection term with the standard strong-form weak term $+\rho c\,(\mathbf{u}\cdot\nabla T)\,\phi$ genuinely enforces $\nabla T\cdot\mathbf{n} = 0$ on lateral walls, restores physical temperatures everywhere ($T \in [-33.8, -14.0]^\circ\mathrm{C}$, agreeing with SUPG stabilization to $0.31\%$ $L^2$), and converges cleanly in all Glen-coarse runs (`du/u < 1e-4`).

## Repository Contents

```text
coarse_solve_reevaluation/
├── README.md                              # This summary report
├── delineate_rema_basin.py                # REMA v2.0 surface-slope basin delineation & Gmsh MSH v2.2 builder
├── run_coarse_and_fine_experiments.py     # MPI-parallel Firedrake 3D thermomechanical full-Stokes runner
├── evaluate_and_plot_experiment.py        # GNSS calibration, transient T, age-speed/SMB metrics & all 12 figures
├── make_comparison_report.py              # Cross-experiment aggregator & comparison figure generator
├── run_all_on_sermeq.sh                   # End-to-end cluster runner script
├── basin_geometry/
│   ├── allan_hills_rema_basin.geojson     # REMA drainage-basin polygon (EPSG:3031)
│   ├── allan_hills_basin_2d.msh           # 2D Gmsh v2.2 triangulation (1,925 triangles)
│   ├── basin_delineation_metrics.json     # Flowline alignment & GNSS slope-smoothing diagnostics
│   └── basin_delineation_diagnostics.png  # 3-panel basin delineation & flowline alignment diagnostic
├── results/
│   ├── comparison_summary.md              # Auto-generated quantitative table + qualitative checklist
│   ├── comparison_summary_table.csv       # Full 40-column cross-experiment metrics table
│   └── <exp>_{metrics.json, core_samples.csv, gnss_stakes.csv, spaulding_smb.csv}
└── figures/
    ├── comparisons/
    │   ├── compare_coarse_basin_velocities.png # Full-basin opaque surface speed heatmap & streamlines (3-panel)
    │   ├── compare_speed10m_maps.png           # 10 m speed maps & log10(exp/base) ratio maps
    │   ├── compare_metrics_overview.png        # 4-panel borehole U10, OLS, GNSS E_opt, thermal MAE
    │   └── compare_boundary_coupling_decay.png # Wall inflow, GNSS inward decay & raw U10 ratios
    └── {baseline, rect_glen, basin_glen, basin_glen_noslip, basin_newtonian}/
        ├── C2_Fig01.png   # Main Fig. 1 (10 m flow & nunatak stagnation zoom)
        ├── C2_Fig02.png   # Main Fig. 2 (Stagnation transect A-A')
        ├── C2_Fig03.png   # Main Fig. 3 (Age vs U10 colored by d38Ar/36Ar + speed map)
        ├── C2_FigS01.png  # SI Fig. 1 (GNSS residuals & E RMSE curve)
        ├── C2_FigS02.png  # SI Fig. 2 (Borehole temperatures A-F, residuals G-L, no-adv MAE M)
        ├── C2_FigS03.png  # SI Fig. 3 (Age vs U10 colored by d18O)
        ├── C2_FigS05.png  # SI Fig. 5 (Submerged ridge transect B-B')
        ├── FigS08.png     # SI Fig. 6 (Cul-de-sac transect C-C')
        ├── C2_FigS07.png  # SI Fig. 7 (Fine-mesh viscosity convergence)
        ├── C2_FigS06.png  # SI Fig. 8 (Full-basin opaque surface speed heatmap, streamlines & fine domain)
        ├── C2_FigS09.png  # SI Fig. 9 (4-height 5/10/20/50 m sensitivity)
        └── C2_FigS10.png  # SI Fig. 10 (Spaulding observed vs modeled steady-geometry SMB)
```

*(Large binary checkpoints `.h5`, source DEM rasters `.tif`/`.npz`, and unpublished geochemistry Excel spreadsheets are excluded from git via `.gitignore`.)*

## Reproduction

All five 3D full-Stokes experiments (`1,982,394` fine-mesh `CG3/CG1` DOFs, 21 coupled Picard updates each, ~100 GiB peak RAM on 12 MPI ranks) were executed on `sermeq` with Firedrake 2026.10.0, PETSc 3.26.0, and MUMPS 5.9.1:

```bash
# 1. Delineate REMA drainage basin & build 2D mesh (requires REMA v2.0 500 m + Bedmap3 + local DEMs)
python delineate_rema_basin.py --inputs-dir <INPUTS_DIR> --out-dir basin_geometry

# 2. Run one coarse-to-fine coupled experiment in MPI
OMP_NUM_THREADS=1 mpiexec -n 12 python run_coarse_and_fine_experiments.py \
    --experiment basin_glen --inputs-dir <INPUTS_DIR> --out-dir <OUT_DIR>/basin_glen

# 3. Calibrate E_opt, extract all metrics, and regenerate all 12 paper & SI figures (serial)
python evaluate_and_plot_experiment.py \
    --experiment basin_glen --title-tag "Exp 2: REMA Basin Glen (Free-slip)" \
    --exp-dir <OUT_DIR>/basin_glen --inputs-dir <INPUTS_DIR> --repo-dir .. \
    --fig-dir figures/basin_glen --metrics-out results/basin_glen_metrics.json

# 4. Aggregate all experiments and generate cross-experiment comparison figures
python make_comparison_report.py --results-dir results --cmp-fig-dir figures/comparisons --inputs-dir <INPUTS_DIR> --repo-dir ..
```
