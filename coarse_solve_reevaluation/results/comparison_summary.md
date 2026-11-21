# Coarse-Solve Reevaluation: Quantitative vs. Qualitative Comparison

This report evaluates the influence of two coarse-region modeling changes on all results and figures in the Allan Hills ice-flow paper:

1. **Experiment 1 (`rect_glen`)**: Replacing the Newtonian coarse solve ($\mu_0 = 5\times 10^{15}\,\mathrm{Pa\,s}$) with a coupled thermomechanical Glen's-flow ($n=3$, temperature-dependent $A(T)$) solve on the same 1600 m rectangular domain, coupled into the 80 m fine mesh.
2. **Experiment 2 (`basin_glen`)**: Replacing the rectangular coarse domain with a modern-day drainage basin delineated from REMA v2.0 surface slopes ($\sigma = 1.5\,\mathrm{km}$), solved with coupled Glen's flow and physically natural free-slip impermeable lateral walls ($\mathbf{u}\cdot\mathbf{n} = 0$), coupled into the 80 m fine mesh.
3. **Sensitivities (`basin_glen_noslip`, `basin_newtonian`)**: Testing no-slip lateral walls on the drainage basin and Newtonian rheology on the drainage basin to separate domain geometry, lateral boundary condition, and rheology effects.

## Key Metrics Comparison

| experiment | E_opt | gnss_horiz_rmse_m_yr | gnss_mean_ang_err_deg | U10_ALHIC1902_mm_yr | U10_ALHIC2302_mm_yr | U10_ALHIC1502_mm_yr | U10_ALHIC1901_mm_yr | U10_ALHIC2201_mm_yr | U10_ALHIC2301_mm_yr | U10_ALHIC2401_mm_yr | ols10_r2 | ols10_slope_ma_per_dec | ols10_p | win10_extrap_age_ma | win10_min_disp_from_base_m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1.967 | 0.039 | 11.148 | 0.496 | 0.376 | 9.475 | 12.926 | 15.128 | 48.554 | 52.121 | 0.856 | -2.244 | 0.024 | 9.813 | 0.000 |
| rect_glen | 1.131 | 0.051 | 11.622 | 0.300 | 0.227 | 6.870 | 9.392 | 10.572 | 27.935 | 29.986 | 0.858 | -2.139 | 0.024 | 10.022 | 0.000 |
| basin_glen | 1.137 | 0.051 | 15.258 | 0.302 | 0.229 | 6.165 | 8.485 | 9.545 | 28.094 | 30.156 | 0.856 | -2.205 | 0.024 | 10.197 | 0.000 |
| basin_glen_noslip | 1.141 | 0.051 | 15.266 | 0.303 | 0.229 | 6.181 | 8.507 | 9.572 | 28.205 | 30.276 | 0.856 | -2.206 | 0.024 | 10.193 | 0.000 |
| basin_newtonian | 1.979 | 0.039 | 13.102 | 0.501 | 0.379 | 8.568 | 11.673 | 13.799 | 48.865 | 52.454 | 0.856 | -2.316 | 0.024 | 9.967 | 0.000 |

## Qualitative Robustness Checklist

| Paper qualitative finding | baseline | rect_glen | basin_glen | basin_glen_noslip | basin_newtonian |
|---|---|---|---|---|---|
| Opposing nunatak counterflow & stagnation zone (Fig. 1b, Fig. 2) | YES | YES | YES | YES | YES |
| Slow-flow (<2 mm/yr) at ALHIC1902 & ALHIC2302 | YES | YES | YES | YES | YES |
| Strict speed ordering: Nunatak < Submerged Ridge < Cul-de-sac | YES | YES | YES | YES | YES |
| Strong inverse age–log10(U10) fit (5 non-CDS cores, R² > 0.80, p < 0.05) | YES (R²=0.86, p=0.024) | YES (R²=0.86, p=0.024) | YES (R²=0.86, p=0.024) | YES (R²=0.86, p=0.024) | YES (R²=0.86, p=0.024) |
| CDS speed order-of-magnitude faster than stagnation (>30x) | YES (98x / 139x) | YES (93x / 132x) | YES (93x / 132x) | YES (93x / 132x) | YES (98x / 138x) |
| Height robustness across 5, 10, 20, 50 m (all R² > 0.80) | YES (0.86, 0.86, 0.86, 0.87) | YES (0.86, 0.86, 0.86, 0.87) | YES (0.86, 0.86, 0.85, 0.86) | YES (0.86, 0.86, 0.85, 0.86) | YES (0.86, 0.86, 0.85, 0.86) |
| Borehole temperature advection signal (Nunatak no-adv MAE < Ridge & CDS) | YES | YES | YES | YES | YES |
| Weak coarse-mesh steady-SMB association (Fig. S10) | r=0.25 (n=24) | r=0.36 (n=24) | r=0.34 (n=15) | r=0.34 (n=15) | r=0.06 (n=15) |
