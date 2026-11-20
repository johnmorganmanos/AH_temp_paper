#!/usr/bin/env bash
# Master runner for all Allan Hills coarse-solve reevaluation experiments on sermeq.
# Designed to run inside a tmux session (e.g., tmux new-session -d -s ah-runs "bash ...").
set -euo pipefail

BASE=/data/fast1/bpl7/ah-coarse-quest
export PETSC_DIR="$BASE/petsc"
export PETSC_ARCH=arch-firedrake-default
export PATH="$PETSC_DIR/$PETSC_ARCH/bin:$PATH"
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

source "$BASE/venv/bin/activate"

REPO="$BASE/repo"
SCRIPTS="$REPO/coarse_solve_reevaluation"
INPUTS="$BASE/inputs"
OUT="$BASE/outputs"
RESULTS="$OUT/results"
FIGS="$OUT/figures"
MPI_RANKS="${MPI_RANKS:-8}"

mkdir -p "$RESULTS" "$FIGS"

echo "================================================================"
echo "Allan Hills Coarse-Solve Reevaluation Suite (MPI_RANKS=$MPI_RANKS)"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

# Ensure xarray is available for reading Borehole_Temps/*.pkl
python -c "import xarray" 2>/dev/null || pip install -q xarray

run_one() {
  local exp="$1"
  local tag="$2"
  echo "================================================================"
  echo "=== Running experiment: $exp ($tag) ==="
  echo "================================================================"
  mpiexec -n "$MPI_RANKS" python -u "$SCRIPTS/run_coarse_and_fine_experiments.py" \
    --experiment "$exp" \
    --inputs-dir "$INPUTS" \
    --out-dir "$OUT/$exp"

  echo "=== Evaluating & regenerating ALL figures for: $exp ==="
  python -u "$SCRIPTS/evaluate_and_plot_experiment.py" \
    --experiment "$exp" \
    --title-tag "$tag" \
    --exp-dir "$OUT/$exp" \
    --inputs-dir "$INPUTS" \
    --repo-dir "$REPO" \
    --fig-dir "$FIGS/$exp" \
    --metrics-out "$RESULTS/${exp}_metrics.json"

  # Update incremental comparison report after each finished experiment
  python -u "$SCRIPTS/make_comparison_report.py" \
    --results-dir "$RESULTS" \
    --cmp-fig-dir "$FIGS/comparisons"
}

run_one "baseline" "Baseline (Rect. Newtonian)"
run_one "rect_glen" "Exp 1: Rect. Glen's Law"
run_one "basin_glen" "Exp 2: REMA Basin Glen (Free-slip)"
run_one "basin_glen_noslip" "Exp 2b: REMA Basin Glen (No-slip)"
run_one "basin_newtonian" "Diag: REMA Basin Newtonian (Free-slip)"

echo "================================================================"
echo "ALL EXPERIMENTS AND FIGURES COMPLETE: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"
