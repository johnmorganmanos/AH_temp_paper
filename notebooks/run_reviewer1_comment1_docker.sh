#!/usr/bin/env bash
# Execute the Comment 1 notebook in the recovered Allan Hills Firedrake container.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
container="${ALLAN_HILLS_CONTAINER:-allanhills_v1}"
refresh_geometry=false
if [[ "${1:-}" == "--refresh-geometry" ]]; then
  refresh_geometry=true
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--refresh-geometry]" >&2
  exit 2
fi

docker start "$container" >/dev/null
work="$(docker exec "$container" mktemp -d /tmp/reviewer1-comment1.XXXXXX | tr -d '\r')"
case "$work" in
  /tmp/reviewer1-comment1.*) ;;
  *) echo "Unexpected temporary path: $work" >&2; exit 1 ;;
esac
cleanup() {
  docker exec "$container" rm -rf -- "$work" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker exec "$container" mkdir -p \
  "$work/notebooks/reviewer1_comment1_inputs" \
  "$work/notebooks/reviewer1_comment1_outputs" \
  "$work/Saved_Models"

docker cp "$repo_dir/notebooks/reviewer1_comment1_surface_mass_balance.ipynb" \
  "$container:$work/notebooks/"
docker cp "$repo_dir/notebooks/reviewer1_comment1_model.py" \
  "$container:$work/notebooks/"
docker cp "$repo_dir/notebooks/export_reviewer1_comment1_geometry.py" \
  "$container:$work/notebooks/"
docker cp "$repo_dir/notebooks/reviewer1_comment1_inputs/spaulding_table_s4.csv" \
  "$container:$work/notebooks/reviewer1_comment1_inputs/"
docker cp "$repo_dir/Saved_Models/coarse_mesh_flow.h5" \
  "$container:$work/Saved_Models/"

if "$refresh_geometry"; then
  docker exec --workdir "$work" \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e PYOP2_CACHE_DIR=/tmp/reviewer1-comment1-pyop2 \
    -e PYOP2_CFLAGS='-march=x86-64 -mtune=generic' \
    -e PYOP2_CXXFLAGS='-march=x86-64 -mtune=generic' \
    "$container" python3 notebooks/export_reviewer1_comment1_geometry.py
  docker cp \
    "$container:$work/notebooks/reviewer1_comment1_inputs/regional_800m_geometry.csv" \
    "$repo_dir/notebooks/reviewer1_comment1_inputs/"
else
  docker cp "$repo_dir/notebooks/reviewer1_comment1_inputs/regional_800m_geometry.csv" \
    "$container:$work/notebooks/reviewer1_comment1_inputs/"
fi

docker exec --workdir "$work" \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e PYOP2_CACHE_DIR=/tmp/reviewer1-comment1-pyop2 \
  -e PYOP2_CFLAGS='-march=x86-64 -mtune=generic' \
  -e PYOP2_CXXFLAGS='-march=x86-64 -mtune=generic' \
  "$container" jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  notebooks/reviewer1_comment1_surface_mass_balance.ipynb

docker cp \
  "$container:$work/notebooks/reviewer1_comment1_surface_mass_balance.ipynb" \
  "$repo_dir/notebooks/"
mkdir -p "$repo_dir/notebooks/reviewer1_comment1_outputs"
for output in \
  figure_spaulding_observed_vs_modeled_smb.pdf \
  figure_spaulding_observed_vs_modeled_smb.png \
  spaulding_crossplot_summary.json \
  spaulding_observed_vs_modeled_sites.csv; do
  docker cp "$container:$work/notebooks/reviewer1_comment1_outputs/$output" \
    "$repo_dir/notebooks/reviewer1_comment1_outputs/"
done

echo "Executed notebook and copied the cross-plot products into the repository."
