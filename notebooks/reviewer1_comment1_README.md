# Reviewer 1, Comment 1: observed-versus-modeled surface mass balance

This directory contains the executed diagnostic notebook and draft
supplementary cross plot developed in response to Reviewer 1, Comment 1.

## Contents

- `reviewer1_comment1_surface_mass_balance.ipynb`: fully executed analysis,
  equations, quality checks, interpretation, and inline figure.
- `reviewer1_comment1_model.py`: minimal Firedrake mesh reconstruction helper.
- `export_reviewer1_comment1_geometry.py`: provenance-preserving exporter for
  the compact regional geometry snapshot.
- `run_reviewer1_comment1_docker.sh`: one-command execution in the recovered
  Allan Hills Firedrake container.
- `reviewer1_comment1_inputs/`: locked Spaulding observations and compact model
  geometry, with source documentation.
- `reviewer1_comment1_outputs/`: the PDF/PNG cross plot, site-level values, and
  machine-readable summary.

The notebook compares observed Spaulding surface balance with

\[
\dot a_{\mathrm{modeled}}=u_s h_x+v_s h_y-w_s,
\]

evaluated from the saved 800 m regional parent solution. It does not
extrapolate outside the model domain. The primary comparison contains 24 of
the 25 paper-retained ice/firn sites; M16 lies outside the regional domain.

## Reproduction

The tracked `Saved_Models/coarse_mesh_flow.h5` checkpoint and the two compact
CSV inputs are sufficient to reproduce the calculation in a compatible
single-rank Firedrake environment. On the machine where the recovered model
container is installed, run:

```bash
notebooks/run_reviewer1_comment1_docker.sh
```

The runner expects an already restored compatible container and defaults to
`allanhills_v1`. The locally verified container uses image tag
`allanhills-transfer:20261026-2e078792`. Set `ALLAN_HILLS_CONTAINER` to use
another compatible container name. The notebook verifies SHA-256 digests
before loading any model or observational input.

To regenerate the compact geometry snapshot from the original recovered DEMs
before executing the notebook, run:

```bash
notebooks/run_reviewer1_comment1_docker.sh --refresh-geometry
```

The raw DEMs are not added to Git: together they occupy approximately 123 MB.
The geometry snapshot instead records the exact bed and surface values at all
988 horizontal nodes used by the regional mesh.
