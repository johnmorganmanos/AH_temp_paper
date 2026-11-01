#!/usr/bin/env python3
"""Export the compact 800 m regional geometry used by the Comment 1 notebook."""

from __future__ import annotations

import argparse
import copy
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import pandas as pd


BED_SHA256 = "25bef320cf80c79f634bd8b5927a12ad5943eb3aebce7679920438bd7790a0a6"
SURFACE_SHA256 = "6ed62ca01b80339922a90e9963bf6dc4b992785512075c9908c683474ab13ca4"
RESAMPLE_FACTOR = 0.01
EXPECTED_SHAPE = (26, 38)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_input(path: Path, expected_sha256: str) -> Path:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {observed} != {expected_sha256}")
    return path


def export_geometry(bed_path: Path, surface_path: Path, output_path: Path) -> None:
    # Netgen is optional; its x86 binary cannot load under Apple emulation.
    sys.modules["netgen"] = None
    import firedrake as fd
    import rasterio
    from rasterio.enums import Resampling
    from scipy.interpolate import griddata

    with rasterio.open(bed_path) as bed_source, rasterio.open(surface_path) as surface_source:
        bed_height = int(bed_source.height * RESAMPLE_FACTOR)
        bed_width = int(bed_source.width * RESAMPLE_FACTOR)
        surface_height = int(surface_source.height * RESAMPLE_FACTOR)
        surface_width = int(surface_source.width * RESAMPLE_FACTOR)
        if (bed_height, bed_width) != (surface_height, surface_width):
            raise RuntimeError("The resampled bed and surface grids differ in shape.")
        bed_dem = bed_source.read(
            1,
            out_shape=(bed_source.count, bed_height, bed_width),
            resampling=Resampling.bilinear,
        )
        surface_dem = surface_source.read(
            1,
            out_shape=(surface_source.count, surface_height, surface_width),
            resampling=Resampling.bilinear,
        )
        bed_transform = bed_source.transform * bed_source.transform.scale(
            bed_source.width / bed_dem.shape[-1], bed_source.height / bed_dem.shape[-2]
        )
        surface_transform = surface_source.transform * surface_source.transform.scale(
            surface_source.width / surface_dem.shape[-1],
            surface_source.height / surface_dem.shape[-2],
        )
    if bed_transform != surface_transform:
        raise RuntimeError("Resampled bed and surface transforms differ.")

    bed_dem = np.asarray(bed_dem, dtype=float).squeeze()
    surface_dem = np.asarray(surface_dem, dtype=float).squeeze()
    if bed_dem.shape != EXPECTED_SHAPE or surface_dem.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"Expected resampled shape {EXPECTED_SHAPE}; found {bed_dem.shape} and {surface_dem.shape}."
        )
    if not np.isfinite(bed_dem).all() or not np.isfinite(surface_dem).all():
        raise RuntimeError("Resampled DEM arrays contain non-finite values.")

    # Preserve the published mesh-generation correction exactly.
    negative_thickness = surface_dem - bed_dem
    negative_thickness[negative_thickness > 0] = 0
    zero_offset = copy.deepcopy(negative_thickness)
    zero_offset[zero_offset < 0] = -1
    bed_dem = bed_dem + negative_thickness + zero_offset

    height, width = surface_dem.shape
    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(surface_transform, rows, columns)
    xs = np.asarray(xs, dtype=float).reshape(-1)
    ys = np.asarray(ys, dtype=float).reshape(-1)
    raster_points = np.column_stack([xs, ys])

    base_mesh = fd.RectangleMesh(
        width - 1,
        height - 1,
        float(xs.max()),
        float(ys.max()),
        originX=float(xs.min()),
        originY=float(ys.min()),
    )
    base_coordinates = np.asarray(base_mesh.coordinates.dat.data_ro[:, :2], dtype=float)
    surface_values = griddata(
        raster_points,
        surface_dem.reshape(-1),
        base_coordinates,
        method="nearest",
        fill_value=np.nan,
        rescale=False,
    )
    bed_values = griddata(
        raster_points,
        bed_dem.reshape(-1),
        base_coordinates,
        method="nearest",
        fill_value=np.nan,
        rescale=False,
    )
    if not np.isfinite(bed_values).all() or not np.isfinite(surface_values).all():
        raise RuntimeError("DEM-to-mesh interpolation produced non-finite elevations.")

    geometry = pd.DataFrame(
        {
            "x_m": base_coordinates[:, 0],
            "y_m": base_coordinates[:, 1],
            "bed_elevation_m": bed_values,
            "surface_elevation_m": surface_values,
        }
    ).sort_values(["x_m", "y_m"], kind="stable")
    if len(geometry) != EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1]:
        raise RuntimeError(f"Expected 988 base nodes; found {len(geometry)}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geometry.to_csv(output_path, index=False, float_format="%.17g", lineterminator="\n")
    print(f"Wrote {output_path} ({len(geometry)} rows)")
    print(f"SHA-256 {file_sha256(output_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bed",
        type=Path,
        default=Path(
            "/root/AH_temp_paper/Meshes/ThreeD_meshing/"
            "bedmachine_ian_merged_bed_large_aoi_8m_aligned_fixed.tif"
        ),
    )
    parser.add_argument(
        "--surface",
        type=Path,
        default=Path(
            "/root/AH_temp_paper/Meshes/ThreeD_meshing/rema_large_aoi_align_fixed.tif"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "reviewer1_comment1_inputs/regional_800m_geometry.csv",
    )
    args = parser.parse_args()
    export_geometry(
        locked_input(args.bed, BED_SHA256),
        locked_input(args.surface, SURFACE_SHA256),
        args.output,
    )


if __name__ == "__main__":
    main()
