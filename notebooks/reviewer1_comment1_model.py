"""Reconstruct the 800 m regional mesh used by the Comment 1 diagnostic.

The checked-in geometry snapshot contains the bed and surface elevations at
the 26 x 38 horizontal nodes of the published regional model.  Keeping this
small snapshot with the analysis avoids depending on two large source DEMs
that are present in the recovered model environment but not in the public
repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MESH_LAYERS = 5
MESH_POINT_TOLERANCE = 1.0e-10
GEOMETRY_COLUMNS = (
    "x_m",
    "y_m",
    "bed_elevation_m",
    "surface_elevation_m",
)


def load_geometry_snapshot(path: Path) -> pd.DataFrame:
    """Load and validate the compact regional geometry snapshot."""

    frame = pd.read_csv(path)
    if tuple(frame.columns) != GEOMETRY_COLUMNS:
        raise ValueError(
            f"Geometry columns must be {GEOMETRY_COLUMNS}; found {tuple(frame.columns)}."
        )
    numeric = frame.loc[:, GEOMETRY_COLUMNS].to_numpy(dtype=float)
    if numeric.ndim != 2 or not np.isfinite(numeric).all():
        raise ValueError("Geometry values must be a finite two-dimensional array.")
    if frame.duplicated(["x_m", "y_m"]).any():
        raise ValueError("Geometry contains duplicate horizontal coordinates.")
    x_values = np.sort(frame["x_m"].unique())
    y_values = np.sort(frame["y_m"].unique())
    if len(frame) != len(x_values) * len(y_values):
        raise ValueError("Geometry does not form a complete rectangular grid.")
    if (frame["surface_elevation_m"] <= frame["bed_elevation_m"]).any():
        raise ValueError("Every regional mesh column must have positive thickness.")
    return frame.sort_values(["x_m", "y_m"], kind="stable").reset_index(drop=True)


def mesh_columns_from_coordinates(coordinates: np.ndarray) -> pd.DataFrame:
    """Return one exact bed/surface pair for every horizontal mesh column."""

    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Mesh coordinates must be a finite N x 3 array.")
    frame = pd.DataFrame(values, columns=["x_m", "y_m", "z_m"])
    grouped = frame.groupby(["x_m", "y_m"], as_index=False, sort=True)["z_m"].agg(
        bed_elevation_m="min",
        surface_elevation_m="max",
        vertical_nodes="size",
    )
    if (grouped["surface_elevation_m"] <= grouped["bed_elevation_m"]).any():
        raise ValueError("At least one mesh column has nonpositive ice thickness.")
    if not (grouped["vertical_nodes"] == MESH_LAYERS + 1).all():
        counts = grouped["vertical_nodes"].value_counts().sort_index().to_dict()
        raise ValueError(
            f"Expected {MESH_LAYERS + 1} coordinate nodes per mesh column; found {counts}."
        )
    return grouped.sort_values(["x_m", "y_m"], kind="stable").reset_index(drop=True)


def build_mesh_from_geometry(path: Path) -> dict[str, Any]:
    """Reconstruct the terrain-following regional mesh from its node snapshot."""

    try:
        import firedrake as fd
        from firedrake.__future__ import interpolate as future_interpolate
    except ImportError as error:
        raise RuntimeError("A compatible Firedrake environment is required.") from error

    communicator_size = getattr(fd.COMM_WORLD, "size", None)
    if communicator_size is None:
        communicator_size = fd.COMM_WORLD.Get_size()
    if communicator_size != 1:
        raise RuntimeError("The legacy checkpoint was written in serial; use one MPI rank.")

    geometry = load_geometry_snapshot(path)
    x_values = np.sort(geometry["x_m"].unique())
    y_values = np.sort(geometry["y_m"].unique())
    width = len(x_values)
    height = len(y_values)

    base_mesh = fd.RectangleMesh(
        width - 1,
        height - 1,
        float(x_values.max()),
        float(y_values.max()),
        originX=float(x_values.min()),
        originY=float(y_values.min()),
    )
    base_coordinates = np.asarray(base_mesh.coordinates.dat.data_ro[:, :2], dtype=float)
    saved_coordinates = geometry[["x_m", "y_m"]].to_numpy(dtype=float)
    current_order = np.lexsort((base_coordinates[:, 1], base_coordinates[:, 0]))
    saved_order = np.lexsort((saved_coordinates[:, 1], saved_coordinates[:, 0]))
    if base_coordinates.shape != saved_coordinates.shape or not np.allclose(
        base_coordinates[current_order],
        saved_coordinates[saved_order],
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise RuntimeError("The geometry snapshot does not match the reconstructed base grid.")

    bed_values = np.empty(len(base_coordinates), dtype=float)
    surface_values = np.empty(len(base_coordinates), dtype=float)
    bed_values[current_order] = geometry["bed_elevation_m"].to_numpy(dtype=float)[saved_order]
    surface_values[current_order] = geometry["surface_elevation_m"].to_numpy(dtype=float)[saved_order]

    base_space = fd.FunctionSpace(base_mesh, "CG", 1)
    base_bed = fd.Function(base_space, name="reviewer1_regional_bed")
    base_surface = fd.Function(base_space, name="reviewer1_regional_surface")
    base_bed.dat.data_wo[:] = bed_values
    base_surface.dat.data_wo[:] = surface_values

    unit_extruded_mesh = fd.ExtrudedMesh(base_mesh, layers=MESH_LAYERS)
    vertical_constant = fd.FiniteElement("R", "interval", 0)
    extruded_space = fd.FunctionSpace(
        unit_extruded_mesh,
        fd.TensorProductElement(base_space.ufl_element(), vertical_constant),
    )
    extruded_bed = fd.Function(extruded_space)
    extruded_surface = fd.Function(extruded_space)
    extruded_bed.dat.data_wo[:] = bed_values
    extruded_surface.dat.data_wo[:] = surface_values
    x_coordinate, y_coordinate, z_coordinate = fd.SpatialCoordinate(unit_extruded_mesh)
    new_coordinates = fd.assemble(
        future_interpolate(
            fd.as_vector(
                [
                    x_coordinate,
                    y_coordinate,
                    (1 - z_coordinate) * extruded_bed + z_coordinate * extruded_surface,
                ]
            ),
            unit_extruded_mesh.coordinates.function_space(),
        )
    )
    mesh = fd.Mesh(new_coordinates)
    mesh.tolerance = MESH_POINT_TOLERANCE
    return {
        "firedrake": fd,
        "mesh": mesh,
        "base_bed": base_bed,
        "base_surface": base_surface,
        "resampled_shape": [height, width],
        "bed_range_m": [float(bed_values.min()), float(bed_values.max())],
        "surface_range_m": [float(surface_values.min()), float(surface_values.max())],
    }
