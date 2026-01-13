import numpy as np
import copy  # <- can be removed if you use thickness.copy() instead (see below)

import rasterio
from rasterio.enums import Resampling

from firedrake import (
    RectangleMesh, ExtrudedMesh, Mesh,
    FunctionSpace, SpatialCoordinate, Function,
    FiniteElement, TensorProductElement,
    as_vector, assemble
)
from firedrake.__future__ import interpolate

from scipy.interpolate import griddata


def _make_mesh_from_dem(bed_path, surf_path, *, resamp_factor=1/200, layers=15):
    """
    Build a Firedrake 3D extruded mesh from bed + surface DEM rasters,
    resampling DEMs by resamp_factor and extruding into `layers`.
    """

    src_bed = rasterio.open(bed_path)
    src_surf = rasterio.open(surf_path)

    bed_DEM = src_bed.read(
        1,
        out_shape=(
            src_bed.count,
            int(src_bed.height * resamp_factor),
            int(src_bed.width * resamp_factor),
        ),
        resampling=Resampling.bilinear,
    )

    surf_DEM = src_surf.read(
        1,
        out_shape=(
            src_surf.count,
            int(src_surf.height * resamp_factor),
            int(src_surf.width * resamp_factor),
        ),
        resampling=Resampling.bilinear,
    )

    # scale image transform (use surface transform for x/y)
    src_surf_transform = src_surf.transform * src_surf.transform.scale(
        (src_surf.width / surf_DEM.shape[-1]),
        (src_surf.height / surf_DEM.shape[-2]),
    )

    # Ensure bed is not above surface (avoid negative thickness).
    thickness = surf_DEM - bed_DEM
    thickness[thickness > 0] = 0

    # copy not deepcopy is enough (or use thickness.copy())
    zero_offset = thickness.copy()
    zero_offset[zero_offset < 0] = -1
    bed_DEM = bed_DEM + thickness + zero_offset

    height, width = surf_DEM.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src_surf_transform, rows, cols)
    xs = np.array(xs)
    ys = np.array(ys)

    nx = width - 1
    ny = height - 1

    Lx = xs.max()
    Ly = ys.max()
    origin_x = xs.min()
    origin_y = ys.min()

    base_mesh = RectangleMesh(nx, ny, Lx, Ly, originX=origin_x, originY=origin_y)
    unit_extruded_mesh = ExtrudedMesh(base_mesh, layers=layers)

    base_fs = FunctionSpace(base_mesh, "CG", 1)

    extruded_element = FiniteElement("R", "interval", 0)
    extruded_space = FunctionSpace(
        unit_extruded_mesh,
        TensorProductElement(base_fs.ufl_element(), extruded_element),
    )

    # Interpolate DEM values onto mesh vertex coordinates (nearest neighbor).
    points = np.array([xs, ys]).T
    xi = base_mesh.coordinates.dat.data_ro[:, :2]

    surf_interp = griddata(
        points, surf_DEM.flatten(), xi,
        method="nearest", fill_value=np.nan, rescale=False
    )
    bed_interp = griddata(
        points, bed_DEM.flatten(), xi,
        method="nearest", fill_value=np.nan, rescale=False
    )

    extruded_bed = Function(extruded_space)
    extruded_bed.dat.data_wo[:] = bed_interp

    extruded_surface = Function(extruded_space)
    extruded_surface.dat.data_wo[:] = surf_interp

    # Build new coordinates via vertical interpolation between bed/surface.
    x, y, z = SpatialCoordinate(unit_extruded_mesh)
    new_coordinates = assemble(
        interpolate(
            as_vector([x, y, (1 - z) * extruded_bed + z * extruded_surface]),
            unit_extruded_mesh.coordinates.function_space(),
        )
    )

    return Mesh(new_coordinates)


def mesh_creator(
    fine_bed_path, fine_surf_path,
    coarse_bed_path, coarse_surf_path,
    coarse_resamp_factor=1/200,
    fine_resamp_factor=1/10,
    layers=15,
):
    # Loop over two calls to the same function (coarse then fine), as requested.
    return [
        _make_mesh_from_dem(
            coarse_bed_path, coarse_surf_path,
            resamp_factor=coarse_resamp_factor, layers=layers
        ),
        _make_mesh_from_dem(
            fine_bed_path, fine_surf_path,
            resamp_factor=fine_resamp_factor, layers=layers
        ),
    ]
