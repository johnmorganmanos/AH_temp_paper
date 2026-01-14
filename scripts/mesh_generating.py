import firedrake
import numpy as np
import pickle as pkl
from numpy import pi as π
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import firedrake
from firedrake import Constant, inner, sqrt, tr, grad, div, as_vector, exp,sym, as_vector, dx, ds, Mesh, Function, project, TransferManager
import meshpy, meshpy.geometry, meshpy.triangle
import irksome
from irksome import Dt
from scipy.signal import detrend
import copy
import matplotlib
import irksome
from irksome import Dt
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import pickle as pkl
import tqdm
import emcee
import corner
import itertools
import xarray
import dtscalibration
import glob
from rasterio.plot import show
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
import sys
from pyproj import Transformer
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from scripts.AH_temp_funcs import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import shapefile as sf

import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask


from firedrake import *
from firedrake.__future__ import interpolate

### Just make this a single function and make two calls to it for each mesh. ###

def mesh_creator(src_bed_path, src_surf_path, resamp_factor=1/200, layers=15):
    
    """
    Change the DEM (model) resolutions as you need below.
    """

    resolution_of_model = str(8 * (1/resamp_factor))


    # Load DEM

    src_bed = rasterio.open(src_bed_path)
    src_surf = rasterio.open(src_surf_path)

    bed_DEM = src_bed.read(1,
                            out_shape=(
                                src_bed.count,
                                int(src_bed.height * resamp_factor),
                                int(src_bed.width * resamp_factor)
                            ),
                            resampling=Resampling.bilinear
                        )

    # scale image transform
    src_bed_transform = src_bed.transform * src_bed.transform.scale(
        (src_bed.width / bed_DEM.shape[-1]),
        (src_bed.height / bed_DEM.shape[-2])
    )


    # src_surf = rasterio.open("../Meshes/ThreeD_meshing/rema_large_aoi_align_fixed.tif")

    surf_DEM = src_surf.read(1,
                            out_shape=(
                                src_surf.count,
                                int(src_surf.height * resamp_factor),
                                int(src_surf.width * resamp_factor)
                            ),
                            resampling=Resampling.bilinear
                        )

    # scale image transform
    src_surf_transform = src_surf.transform * src_surf.transform.scale(
        (src_surf.width / surf_DEM.shape[-1]),
        (src_surf.height / surf_DEM.shape[-2])
    )

    ### Check if the bed DEMs is higher than the surface DEM (thickness is negative) ###

    thickness = surf_DEM - bed_DEM
    thickness[thickness > 0] = 0  # if positive, set to 0. This will mask the negative thickness values
    zero_offset = copy.deepcopy(thickness)
    zero_offset[zero_offset < 0 ] = -1 # this mask will lower the bed a further 1m in the areas where thickness is negative so the model doesn't crash on a zero.
    bed_DEM = bed_DEM + thickness + zero_offset # add the negative values to the bed to push it lower to match the surface DEM

    height = surf_DEM.shape[0]
    width = surf_DEM.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src_surf_transform, rows, cols)
    lons= np.array(xs)
    lats = np.array(ys)

    surf_elevs = surf_DEM.flatten()
    bed_elevs = bed_DEM.flatten()

    nx = width - 1
    ny = height - 1

    Lx = lons.max() # x extent, get this from the DEM
    Ly = lats.max() # y extent, get this from the DEM

    origin_x = lons.min() # origin (lower left corner) in x, get this from the DEM
    origin_y = lats.min() # origin (lower left corner) in y, get this from the DEM

    base_mesh = RectangleMesh(nx,ny, Lx, Ly, originX=origin_x, originY=origin_y)
    # Make a height 1 mesh.
    unit_extruded_mesh = ExtrudedMesh(base_mesh, layers=layers)

    base_fs = FunctionSpace(base_mesh, "CG", 1)

    x,y, = SpatialCoordinate(base_mesh)

    # You could set this field any way you like.
    ## Bed and surface DEM resolution should be nx+1, ny+1 as to match the coordinates of the mesh.

    # Now we transfer the bathymetry field into a depth-averaged field.
    extruded_element = FiniteElement("R", "interval", 0)
    extruded_space = FunctionSpace(unit_extruded_mesh,
                                    TensorProductElement(base_fs.ufl_element(),
                                                        extruded_element))


    from scipy.interpolate import griddata
    ## interpolate the DEMs to the mesh coordinates so it is given in the mesh coordinate order
    points = np.array([xs,ys]).T # DEM pixel coordinates
    xi = base_mesh.coordinates.dat.data_ro[:,:2] # Get the mesh coordinates from the base mesh
    surf_interp = griddata(points, surf_DEM.flatten(), xi, method='nearest', fill_value=np.nan, rescale=False)
    bed_interp = griddata(points, bed_DEM.flatten(), xi, method='nearest', fill_value=np.nan, rescale=False)

    ## Extrude the bed and surface
    extruded_bed = Function(extruded_space)
    extruded_bed.dat.data_wo[:] = bed_interp

    extruded_surface = Function(extruded_space)
    extruded_surface.dat.data_wo[:] = surf_interp

    # Build a new coordinate field by change of coordinates.
    x, y, z = SpatialCoordinate(unit_extruded_mesh)
    new_coordinates = assemble(
        interpolate(
            as_vector([x, y, (1-z) * extruded_bed + z * extruded_surface]),
            unit_extruded_mesh.coordinates.function_space()
        )
    )


    # Finally build the mesh you are actually after.
    mesh_src = Mesh(new_coordinates) # Mesh(new_coordinates, tolerance=50)
    # x_src, y_src, z_src = SpatialCoordinate(mesh_src)

    return mesh_src