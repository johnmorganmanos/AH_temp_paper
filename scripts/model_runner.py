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


"""
This script runs the ice flow model. 

We create a coarse mesh then solve for the flow. Then, we create a fine mesh and
use the flow from the coarse mesh as the boundary conditions for the fine mesh.

"""

"""
Change the DEM (model) resolutions as you need below.
"""

upscale_factor_coarse = 1/200
upscale_factor_fine = 1/10
e_param=1.97


resolution_of_model = 8 * (1/upscale_factor_fine)
resolution_of_model = str(resolution_of_model)
"""Load the DEMs"""
# Fine DEM

src_bed_fine = rasterio.open("../Meshes/crop_aoi/main_boreholes_aoi/main_boreholes_aoi_fix_bed.tif")
src_surf_fine = rasterio.open("../Meshes/crop_aoi/main_boreholes_aoi/main_boreholes_aoi_fix_surf.tif")

# Course DEM

src_bed_coarse = rasterio.open("../Meshes/ThreeD_meshing/bedmachine_ian_merged_bed_large_aoi_8m_aligned_fixed.tif")
src_surf_coarse = rasterio.open("../Meshes/ThreeD_meshing/rema_large_aoi_align_fixed.tif")

### Load the outline of the good dem

with rasterio.open('../Meshes/good_dem_outline/ian_bed_outline_raster_full.tif') as src_good_dem:
    good_dem_filter = src_good_dem.read(2)

# Load tiff files of the DEMs

src_bed = src_bed_coarse
src_surf = src_surf_coarse

bed_DEM = src_bed.read(1,
                        out_shape=(
                            src_bed.count,
                            int(src_bed.height * upscale_factor_coarse),
                            int(src_bed.width * upscale_factor_coarse)
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
                            int(src_surf.height * upscale_factor_coarse),
                            int(src_surf.width * upscale_factor_coarse)
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

layers = 5 # number of layers in z direction of the extruded mesh
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
x_src, y_src, z_src = SpatialCoordinate(mesh_src)


"""

Now, we make the firedrake model for the coarse mesh solve.

"""
### Flow model parameters and momentum equation ###
μ = Constant(5e15) #Ice


pressure_space = firedrake.FunctionSpace(mesh_src, "CG", 1)
velocity_space = firedrake.VectorFunctionSpace(mesh_src, "CG", 3)
Y = velocity_space * pressure_space
flow = firedrake.Function(Y)
u, p = firedrake.split(flow)
v, q = firedrake.TestFunctions(flow.function_space())

τ = 2 * μ * ε(u)

g = as_vector((0, 0, grav))
f =  ρ * g

F_momentum = (inner(τ, ε(v)) - q * div(u) - p * div(v) - inner(f, v)) * dx

### Boundary conditions ###
face_ids = ['top', 'bottom', 1, 2, 3, 4]

bc_stokes = []
for id in face_ids:
    if id == 'top': pass # Skip the top face for now (free flow)
    else:
        bc = firedrake.DirichletBC(Y.sub(0), as_vector((0, 0, 0)), id) # No flow on the boundaries
        bc_stokes.append(bc)

### Solve the flow model on the coarse mesh ###

basis = firedrake.VectorSpaceBasis(constant=True, comm=firedrake.COMM_WORLD)
nullspace = firedrake.MixedVectorSpaceBasis(Y, [Y.sub(0), basis])

stokes_problem = firedrake.NonlinearVariationalProblem(F_momentum, flow, bc_stokes)
parameters = {
    "nullspace": nullspace,
    "solver_parameters": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}

stokes_solver = firedrake.NonlinearVariationalSolver(stokes_problem, **parameters)

print('Starting Coarse Solve')
stokes_solver.solve()

print('Saving the coarse mesh flow model solution...')
from firedrake.checkpointing import DumbCheckpoint

chk = DumbCheckpoint("../Saved_Models/coarse_mesh_flow_final_model", mode=FILE_CREATE)
chk.store(flow, name="coarse_mesh_flow")

import sys
sys.exit()

"""

Interpolate the solution onto a standard mesh from 0 - 1, so we can transfer the solution to
a fine mesh later.

"""

pressure_space_coarse_norm = firedrake.FunctionSpace(unit_extruded_mesh, "CG", 1)
velocity_space_coarse_norm = firedrake.VectorFunctionSpace(unit_extruded_mesh, "CG", 3)
Y_coarse_norm = velocity_space_coarse_norm * pressure_space_coarse_norm
flow_coarse_norm = firedrake.Function(Y_coarse_norm)

flow_coarse_norm.sub(0).dat.data[:] = flow.sub(0).dat.data_ro
flow_coarse_norm.sub(1).dat.data[:] = flow.sub(1).dat.data_ro

""""

Now, we load the fine mesh into DEM points preocessable by the model.

"""

src_bed = src_bed_fine
src_surf = src_surf_fine

bed_DEM = src_bed.read(1,
                        out_shape=(
                            src_bed.count,
                            int(src_bed.height * upscale_factor_fine),
                            int(src_bed.width * upscale_factor_fine)
                        ),
                        resampling=Resampling.bilinear
                    )

# scale image transform
src_bed_transform = src_bed.transform * src_bed.transform.scale(
    (src_bed.width / bed_DEM.shape[-1]),
    (src_bed.height / bed_DEM.shape[-2])
)

surf_DEM = src_surf.read(1,
                        out_shape=(
                            src_surf.count,
                            int(src_surf.height * upscale_factor_fine),
                            int(src_surf.width * upscale_factor_fine)
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

"""
Create a unite fine mesh to interpolate the coarse model results onto.
"""

layers = 5 # number of layers in z direction of the extruded mesh
nx = width - 1
ny = height - 1

Lx = lons.max() # x extent, get this from the DEM
Ly = lats.max() # y extent, get this from the DEM

origin_x = lons.min() # origin (lower left corner) in x, get this from the DEM
origin_y = lats.min() # origin (lower left corner) in y, get this from the DEM

base_mesh = RectangleMesh(nx,ny, Lx, Ly, originX=origin_x, originY=origin_y)
# Make a height 1 mesh.
unit_extruded_mesh = ExtrudedMesh(base_mesh, layers=layers)

xi = base_mesh.coordinates.dat.data_ro[:,:2] # Get the mesh coordinates from the base mesh

pressure_space_fine_norm = firedrake.FunctionSpace(unit_extruded_mesh, "CG", 1)
velocity_space_fine_norm = firedrake.VectorFunctionSpace(unit_extruded_mesh, "CG", 3)
Y_fine_norm = velocity_space_fine_norm * pressure_space_fine_norm


print('Interpolating coarse solve onto fine mesh...')
"""
Interpolate the model results onto the fine mesh unit extruded mesh.
"""
flow_fine_interp = assemble(interpolate(flow_coarse_norm, Function(Y_fine_norm)))

"""
Now create the fine mesh into real coordinates.
"""

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
mesh_fine = Mesh(new_coordinates) # Mesh(new_coordinates, tolerance=50)
x_fine, y_fine, z_fine = SpatialCoordinate(mesh_fine)

"""

Build the flow model function spaces and momentum equation.

"""

μ = Constant(5e15) #Ice


pressure_space = firedrake.FunctionSpace(mesh_fine, "CG", 1)
velocity_space = firedrake.VectorFunctionSpace(mesh_fine, "CG", 3)
Y = velocity_space * pressure_space
flow = firedrake.Function(Y)
u, p = firedrake.split(flow)
v, q = firedrake.TestFunctions(flow.function_space())

τ = 2 * μ * ε(u)
g = as_vector((0, 0, grav))
f =  ρ * g

F_momentum = (inner(τ, ε(v)) - q * div(u) - p * div(v) - inner(f, v)) * dx


"""
Boundary of the conditions for the fine mesh model. This is the results from the coarse mesh.
"""

# Collapse velocity subspace for compatibility
V_trans = Y.sub(0).collapse()

# Interpolate old flow into that space
u_bc_func = Function(V_trans).interpolate(flow_fine_interp.sub(0))

face_ids = ['top', 'bottom', 1, 2, 3, 4]

bc_stokes = []
for id in face_ids:
    if id == 'top': pass # Skip the top face for now (free flow)
    elif id == 'bottom': 
        bc = firedrake.DirichletBC(Y.sub(0), as_vector((0, 0, 0)), id) # No flow on the bed
        bc_stokes.append(bc)
    else:
        bc = firedrake.DirichletBC(Y.sub(0), u_bc_func, id) # coarse mesh solution on the vertical sides
        bc_stokes.append(bc)

"""

Solve the flow model on the fine mesh.

"""

basis = firedrake.VectorSpaceBasis(constant=True, comm=firedrake.COMM_WORLD)
nullspace = firedrake.MixedVectorSpaceBasis(Y, [Y.sub(0), basis])

stokes_problem = firedrake.NonlinearVariationalProblem(F_momentum, flow, bc_stokes)
parameters = {
    "nullspace": nullspace,
    "solver_parameters": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}

stokes_solver = firedrake.NonlinearVariationalSolver(stokes_problem, **parameters)

print('Starting fine mesh solve...')
stokes_solver.solve()

print('Saving the fine mesh flow model solution before viscosity updating...')
from firedrake.checkpointing import DumbCheckpoint

chk = DumbCheckpoint("../Saved_Models/fine_mesh_flow_"+resolution_of_model+"m_constant_viscosity"+str(e_param), mode=FILE_CREATE)
chk.store(flow, name="fine_mesh_flow_constant")

from ufl import Measure
dx = ufl.Measure("dx", domain=mesh_fine)

V = firedrake.FunctionSpace(mesh_fine, "CG", 1)
# V = firedrake.FunctionSpace(mesh, extruded_element)

T = firedrake.Function(V)
ϕ = firedrake.TestFunction(V)

geo_flux = 0.05 # W/m^2 
k = Constant(2.22)  # W / m C

geothermal_flux = -geo_flux*ϕ  * ds_b
F_diffusion = k*inner(grad(T), grad(ϕ)) * dx
F_advection = - ρ * c * T * inner(u, grad(ϕ)) * dx

F = F_diffusion + F_advection + geothermal_flux

T_mean = -33 #average temp (C)

temperature_expr = T_mean - (z_fine - 2000)*.01

surface_temp_bc = firedrake.DirichletBC(V, temperature_expr, 'top')

print('Running temperature model...')

firedrake.solve(F == 0, T, [surface_temp_bc])

print("Saving the temperature model solution ...")
chk = DumbCheckpoint("../Saved_Models/fine_mesh_temperature_"+resolution_of_model+"m_constant_viscosity"+str(e_param), mode=FILE_CREATE)
chk.store(T, name="fine_mesh_temp")

print("Updating the viscosity . . .")

T_new,flow, stokes_solver = viscosity_updater_3d(x_fine,z_fine,ϕ,T,flow,u,V,surface_temp_bc,mesh_fine, μ, u_bc_func, enhancement_factor=e_param)

print("Saving the flow model solution with updated viscosity ...")
chk = DumbCheckpoint("../Saved_Models/fine_mesh_flow_"+resolution_of_model+"m_updated_viscosity_e"+str(e_param), mode=FILE_CREATE)
chk.store(flow, name="fine_mesh_flow")

print("Saving the temperature model solution with updated viscosity...")
chk = DumbCheckpoint("../Saved_Models/fine_mesh_temperature_"+resolution_of_model+"m_updated_viscosity_e"+str(e_param), mode=FILE_CREATE)
chk.store(T_new, name="fine_mesh_temp")