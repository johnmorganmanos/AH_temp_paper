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

def coarse_boundary_conditions(Y):

    ### Boundary conditions ###
    face_ids = ['top', 'bottom', 1, 2, 3, 4]

    bc_stokes = []
    for id in face_ids:
        if id == 'top': pass # Skip the top face for now (free flow)
        else:
            bc = firedrake.DirichletBC(Y.sub(0), Constant((0,0,0)), id) # No flow on the boundaries
            bc_stokes.append(bc)


    return bc_stokes

def fine_boundary_conditions(Y):

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
    return bc_stokes

def run_flow_model(mesh_src):
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

    ### Get the boundary conditions ###

    bc_stokes = coarse_boundary_conditions(Y)
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

    stokes_solver.solve()

    return flow, Y