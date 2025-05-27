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

# Constants
spy = 365.25 * 24 * 60 * 60
ρ = Constant(917.0)    # kg / m^3
c = Constant(2180)     # J / kg C
k = Constant(2.22)  # W / m C
α = (k/(ρ*c)) # diffusivity
A = 3.5e-25 
n = 3
R = 8.31446261815324
act_vol = -1.74e-5
Q = 6e4 # activatyion energy J/mol

dTdz = 2e-2 #C per 100m
geo_flux = Constant(60e-3) #W/m^2

λ = Constant(10.0)    # m
grav = Constant(-9.81) #m/s^2

vel_mpy = Constant(0.50) #m/yr
vel = vel_mpy / spy

## Mesh IDs ##
bed_id = [4]
left_id = [1]
surface_id = [2]
right_id = [3]

def ε(u):
    return sym(grad(u))

def interpolate_2d_grid(left_boundary, right_boundary, num_columns):
    """
    Interpolates a 2D grid between two lines acting as the left and right boundary conditions.
    
    Parameters:
    left_boundary: list of tuples (x, y), the left boundary line's points
    right_boundary: list of tuples (x, y), the right boundary line's points
    num_columns: int, number of interpolation points (columns) in the grid between the boundaries

    Returns:
    grid: 2D numpy array representing the interpolated grid
    """
    # Ensure both boundaries have the same number of points (rows)
    assert len(left_boundary) == len(right_boundary), "Left and right boundaries must have the same number of points"
    
    # Number of rows in the grid
    num_rows = len(left_boundary)
    
    # Initialize the grid
    grid = np.zeros((num_rows, num_columns, 2))
    
    # Interpolate each row between the corresponding points on the left and right boundaries
    for i in range(num_rows):
        p_left = np.array(left_boundary[i])
        p_right = np.array(right_boundary[i])
        
        # Interpolate linearly between left and right boundaries for each row
        for j, t in enumerate(np.linspace(0, 1, num_columns)):
            grid[i, j, :] = p_left * (1 - t) + p_right * t
    
    return grid

def merge(temp, depth, num_points):
    """
    Resamples temperature and depth data to a specified number of points.
    
    Args:
        temp (list): Temperature values.
        depth (list): Depth values.
        num_points (int): Number of samples in output.
    
    Returns:
        list of tuples: List of (temp, depth) tuples resampled to num_points.
    """
    temp = np.array(temp)[::-1]
    depth = np.array(depth)[::-1]

    # Ensure inputs are same length
    if len(temp) != len(depth):
        raise ValueError("Temperature and depth lists must be the same length.")

    # Interpolation over depth
    interp_depth = np.linspace(depth.min(), depth.max(), num_points)
    interp_temp = np.interp(interp_depth, depth, temp)

    merged = list(zip(interp_temp, interp_depth))
    return merged

def flow_bcs(mesh, z, Y):
    w_s_divide = -0.05 / spy
    w_s_abl = 0.02 / spy
    
    divide_heights = []
    for i in mesh.coordinates.dat.data:
        if i[0] == np.max(mesh.coordinates.dat.data[:,0]):
            divide_heights.append(i[1])
    divide_depth = np.max(divide_heights) - np.min(divide_heights)
    w_divide = w_s_divide * (((z - np.min(divide_heights)) / divide_depth))
    
    abl_boundary_heights = []
    for i in mesh.coordinates.dat.data:
        if i[0] == 0:
            abl_boundary_heights.append(i[1])
    abl_depth = np.max(abl_boundary_heights) - np.min(abl_boundary_heights)
    w_abl = w_s_abl * (((z - np.min(abl_boundary_heights)) / abl_depth))
    
    bc_l = firedrake.DirichletBC(Y.sub(0), as_vector((0, w_abl)), left_id)
    bc_r = firedrake.DirichletBC(Y.sub(0), as_vector((0, w_divide)), right_id)
    bc_b = firedrake.DirichletBC(Y.sub(0), as_vector((0, 0)), bed_id)

    # Test with zero flow along boundaries
    # bc_l = firedrake.DirichletBC(Y.sub(0), as_vector((0, 0)), left_id)
    # bc_r = firedrake.DirichletBC(Y.sub(0), as_vector((0, 0)), right_id)
    # bc_b = firedrake.DirichletBC(Y.sub(0), as_vector((0, 0)), bed_id)
    
    bc_stokes=[bc_l, bc_b, bc_r]

    return bc_stokes

def initial_conditions(mesh, climate_trend):
    μ = Constant(5e15) #Ice
    x, z, V, T, ϕ = starting_stuff(mesh)

    pressure_space = firedrake.FunctionSpace(mesh, "CG", 1)
    velocity_space = firedrake.VectorFunctionSpace(mesh, "CG", 2)
    Y = velocity_space * pressure_space
    y = firedrake.Function(Y)
    u, p = firedrake.split(y)
    v, q = firedrake.TestFunctions(y.function_space())
    
    τ = 2 * μ * ε(u)
    g = as_vector((0, grav))
    f =  ρ * g
    
    F_momentum = (inner(τ, ε(v)) - q * div(u) - p * div(v) - inner(f, v)) * dx

    ### Flow boundary conditions ###
    bc_stokes = flow_bcs(mesh,z,Y)

    basis = firedrake.VectorSpaceBasis(constant=True, comm=firedrake.COMM_WORLD)
    nullspace = firedrake.MixedVectorSpaceBasis(Y, [Y.sub(0), basis])
    
    stokes_problem = firedrake.NonlinearVariationalProblem(F_momentum, y, bc_stokes)
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
    

    T_mean = climate_trend[0]+0.00001 #average temp (C)
    T_surface = Constant(T_mean)

    
    temperature_expr = T_mean - (z - 2000)*.01
    top_temp_bc = firedrake.DirichletBC(V, temperature_expr, surface_id)
    bcs_temp = [top_temp_bc]
    
    ### Defining the variational problem for temperature ###
    geothermal_flux = -geo_flux*ϕ  * ds((bed_id[0]))
    
    F_diffusion = k * inner(grad(T), grad(ϕ)) * dx
    F_advection = - ρ * c * T * inner(u, grad(ϕ)) * dx
    
    F_0 = F_advection + F_diffusion + geothermal_flux
    
    firedrake.solve(F_0 == 0, T, bcs_temp)

    return x, z, ϕ, T, y, u, V, bcs_temp

def viscosity_updater(x, z, ϕ, T, y, u, V, bcs_temp, mesh):
    velocity_field = y.sub(0).dat.data
    temp_field = T.dat.data
    
    for i in range(100):
        prev_temp_field = copy.deepcopy(temp_field)
        prev_velocity = copy.deepcopy(velocity_field)
        ϵ_ = sym(grad(u))
        
        ϵ_effective = sqrt((inner(ϵ_, ϵ_)+tr(ϵ_)**2)*0.5)
        
        # A_new = A*exp(-((Q + (p*act_vol))/((T+273.15)*R)))
    
        A_new = A*exp(-(Q/R*( 1/(T+273.15) - 1/263))) #Does not account for melting point depression
        μ_new =  0.5*(A_new**(-1/n))*(ϵ_effective**((1/n)-1))
        
        μ_new_field = Function(V).project(μ_new)
        ϵ_effective_field = Function(V).project(ϵ_effective)
        A_new_field = Function(V).project(A_new)
    
        def ε(u):
            return sym(grad(u))
        
        ### Build the stokes flow model ###
        
        pressure_space = firedrake.FunctionSpace(mesh, "CG", 1)
        velocity_space = firedrake.VectorFunctionSpace(mesh, "CG", 2)
        Y = velocity_space * pressure_space
        
        y = firedrake.Function(Y)
        u, p = firedrake.split(y)
        v, q = firedrake.TestFunctions(y.function_space())
        
        τ = 2* μ_new * ε(u)#  2 * μ_new_field *  ϵ_
        g = as_vector((0, grav))
        f =  ρ * g
        F_momentum = (inner(τ, ε(v)) - q * div(u) - p * div(v) - inner(f, v)) * dx
    
        basis = firedrake.VectorSpaceBasis(constant=True, comm=firedrake.COMM_WORLD)
        nullspace = firedrake.MixedVectorSpaceBasis(Y, [Y.sub(0), basis])
        
        bc_stokes = flow_bcs(mesh,z,Y)
        stokes_problem = firedrake.NonlinearVariationalProblem(F_momentum, y, bc_stokes)
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
    
        
        ### Get the new velocity field solved for ###
        velocity_field = y.sub(0).dat.data
    
    
        
        ### Build the temperature model ###
        
        F_diffusion = k*inner(grad(T), grad(ϕ)) * dx
        F_advection = - ρ * c * T * inner(u, grad(ϕ)) * dx
        geothermal_flux = -geo_flux*ϕ  * ds((bed_id[0]))
     
        F_0 = F_advection + F_diffusion + geothermal_flux
    
        ### Solve for the temperature field
        firedrake.solve(F_0 == 0, T, bcs_temp)
    
        ### Get the new temp field we just solved for ###
        temp_field = T.dat.data
    
        ### Calculate the residuals in the temp field ###
        residual = np.sum(np.abs((prev_temp_field - temp_field)))/temp_field.shape[0]
        print(residual)
        if residual < 0.01: #If the residual is less than the 0.01 m/yr, let's call it good.
            break
    return T,y, stokes_solver

def starting_stuff(mesh):
    x, z = firedrake.SpatialCoordinate(mesh)
    element = firedrake.FiniteElement("CG", "quadrilateral", 1)
    V = firedrake.FunctionSpace(mesh, element)
    
    T = firedrake.Function(V)
    ϕ = firedrake.TestFunction(V)

    return x, z, V, T, ϕ

def climate_modeler():
    return