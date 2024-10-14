from mpi4py import MPI
import dolfinx as dlf
import ufl
import dolfinx.fem.petsc as dlfp
from dolfinx import fem
import numpy as np
from pathlib import Path
import time

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))
from DarcyFlow.common import EquSolverDarcyFlow
from core.probability import GaussianElliptic2
from core.misc import construct_measure_matrix
from core.plot import plot_mesh, plot_fun2d


methods = [
    "SequentialMonteCarlo", "rMAP", "MCMC", "OptimizationMethods", "LaplaceApproximate",
    "VariationalInference"
]

data_folder_list, results_folder_list = [], []
for idx, name_method in enumerate(methods):
    data_folder_list.append(Path(name_method + "/data_sparse_measure"))
    results_folder_list.append(Path(name_method + "/results"))
    data_folder_list[idx].mkdir(exist_ok=True, parents=True)
    results_folder_list[idx].mkdir(exist_ok=True, parents=True)

nx = 500
degree = 1
msh = dlf.mesh.create_unit_square(MPI.COMM_WORLD, nx, nx, dlf.mesh.CellType.triangle)
equ_solver = EquSolverDarcyFlow(msh, degree=degree)

## generate truth and save
params = {
    "theta": lambda x: 1 + 0.0*x[0],
    "ax": lambda x: 1 + 0.0*x[0],
    "mean": lambda x: 0.0*np.sin(2*np.pi*x[0])*np.cos(2*np.pi*x[1])
}
measure = GaussianElliptic2(equ_solver.Vh, params)
true_fun = fem.Function(measure.Vh)
true_fun.x.array[:] = measure.generate_sample()

## The following codes for saving need to be refined when the functionality of FEniCSx complete
for idx in range(len(methods)):
    plot_fun2d(
        true_fun, nx=200, show=False, path=results_folder_list[idx] / "true_fun_sparse.svg", \
        grid_on=False, package="matplotlib"
    )
    filename = data_folder_list[idx]/"true_function.xdmf"
    with dlf.io.XDMFFile(msh.comm, filename, "w") as xdmf:
        xdmf.write_mesh(msh)
        ## In FEniCSx, there seems still lack complete io function.
        ## A solution can be found at https://github.com/jorgensd/adios4dolfinx.
        ## However, there seems some errors when I try to use the functions provided by adios4dolfinx.
        xdmf.write_function(true_fun)
    ## Temporary I just store the information of functions
    f = open(data_folder_list[idx]/'fun_type_info.txt', "w")
    f.write("Lagrange")
    f.close()
    f = open(data_folder_list[idx]/'fun_degree_info.txt', "w")
    f.write(str(degree))
    f.close()
    np.save(data_folder_list[idx]/"fun_data", true_fun.x.array)

## solve equations to generate data
w = equ_solver.forward_solve(true_fun.x.array)

xx = np.array([0.6])
yy = np.linspace(0.2, 0.8, 20)

# xx = np.linspace(0.01, 0.99, 10)
# yy = np.linspace(0.01, 0.99, 10)
coordinates = []
for x in xx:
    for y in yy:
        coordinates.append([x, y, 0])
coordinates = np.array(coordinates)

S = construct_measure_matrix(equ_solver.Vh, coordinates)
d = S@w
for idx in range(len(methods)):
    np.save(data_folder_list[idx]/"clean_data", d)

noise_levels = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

for noise_level in noise_levels:
    d_noisy = d + noise_level*max(abs(d))*np.random.normal(0, 1, (len(d),))
    for idx in range(len(methods)):
        np.save(data_folder_list[idx]/"measure_coordinates", coordinates)
        np.save(data_folder_list[idx]/("noisy_data_" + str(noise_level)), d_noisy)


