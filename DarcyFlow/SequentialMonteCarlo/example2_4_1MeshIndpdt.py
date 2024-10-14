## The following codes control the number of threads the numpy and scipy can employed.
## Sometimes, using more cores will lead to worser performance.
import os
nthreads = 1
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)
os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
## The above must be added before all of the codes.
import sys
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from mpi4py import MPI
import dolfinx as dlf
from dolfinx import fem
import numpy as np
from scipy.special import logsumexp
from pathlib import Path
import time
import argparse

from DarcyFlow.common import EquSolverDarcyFlow, ModelDarcyFlow
from core.probability import GaussianElliptic2
from core.noise import NoiseGaussianIID
from core.approximate_sample import rMAP
from core.sample import SMC, pCN_preserve_GMM
from core.misc import project, error_compare
from core.plot import plot_fun2d
from Mix_Gaussian import Mix_Gauss_IS_proposal

np.random.seed(12345)
## basic info of the MPI
comm = MPI.COMM_WORLD
name = MPI.Get_processor_name()
rank = comm.rank

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=20, help='specify nx')
args = parser.parse_args()
print("Arguments passed to the program:")
for arg_name in vars(args).keys():
    print(f"{arg_name}: {getattr(args, arg_name)}")
nx = args.nx

## set the data and results path
data_folder = Path("data")
results_folder = Path("results/example2MeshIndpdt")
data_folder.mkdir(exist_ok=True, parents=True)
results_folder.mkdir(exist_ok=True, parents=True)

## load measured data
noise_level = 0.02
data = {"coordinates": None, "data": None}
data["coordinates"] = np.load(data_folder/"measure_coordinates.npy", allow_pickle=True)
datafile = "noisy_data_" + str(noise_level) + ".npy"
data["data"] = np.load(data_folder/datafile, allow_pickle=True)
clean_data = np.load(data_folder/"clean_data.npy")

## construct prior, noise, equ_solver, and finally construct model
msh = dlf.mesh.create_unit_square(MPI.COMM_SELF, nx, nx, dlf.mesh.CellType.triangle)
equ_solver = EquSolverDarcyFlow(msh, degree=1)
## generate truth and save
params = {
    "theta": lambda x: 1 + 0.0*x[0],
    "ax": lambda x: 1 + 0.0*x[0],
    "mean": lambda x: 0.0*np.sin(2*np.pi*x[0])*np.cos(2*np.pi*x[1])
}
prior = GaussianElliptic2(equ_solver.Vh, params)
noise = NoiseGaussianIID(len(data["data"]))
noise.set_parameters(std_dev=noise_level * max(abs(clean_data)))
model = ModelDarcyFlow(prior, equ_solver, noise, data)


if rank == 0:
    ## calculate the relative error of posterior mean with the background truth
    with dlf.io.XDMFFile(MPI.COMM_SELF, data_folder/"true_function.xdmf", "r") as xdmf:
        msh = xdmf.read_mesh()
    f = open(data_folder/'fun_type_info.txt', "r")
    fun_type = f.read()
    f = open(data_folder/'fun_degree_info.txt', "r")
    fun_degree = int(f.read())
    f.close()
    Vh_true = fem.FunctionSpace(msh, (fun_type, fun_degree))
    true_fun = fem.Function(Vh_true)
    true_fun.x.array[:] = np.array(np.load(data_folder/"fun_data.npy"))
    true_fun = project(true_fun, model.function_space)
    plot_fun2d(
        true_fun, nx=200, show=False, path=results_folder / ("truth.png"),
        grid_on=False, package="matplotlib"
    )
    
    def file_process(file_name, contents=None):
        '''
        Delete the old file and create a file with same name
        '''
        # Check if the file exists
        if os.path.exists(file_name):
            # Delete the existing file
            try:
                os.remove(file_name)
                print(f"Deleted existing '{file_name}'.")
            except OSError as e:
                print(f"Error deleting '{file_name}': {e}")

        # Create a new empty file
        try:
            with open(file_name, "w") as f:
                if contents is None:
                    pass
                else:
                    f.write(contents)
            print(f"Created empty '{file_name}'.")
        except OSError as e:
            print(f"Error creating '{file_name}': {e}")

    file_weights = results_folder / "weights.txt"
    file_relative_errors = results_folder / "relative_errors.txt"
    file_times = results_folder / "times.txt"
    file_acc_rates = results_folder / "acc_rates.txt"
    file_find_h = results_folder / "h_and_ESS.txt"
    file_h = results_folder / ("h"+str(nx)+".txt")

    file_process(file_weights, contents="----weights of each particle----\n")
    file_process(file_relative_errors, contents="----relative_errors----\n")
    file_process(file_times, contents="----running times----\n")
    file_process(file_acc_rates, contents="----average accept rate----\n")
    file_process(file_find_h, contents="----h and ESS:----\n")
    file_process(file_h, contents="\n")


    ## Delete all of the generated figures except the true_fun.png
    for filename in os.listdir(results_folder):
        # Check if it's a .png file (case-insensitive)
        if filename.lower().endswith(".png") and filename != "true_fun.png":
            # Delete the file
            try:
                os.remove(os.path.join(results_folder, filename))
                print(f"Deleted file: {filename}")
            except OSError as e:
                print(f"Error deleting {filename}: {e}")


############################### SMC ###########################################
num_particles = 1000
smc = SMC(comm, num_particles,model=model)
smc.prepare()
smc.gather_samples()
GMM=Mix_Gauss_IS_proposal(prior,comm=comm,max_num_GM=2)

num_particles_smc2 = 200 ## used in smc2, which aimes to find h
idx_h = 0
h = 0.4  # at idx_h=0, the initial guess
sum_h = 0  # if sum h=1 ,we should move to next theta
while sum_h < 1: 
    start_times = time.time()
    if idx_h>1: # at idx_h>0, the initial geuss is 3*h, but it should < 1
        h=min(1,4*h) 
    while 1: # this loop finds h using bisection
        smc2 = SMC(comm, num_particles_smc2, model)
        smc2_particles = smc.particles[:num_particles_smc2]
        smc2.prepare(smc2_particles)
        smc2.resampling(potential_fun=model.loss_res, h=h)
        # if rank == 0:
        #     print('h=1e-'+str(i+1), ",ESS=", int(smc2.ESS))
        if smc2.ESS > 0.6*num_particles_smc2:
            # print(smc2.ESS,num_particles_smc2,'idx_h=',idx_h)
            if sum_h+h>=1:
                h=1-sum_h
                #break
            break
        h /= 1.4#
    if rank == 0: # write h and ESS in h_and_ESS.txt
        with open(file_find_h, "a") as f:
            formatted_string = f"{idx_h+1:2d}, h = {h:.7f}, "
            formatted_string += f"Sum h = {sum_h:.7f},ESS={int(smc2.ESS):2d}/100\n"
            f.write(formatted_string)
        with open(file_h, "a") as f:
            formatted_string = f"{sum_h:.7f}\n"
            f.write(formatted_string)
    sum_h += h
    idx_h += 1
###################################################################################
    smc.resampling(potential_fun=model.loss_res,h=h)
    GMM.update_samples(smc.particles)
    sampler = pCN_preserve_GMM(model, GMM,reduce_chain=np.int64(1e4))
    smc.transition(sampler=sampler,len_chain=5, info_acc_rate=False)
    smc.gather_samples()
    smc.gather_acc_rates()
    acc_rates = smc.acc_rates
       
    end_times = time.time()
    start_times = comm.gather(start_times, root=0)
    end_times = comm.gather(end_times, root=0)
    
    if rank == 0:
        start_time = min(start_times)
        end_time = max(end_times)
        with open(file_times, "a") as f:
            formatted_string = f"{idx_h+1:2d},"
            formatted_string += f"elapsed time = {end_time - start_time:.5f} seconds."
            formatted_string += "\n"
            f.write(formatted_string)

        if idx_h%1==0:
            np.save(results_folder/("particles"+str(idx_h)), smc.particles)

        ## save mean function for each layer to check the performance
        post_mean_fun = fem.Function(model.function_space)
        tmp = 0.0
        for idx, particle in enumerate(smc.particles):
            tmp += particle
        post_mean_fun.x.array[:] = np.array(tmp / len(smc.particles))
        plot_fun2d(
            post_mean_fun, nx=200, show=False, path=results_folder / ("post_mean_smc_GM_h"+str(idx_h)+".png"),
            grid_on=False, package="matplotlib"
        )

        error_L2, error_max, _ = error_compare(true_fun, post_mean_fun)
        formatted_string = f"{idx_h:2d}, "
        formatted_string += f"err_L2 = {error_L2:.5f}, err_max = {error_max:.5f}\n"
        with open(file_relative_errors, "a") as f:
            f.write(formatted_string)
