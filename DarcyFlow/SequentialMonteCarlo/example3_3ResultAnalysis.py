import numpy as np 
import scienceplots
import matplotlib.pyplot as plt
import matplotlib
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
from pathlib import Path
from DarcyFlow.common import EquSolverDarcyFlow, ModelDarcyFlow
from core.probability import GaussianElliptic2
from core.plot import plot_fun2d
from core.noise import NoiseGaussianIID
from core.misc import project
from Mix_Gaussian import Mix_Gauss_IS_proposal
## basic info of the MPI
comm = MPI.COMM_WORLD
name = MPI.Get_processor_name()
rank = comm.rank
###############################################################################
data_folder = Path("data_sparse_measure")
noise_level = 0.02
data = {"coordinates": None, "data": None}
data["coordinates"] = np.load(data_folder/"measure_coordinates.npy", allow_pickle=True)
datafile = "noisy_data_" + str(noise_level) + ".npy"
data["data"] = np.load(data_folder/datafile, allow_pickle=True)
clean_data = np.load(data_folder/"clean_data.npy")
## construct prior, noise, equ_solver, and finally construct model
nx = 20
msh = dlf.mesh.create_unit_square(MPI.COMM_SELF, nx, nx, dlf.mesh.CellType.triangle)
equ_solver = EquSolverDarcyFlow(msh, degree=1)
params = {
    "theta": lambda x: 0.1 + 0.0*x[0],
    "ax": lambda x: 1 + 0.0*x[0],
    "mean": lambda x: 0*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
}
prior = GaussianElliptic2(equ_solver.Vh, params)
noise = NoiseGaussianIID(len(data["data"]))
noise.set_parameters(std_dev=noise_level * max(abs(clean_data)))
model = ModelDarcyFlow(prior, equ_solver, noise, data)
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
################################################################################

results_folder_list=['./results/example3SMCGM/','./results/example3SMCpCN/']
for result_folder in results_folder_list:
    
    
    if result_folder=='./results/example3SMCpCN/':
        method_name="SMCpCN"
    else:
        method_name="SMCGM"
    
    layer_idx=int(np.load(result_folder+'idx_last_layer.npy'))
    #tem_num=18#34#
    print(layer_idx)
    
    tem= np.load(result_folder+'particles'+str(layer_idx)+'.npy')
    tem_fun = fem.Function(prior.Vh)
    GMM=Mix_Gauss_IS_proposal(prior,samples=tem,comm=comm)
    
    
    figure_folder = Path("results/example3Analysis") 
    figure_folder.mkdir(exist_ok=True, parents=True)
    
    for i in range(len(GMM.clusters)):
        cluster=GMM.clusters[i]
        print(cluster.shape[0])
        tem_fun.x.array[:] =GMM.e @ np.array(GMM.m[i])
        plot_fun2d(
            tem_fun, nx=200, show=False, path=figure_folder/(method_name+"mean"+str(i)+".png"),
            grid_on=False, package="matplotlib"
        )
        
        cluster2=GMM.clusters[i]
        
        tem_fun.x.array[:] =np.diag(np.cov(GMM.e@cluster2.T))
        plot_fun2d(
            tem_fun, nx=200, show=False, path=figure_folder/(method_name+"std"+str(i)+".png"),
            grid_on=False, package="matplotlib"
        )
    
    # matplotlib.use('Qt5Agg')
    matplotlib.use('Agg')
    plt.style.use('science')
    plt.style.use(['science','ieee'])
    plt.rcParams['font.family'] = 'Times New Romann'
    font_size=50
    pad_tem=10
    
    
    fig=plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    even_numbers = np.arange(0, 19, 2)
    ax.set_xticks(even_numbers)
    ax.set_xticklabels([str(num) for num in even_numbers])
    ax.tick_params(axis='both', which='major',labelsize=font_size,pad=pad_tem, width=1,   length=5  ) 
    ax.tick_params(axis='both', which='minor',labelsize=font_size,pad=pad_tem, width=0.5, length=2.5) 
    
    # compare data and forward_solve(posterior mean)
    plt.plot(clean_data,lw=3,color='blue',linestyle='--',label='Truth')
    plt.xlabel('Location Number',labelpad=10,fontsize=50)  # Label for x-axis
    plt.ylabel('Measured Value',labelpad=10,fontsize=50)   # Label for y-axis
    data_err=[]
    for i in range(len(GMM.m)):
        tem_data=model.S@ model.equ_solver.forward_solve(GMM.e @ np.array(GMM.m[i]))
        plt.plot(tem_data,lw=0.8,linestyle='-',color='green')
        tem_err=clean_data-tem_data
        data_err.append(np.sqrt(tem_err@tem_err/clean_data@clean_data))
    plt.plot(tem_data,lw=0.8,linestyle='-',color='green',label='Means of clusters')
    plt.legend(fontsize=50) 
    plt.xticks(fontsize=50)  
    plt.yticks(fontsize=50) 
    
    plt.savefig(figure_folder/(method_name+"_FuVStruth.png"))
    plt.close()
    print('average l2 data err = ',np.mean(data_err))
    plt.style.use('default')



