import os
import sys
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
import numpy as np 
import scienceplots
import matplotlib.pyplot as plt
plt.style.use('science')
plt.style.use(['science','ieee'])
import matplotlib
from scipy.stats import gaussian_kde
from mpi4py import MPI
import dolfinx as dlf
from DarcyFlow.common import EquSolverDarcyFlow
from core.probability import GaussianElliptic2
from Mix_Gaussian import Mix_Gauss_IS_proposal
from pathlib import Path
from core.plot import plot_fun2d
from matplotlib.ticker import MaxNLocator

matplotlib.use('Agg')# the picture is not shown
data_folder1='./results/example2SMCGM/'
data_folder2='./results/example2SMCpCN/'
idx1=int(np.load(data_folder1+'idx_last_layer.npy'))
idx2=int(np.load(data_folder2+'idx_last_layer.npy'))
data1= np.load(data_folder1+'particles'+str(idx1)+'.npy')
data2= np.load(data_folder2+'particles'+str(idx2)+'.npy')
results_folder = Path("results/example2Analysis")
results_folder.mkdir(exist_ok=True, parents=True)
######################################### get GMM #############################
nx = 20
msh = dlf.mesh.create_unit_square(MPI.COMM_SELF, nx, nx, dlf.mesh.CellType.triangle)
equ_solver = EquSolverDarcyFlow(msh, degree=1)
params = {
    "theta": lambda x: 1 + 0.0*x[0],
    "ax": lambda x: 1 + 0.0*x[0],
    "mean": lambda x: 0.0*np.sin(2*np.pi*x[0])*np.cos(2*np.pi*x[1])
}
prior = GaussianElliptic2(equ_solver.Vh, params)
GMM=Mix_Gauss_IS_proposal(prior)
###############################################################################
eig_vecs=GMM.e[:,:50]

fig_size=(16,10)
width_frame_line=1
font_size=50
pad_tem=10


# plot the marginal density of fourier coef
for idx_eig in range(20):
    #######################################################################################################
    samples=data1@GMM.prior.M@eig_vecs[:,idx_eig]
    kde = gaussian_kde(samples)
    x = np.linspace(min(samples), max(samples), 1000)
    kde_values = kde(x)
    fig=plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major',labelsize=font_size,pad=pad_tem, width=1,   length=5  ) 
    ax.tick_params(axis='both', which='minor',labelsize=font_size,pad=pad_tem, width=0.5, length=2.5) 

    # plt.hist(samples, bins=30, density=True, alpha=0.5, color='blue', label='Histogram')
    plt.plot(x, kde_values, label='GM', color='red', linestyle='--',linewidth=2)
    #######################################################################################################
    samples=data2@GMM.prior.M@eig_vecs[:,idx_eig]
    kde = gaussian_kde(samples)
    xx = np.linspace(min(samples), max(samples), 1000)
    kde_values = kde(x)    
    plt.plot(xx, kde_values, label='pCN', color='blue', linestyle='-',linewidth=4)
    ax = plt.gca()  # get axis
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))  # Set the maximum number of major ticks on the x-axis.
    #######################################################################################################
    plt.legend(fontsize=font_size,loc='upper left')
    plt.xlabel('Value',fontsize=font_size,labelpad=pad_tem)
    plt.ylabel('Density',fontsize=font_size,labelpad=pad_tem)
    # plt.title('Density Estimation from 1000 Samples',fontsize=20)
    plt.xticks(fontsize=font_size)  
    plt.yticks(fontsize=font_size) 
    # plt.axvline(x=true_fun.x.array[:]@prior.M@eig_vecs[:,idx_eig], color='red', linestyle='--', label='Specific Value')
    plt.savefig(results_folder/('pdf'+str(idx_eig)+'.png'))
    plt.close()


#compute the average mean of L2 error between two pdfs
err=[]
for idx_eig in range(20):
    samples1=data1@GMM.prior.M@eig_vecs[:,idx_eig]
    kde1 = gaussian_kde(samples1)
    samples2=data2@GMM.prior.M@eig_vecs[:,idx_eig]
    kde2 = gaussian_kde(samples2)
    x = np.linspace(min(min(samples1),min(samples2)), max(max(samples1),max(samples2)), 1000)
    kde_values1 = kde1(x)    
    kde_values2 = kde2(x)
    dx=x[1]-x[0]
    tem=0.5*np.sum(np.abs(kde_values1-kde_values2))*dx# totaal variation distance
    # tem=0.5*np.sum((np.sqrt(kde_values1)-np.sqrt(kde_values2))**2)*dx
    # err.append(np.sqrt(tem))
    err.append(tem)
print(np.mean(err))
################################################################################
from dolfinx import fem
from core.misc import project

data_folder = Path("data")
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
true_fun = project(true_fun, equ_solver.Vh)
plot_fun2d(
    true_fun, nx=200, show=False, path=results_folder / ("truth.png"),
    grid_on=False, package="matplotlib"
)
######################################################################################
true_vec=true_fun.x.array[:]
mean1=np.mean(data1,axis=0)
mean2=np.mean(data2,axis=0)
err1=np.sqrt((mean1-true_vec)@prior.M@(mean1-true_vec))/np.sqrt(true_vec@prior.M@true_vec)
err2=np.sqrt((mean2-true_vec)@prior.M@(mean2-true_vec))/np.sqrt(true_vec@prior.M@true_vec)
print('The relative errors between sample means and truth are:')
print('SMCGM:',err1)
print('SMCpCN:',err2)

matplotlib.use('Qt5Agg')  # the picture is shown
