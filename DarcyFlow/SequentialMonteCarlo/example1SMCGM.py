import os
import sys
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
import numpy as np
import matplotlib.pyplot as plt
from Mix_Gaussian import Mix_Gauss_IS_proposal
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import matplotlib
from core.probability import GaussianElliptic2, simple_sqrt_eig_vals

from pathlib import Path
results_folder = Path("results/example1SMCGM")
results_folder.mkdir(exist_ok=True, parents=True)

dim=10
num_particles=20000
matplotlib.use('Agg')
def clustering(samples):
    def gmm_bic_score(estimator, samples):
        return -estimator.bic(samples)

    param_grid = {
        "n_components": range(1, 6),
        "covariance_type": ["diag"],#full
    }

    grid_search = GridSearchCV(GaussianMixture(), param_grid=param_grid,
                               scoring=gmm_bic_score)
    grid_search.fit(samples)
    labels = grid_search.best_estimator_.predict(samples)
    weights = grid_search.best_estimator_.weights_

    result = grid_search.cv_results_

    idx = np.argmax(result["mean_test_score"])  # i.e. argmin(BIC)
    num_clusters = result["param_n_components"][idx]
    print("The samples are clustered into " +
          str(num_clusters)+" partitions")
    clusters = []
    for i in range(num_clusters):
        idx = labels == i
        clusters.append(samples[idx])
    return weights, clusters

max_value=1/np.sqrt(2)

peak1 = np.zeros(dim)
peak1[1]+=-max_value

peak2 = np.zeros(dim)
peak2[1]+=max_value

peak3 = np.zeros(dim)
peak3[2]+=max_value

peak4 = np.zeros(dim)
peak4[3]+=-max_value


noise_level=0.01
noise_std=max_value*noise_level
peak1+=np.random.normal(0,noise_std,dim)
peak2+=np.random.normal(0,noise_std,dim)
peak3+=np.random.normal(0,noise_std,dim)
peak4+=np.random.normal(0,noise_std,dim)


peaks=[peak1,peak2,peak3,peak4]

class prior_:
    def __init__(self,params=None):
        self.params = params
    def generate_sample(self,n):
        eig_vals=simple_sqrt_eig_vals(self.params)
        ans=[]
        for i in range(dim):
            ans.append(np.random.normal(0,eig_vals[i],n))
            # ans.append(np.random.normal(0,1,n))
        return np.array(ans).T

class GMM_obj:
    def __init__(self,weights,clusters):
        self.weights=weights
        self.clusters=clusters
        self.n_clusters=len(weights)
        self.means=[]
        self.covs=[]
        for i in range(self.n_clusters):
            data=clusters[i]
            self.means.append(np.mean(data, axis=0))
            self.covs.append(np.cov(data, rowvar=False, bias=True))
    def generate_sample(self,n):
        particles=[]
        for i in range(n):
            idx = np.random.choice(self.n_clusters, p=self.weights)
            particle = multivariate_normal.rvs(self.means[idx], self.covs[idx])
            particles.append(particle)
        return np.array(particles)
##################################################
x_values = np.linspace(0, 1, 101)
e1=np.ones(x_values.shape[0])
es=[e1]
for i in range(9):
    es.append(np.cos((i+1)* np.pi*x_values)/np.sqrt(0.5))
eig_vecs=np.array(es)
#################################################
          
params = {
    "theta": lambda x: 0.01 + 0.0*x[0],
    "ax": lambda x: 1 + 0.0*x[0],
    "mean": lambda x: 0*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
}
prior=prior_(params)
particles=prior.generate_sample(num_particles)

def phi(x):
    loss=[]
    for peak in peaks:
        tem_loss=0.5/noise_std**2 *(x-peak)@(x-peak)
        loss.append(tem_loss)
    return -logsumexp(-np.array(loss))

sum_h=0
h = 1e-4
for idx_layer in range(100):
    plt.figure()
    tem=particles[0:1000,:]@eig_vecs#/2
    plt.plot(np.linspace(0, 1, 101),tem.T,linewidth=0.1,linestyle='-',color='skyblue')
    plt.plot(np.linspace(0, 1, 101),tem.T[:,0],linewidth=0.5,linestyle='-',color='skyblue',label='samples')
    ESS=0
    ################### find a good h##########################################        
    while ESS<0.6*num_particles:
        phis=[]
        for i in range(num_particles):
            phis.append(phi(particles[i]))        
        phis=np.array(phis)
        tmp = -h*phis
        weights = np.exp(tmp - logsumexp(tmp))
        
        ESS=1/(weights@weights)
        h/=2
    sum_h+=h
    print('h=',h,'sum_h=',sum_h,'k=',idx_layer)
    if sum_h>1:
        break
    h*=3 
    #############################################################################
    idx_resample = np.random.choice(num_particles, num_particles, True, weights)
    particles = particles[idx_resample]
    weights, clusters=clustering(particles)
    for i in range(len(clusters)):
        print(clusters[i].shape[0])
        plt.plot(np.linspace(0, 1, 101),np.mean(clusters[i],axis=0)@eig_vecs,linestyle='-',color='red')
    plt.plot(np.linspace(0, 1, 101),np.mean(clusters[0],axis=0)@eig_vecs,linestyle='-',color='red',label='means')
    plt.legend()
    plt.savefig(results_folder/('layer'+str(idx_layer)+'samples.png'))
    plt.close()
    
    GMM=GMM_obj(weights,clusters)
    particles=GMM.generate_sample(num_particles)
    


