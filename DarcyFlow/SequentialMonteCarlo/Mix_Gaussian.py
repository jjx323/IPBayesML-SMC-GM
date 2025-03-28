#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spsl
import time
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from core.sample import MCMCBase
from core.misc import project, error_compare
from dolfinx import fem


class Mix_Gauss_IS_proposal:
    # Ref[1]: AN ADAPTIVE INDEPENDENCE SAMPLER MCMC ALGORITHM FOR INFINITE-DIMENSIONAL BAYESIAN
    # INFERENCES, Jinglai Li
    def __init__(self, prior, project_dim=None, samples=None, comm=None, loss=None
                 , max_num_GM=None):
        self.prior = prior
        self.samples_origin=samples
        self.loss = loss
        self.comm = comm
        if project_dim is not None:
            self.sample_dim=project_dim
        else:
            self.sample_dim = self.compute_project_dim()  # i.e. the project dim
        self.project_dim = self.sample_dim
        if max_num_GM is None:
            self.max_num_GM=15
        else:
            self.max_num_GM=max_num_GM
        
        self.samples_high_dim=None
        if samples is not None:
            self.num_samples = samples.shape[0]
            if comm is not None and comm.rank==0:
                print("There are " + str(samples.shape[0]) +
                  " samples. Dim = "+str(samples.shape[1])+".")
            self.update_samples(samples)   
        
    def update_samples(self, samples):
        if self.samples_high_dim is None:
            self.samples_high_dim=samples @ self.prior.M @ self.prior.eigvec
        
        self.samples = self.project(samples, self.project_dim)
        self.weights, self.clusters = self.clustering(self.samples)
        self.x, self.h, self.m, self.cov = self.estimate_paras()
        # print(self.cov)



    def update_loss(self, loss):
        self.loss = loss

    def compute_acc_rate(self, u, v):
        # based om (3.7),(3.1) and (3.2) in [1]
        # return min(1, mu_y_on_mu(v)/mu_y_on_mu(u))
        # where mu_y is the posterior, and mu is the GMM
        def ln_f(u, j):
            # ln(f), where f(.,j) is f(u,xj,hj) in (3.7)
            xj = self.x[j]
            hj = self.h[j]
            alpha = self.alpha
            betaj = self.cov[j]
            u_proj = self.project(u)
            ans = alpha*alpha*xj*xj/betaj+hj*u_proj*u_proj-2*self.alpha*xj*u_proj/betaj
            ans2 = np.log(alpha)/2-np.log(betaj)/2
            return np.sum(ans2)-0.5*sum(ans)

        def ln_mu_y_on_mu(u):
            if self.loss is not None:
                ans1 = -1*self.loss(u)
            else:
                ans1=0

            wf = []
            for j in range(len(self.clusters)):
                wf.append(np.log(self.weights[j])+ln_f(u, j))
            ans2 = logsumexp(wf)
            return ans1-ans2
        acc_rate = np.exp(min(0, ln_mu_y_on_mu(v)-ln_mu_y_on_mu(u)))
        return acc_rate
    
    def a_preserve_GMM(self,u,v):
        self.loss=None
        return self.compute_acc_rate(v,u)   
        

    def compare_GMM_with_posterior(self, len_chain=None):
        # If GMM is a good estimation of mu_y, the acc_rate will not be too low
        # when we sample from posterior using GMM as proposal
                
        if len_chain is None:
            len_chain=self.num_samples
        
        u = self.generate_sample()

        n_acc = 0
        for i in range(len_chain):
            v = self.generate_sample()
            acc_rate = self.compute_acc_rate(u, v)

            if np.random.uniform() < acc_rate:
                n_acc += 1
                u=np.array(v)


            if i > 1 and i % 100 == 0:
                print(str(i), "step is completed. Acc rate = ", str(n_acc/i))
        print("Acc rate of GMM = ", str(n_acc/len_chain))


    def compute_project_dim(self):
        # Note that if we want to use BIC method to decide the num of gauss, then there
        # should be num_samples > sample_dim. In practice, the num where we cut the eigval
        # is also important: too small num_samples or too high cut_num will lead to bad clusters
        # i.e. the codes perform badly in test data generated by 3 gauss prior
        # num_samples=3000, cut_num=0.999 is fine
        # num_samples=300, cut_num=0.99 is fine
        if self.prior.has_eigensystem is False:
            self.prior.eval_eigensystem(200)
        self.alpha, self.e = self.prior.eigval, self.prior.eigvec
        
        
        # i.e. self.samples_high_dim are samples in 200-dim eigen space
        # alpha and e are the names of eigenpairs in [1], note that e[:,i] is the i-th eigvec
        # print(self.e.shape) # [2601,10]
        tem = 0
        for i in range(len(self.alpha)):
            tem += self.alpha[i]
            if tem > 0.999999*sum(self.alpha):
                break
        if i <50:
            i=50 # i should not be too small
        i=20
        self.alpha = self.alpha[0:i]
        self.e = self.e[:, 0:i]
        
        def basis1(x):
            return np.cos(np.pi*x[0])
        tem_fun=fem.Function(self.prior.Vh)
        tem_fun.interpolate(basis1)
        tem_vec=np.array(tem_fun.x.array[:])
        self.e[:,1]=tem_vec/np.sqrt(tem_vec@ self.prior.M @tem_vec)
        def basis2(x):
            return np.cos(np.pi*x[1])
        tem_fun.interpolate(basis2)
        tem_vec=np.array(tem_fun.x.array[:])
        self.e[:,2]=tem_vec/np.sqrt(tem_vec@ self.prior.M @tem_vec)

        
        
        if self.comm is not None and self.comm.rank ==0:  
            print("The samples will be project into a " +
                  str(i)+"-dim space, the eigen value are:")
            np.set_printoptions(linewidth=200)
            print(self.alpha)
            np.set_printoptions(formatter=None) 
        return i

    def estimate_paras(self):
        x, h, m, cov = [], [], [], []  # m,cov are mean and covariance of gauss
        # x and h are paras in [1]
        # cluster.shape=[1000,10], num of samples * project dim
        for cluster in self.clusters:
            Nj = len(cluster)  # num of members in this cluster
            xj, hj = [], []
            mj, covj = [], []

            for k in range(cluster.shape[1]):
                data = cluster[:, k]
                
                mjk = np.mean(data)
                xjk = mjk/self.prior.eigval[k]

                xj.append(xjk)
                mj.append(mjk)

                covjk = data@data/Nj-mjk**2
                hjk = 1/covjk-1/self.prior.eigval[k]

                hj.append(hjk)
                covj.append(covjk)

            x.append(xj)
            h.append(hj)
            m.append(mj)
            cov.append(covj)

        return x, h, m, cov

    def generate_sample(self, n=1):
        samples = []
        for i in range(n):
            idx = np.random.choice(len(self.clusters), p=self.weights)
            mean = np.array(self.m[idx])
            cov = np.array(self.cov[idx])

            tem = np.random.normal(0, 1, cov.shape[0])
            # print("cov=",cov)
            sample_project = mean+tem*np.sqrt(cov)
            sample = sample_project @ self.e.T

            samples.append(sample)
        if n == 1:
            return np.array(samples[0])
        else:
            return np.array(samples)

    def project(self, f, project_dim=None):
        M = self.prior.M
        if len(f.shape)==2:  
            assert f.shape[1] == M.shape[0], "dim dismatch! f is " + \
                str(f.shape)+", but M is "+str(M.shape)
        if len(f.shape) ==1:
            assert f.shape[0]== M.shape[0], "dim dismatch! f is " + \
                str(f.shape)+", but M is "+str(M.shape)
        # project samples from R^(dim_mesh) to low-dim space

        return f @ self.prior.M @ self.e

    def clustering(self, samples):
        # this function clusters the samples, and the result is self.clusters, a list.
        # self.clusters[i] is the i-th cluster
        # Moreover the weights in GMM are also returned, and the means and covs will be computed
        # as ref[1] in self.estimate_paras()
        def gmm_bic_score(estimator, samples):  # gmm: Gauss Mixture Model
            return -estimator.bic(samples)

        param_grid = {
            "n_components": range(1, self.max_num_GM),#15),
            # options: tied has the same cov, full has different covs
            "covariance_type": ["diag"],
        }

        grid_search = GridSearchCV(GaussianMixture(), param_grid=param_grid,
                                   scoring=gmm_bic_score)
        grid_search.fit(samples)
        labels = grid_search.best_estimator_.predict(self.samples)
        self.labels=labels
        weights = grid_search.best_estimator_.weights_

        result = grid_search.cv_results_


        idx = np.argmax(result["mean_test_score"])  # i.e. argmin(BIC)
        num_clusters = result["param_n_components"][idx]
        if self.comm is not None and self.comm.rank==0:
            print("The samples are clustered into " +
              str(num_clusters)+" partitions")
        clusters = []
        self.clusters_origin=[]
        for i in range(num_clusters):
            idx = labels == i
            clusters.append(self.samples[idx])#(self.samples_high_dim[idx]) # 
            # self.clusters_origin.append(self.samples_origin[idx])
        return weights, clusters



class Mix_Gauss_pCN_proposal:
    def __init__(self, model, step, GMM):
        self.GMM = GMM
        self.prior = {
            'eig_val': GMM.alpha,
            'eig_vecs': GMM.e  # eigvecs[:,i] is the i-th vector
        }
        self.phis = self.prior['eig_vecs']  # phis[:,i] is the i-th vector
        self.GMM_paras = {
            # m[j] is mean of the j-th component gauss
            'means': np.array(GMM.m),
            'x': np.array(GMM.x),  # x=mean/alpha
            # covs[j] is cov of the j-th component gauss
            'covs': np.array(GMM.cov),
            # covs[j] are also eigvals of the j-th cov operator
            'eig_vals': np.array(GMM.cov),
            'h': np.array(GMM.h),
            'weights': np.array(GMM.weights)
        }
        self.dim = GMM.project_dim

        self.J = len(GMM.weights)  # num of gauss in GMM
        self.update_step(step)

        self.M = GMM.prior.M
        self.loss = model.loss_res
        self.model = model

    def eig_pairs_of_muduPudv(self, j, k):
        # compute the (j,k)-th gauss in mu(du)P(u,dv), which is a J^2 component GMM

        lambda_j = self.GMM_paras['eig_vals'][j]
        lambda_k = self.GMM_paras['eig_vals'][k]
        t_plus, t_minus = self.compute_t(lambda_j, lambda_k)
        eigvals_plus = (1+self.gamma*t_plus) * lambda_k
        eigvals_minus = (1+self.gamma*t_minus) * lambda_k

        eigvecs_plus = []
        eigvecs_minus = []

        for i in range(self.dim):
            phi = self.phis[:, i]
            eigvecs_plus. append(np.hstack((phi, t_plus[i] * phi)))
            eigvecs_minus.append(np.hstack((phi, t_minus[i] * phi)))

        eig_vals = np.hstack((eigvals_plus, eigvals_minus))
        eig_vecs = np.array(eigvecs_plus+eigvecs_minus).T

        for i in range(self.dim*2):
            tem = eig_vecs[:, i]
            tem_norm = np.sqrt(self.inner_product_H2(tem, tem))
            eig_vecs[:, i] /= tem_norm
        return eig_vals, eig_vecs

    def eig_pairs_of_mudvPvdu(self, j, k):
        # compute the (j,k)-th gauss in mu(dv)P(v,du), which is a J^2 component GMM

        lambda_j = self.GMM_paras['eig_vals'][j]
        lambda_k = self.GMM_paras['eig_vals'][k]
        t_plus, t_minus = self.compute_t(lambda_j, lambda_k)
        eigvals_plus = (1+self.gamma*t_plus) * lambda_k
        eigvals_minus = (1+self.gamma*t_minus) * lambda_k

        eigvecs_plus = []
        eigvecs_minus = []

        for i in range(self.dim):
            phi = self.phis[:, i]
            eigvecs_plus. append(np.hstack((t_plus[i] * phi, phi)))
            eigvecs_minus.append(np.hstack((t_minus[i] * phi, phi)))

        eig_vals = np.hstack((eigvals_plus, eigvals_minus))
        eig_vecs = np.array(eigvecs_plus+eigvecs_minus).T

        for i in range(self.dim*2):
            tem = eig_vecs[:, i]
            tem_norm = np.sqrt(self.inner_product_H2(tem, tem))
            eig_vecs[:, i] /= tem_norm
        return eig_vals, eig_vecs

    def a(self, u, v):
        Phi_u, Phi_v = self.loss(u), self.loss(v)
        LSE2=self.LSEj2k2([u,Phi_u],[v,Phi_v])
        LSE1=self.LSEj1k1([u,Phi_u],[v,Phi_v])
        Delta_LSE=LSE2-LSE1
        return np.exp(min(0,Delta_LSE))

    
    def LSEj2k2(self,u_info,v_info):
        u,Phi_u=u_info
        v,Phi_v=v_info
        idx_set = np.array(range(len(self.GMM_paras['weights'])))
        w = self.GMM_paras['weights']
        lambdas = self.GMM_paras['eig_vals']
        seq=[]
        for j2 in idx_set:
            for k2 in idx_set:
                tem=np.log(w[j2]*w[k2])\
                    -0.5*sum(np.log(lambdas[j2]))\
                    -0.5*sum(np.log(lambdas[k2]))\
                    -Phi_v-0.5*self.CM_inner_product_j2k2(u, v, j2, k2)
                seq.append(tem)
        return logsumexp(np.array(seq))
    
    def LSEj1k1(self,u_info,v_info):
        u,Phi_u=u_info
        v,Phi_v=v_info
        idx_set = np.array(range(len(self.GMM_paras['weights'])))
        w = self.GMM_paras['weights']
        lambdas = self.GMM_paras['eig_vals']
        seq=[]
        for j1 in idx_set:
            for k1 in idx_set:
                tem=np.log(w[j1]*w[k1])\
                    -0.5*sum(np.log(lambdas[j1]))\
                    -0.5*sum(np.log(lambdas[k1]))\
                    -Phi_u-0.5*self.CM_inner_product_j1k1(u, v, j1, k1)
                seq.append(tem)
        return logsumexp(np.array(seq))
        
    def CM_inner_product_j2k2(self, u, v, j2, k2):
        mk2 = self.GMM_paras['means'][k2]
        mj2 = self.GMM_paras['means'][j2]
        m1, m2 = (1-self.gamma)*mj2+self.gamma*mk2, mk2
        m1 = self.phis @ m1  # project to the original hilbert space, i.e. R^2601
        m2 = self.phis @ m2
        m = np.hstack((m1, m2))

        eig_vals, eig_vecs = self.eig_pairs_of_mudvPvdu(j2, k2)
        coord = (np.hstack((u, v))-m)@eig_vecs
        tem = coord/np.sqrt(eig_vals)
        return tem@tem
    
    def CM_inner_product_j1k1(self, u, v, j1, k1):
        mk1 = self.GMM_paras['means'][k1]
        mj1 = self.GMM_paras['means'][j1]
        m1,m2 = mk1, (1-self.gamma)*mj1+self.gamma*mk1
        m1 = self.phis @ m1  # project to the original hilbert space, i.e. R^2601
        m2 = self.phis @ m2
        m = np.hstack((m1, m2))
        
        eig_vals, eig_vecs = self.eig_pairs_of_muduPudv(j1, k1)
        coord = (np.hstack((u, v))-m)@eig_vecs
        tem = coord/np.sqrt(eig_vals)
        return tem@tem
    
    
    def transpose(self, len_chain, u_init=None,u_true=None):
        if u_init is None:
            u = self.GMM.generate_sample()
        else:
            u = u_init
        self.len_chain = len_chain
        self.chain = []
        m = self.GMM_paras['weights']@self.GMM_paras['means']@self.phis.T
        n_acc = 0
        for i in range(self.len_chain):
            v = self.gamma * u+(1-self.gamma)*m+self.beta * \
                self.GMM.generate_sample()

            if np.random.uniform() < self.a(u, v):
                u = np.array(v)
                n_acc += 1

            self.chain.append(np.array(u))
            tem_fun = fem.Function(self.model.function_space)
            tem_fun.x.array[:] = np.array(u)
            if i%10==0 and u_true is not None:
                print(error_compare(tem_fun,u_true)[0])

        acc_rate = n_acc/self.len_chain
        print(acc_rate)
        self.chain = np.array(self.chain)

    def inner_product_H2(self, x, y):
        # inner product in H*H, thus x and y must be in H*H, i.e. 2n-dim vector
        x0, x1 = self.my_split(x)
        y0, y1 = self.my_split(y)

        assert x0.shape == y0.shape
        assert x1.shape == y1.shape
        assert x0.shape == y1.shape

        return x0@self.M @ y0+x1@self.M@y1

    def my_split(self, x):
        assert type(x) is list or type(
            x) is np.ndarray, "x must be a list or np.ndarray!"
        if type(x) is list:
            x0 = x[0]
            x1 = x[1]
        elif type(x) is np.ndarray:
            tem_len = x.shape[0]
            idx_middle = int(tem_len/2)
            x0 = x[0:idx_middle,]
            x1 = x[idx_middle:,]
        return np.array(x0), np.array(x1)

    def compute_t(self, lambda_j, lambda_k):
        l = (lambda_j-lambda_k)/lambda_k
        
        coef=self.beta**2/self.gamma/2
        
        tem1 = coef*l
        tem2 = np.sqrt(coef**2 * l**2 + 1)
        t_plus = tem1+tem2
        t_minus = tem1-tem2
        return t_plus, t_minus

    def update_step(self, step):
        self.beta = step
        self.gamma = np.sqrt(1-step*step)


