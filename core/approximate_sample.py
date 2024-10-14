#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:48:26 2022

@author: Junxiong Jia
"""

from mpi4py import MPI
import numpy as np
import scipy.sparse.linalg as spsl
import scipy.sparse as sps
from dolfinx import fem

from core.eigensystem import double_pass
from core.misc import construct_measure_matrix, Smoother
from core.optimizer import NewtonCG, GradientDescent

class GaussianApproximate:
    '''
    Ref: T. Bui-Thanh, O. Ghattas, J. Martin, G. Stadler, 
    A computational framework for infinite-dimensional Bayesian inverse problems
    part I: The linearized case, with application to global seismic inversion,
    SIAM J. Sci. Comput., 2013
    '''
    def __init__(self, model, hessian_type="accurate"):
        ## hessian_type == "accurate" or "linear_approximate"
        
        assert hasattr(model, "prior") and hasattr(model, "function_space")
        assert hasattr(model, "equ_solver") and hasattr(model, "noise")
        assert hasattr(model, "M") and hasattr(model, "S")
        assert hasattr(model, "linearized_forward_solve")
        assert hasattr(model, "linearized_adjoint_solve")
        assert hasattr(model.noise, "precision_times_param")
        assert hasattr(model.prior, "eval_MCinv")
        assert hasattr(model.prior, "eval_CMinv")
        assert hasattr(model.prior, "M") and hasattr(model.prior, "K")
        assert hasattr(model.prior, "Vh")

        self.name = "LaplaceApproximate"
        self.model = model
        fun = fem.Function(model.function_space)
        self.fun_dim = fun.x.array.shape[0]
        self.prior = model.prior
        self.equ_solver = model.equ_solver
        self.noise = model.noise
        assert sps.issparse(model.M) == True, "model.M should be a sparse matrix"
        self.M = sps.csc_matrix(model.M)
        lamped_elements = np.array(np.sum(self.M, axis=1)).flatten()
        self.M_lamped_half = sps.csc_matrix(sps.diags(np.sqrt(lamped_elements)))
        self.Minv_lamped_half = sps.csc_matrix(sps.diags(np.sqrt(1/lamped_elements)))
        assert sps.issparse(model.S) == True, "model.S should be a sparse matrix"
        self.S = model.S
        self.hessian_type = hessian_type
        
    def set_mean(self, vec):
        assert vec.shape[0] == self.fun_dim
        self.mean = np.array(vec)

    def eval_Minvhalf_times_param(self, u):
        return self.Minv_lamped_half@u

    ## Since the symmetric matrixes are needed for computing eigensystem
    ## F^{*}F = M^{-1} F^T F is not a symmetric matrix, so we multiply M.
    ## The following function actually evaluate M F^{*}F = F^T F
    def _eval_Hessian_misfit_M(self, vec):
        vec = np.squeeze(vec)
        assert vec.ndim == 1 or vec.ndim == 2
        assert vec.shape[0] == self.fun_dim
        if self.hessian_type == "linear_approximate":
            if vec.ndim == 1:
                val = self.model.linearized_forward_solve(vec)
                val = self.noise.precision_times_param(self.S@val)
                val = self.model.linearized_adjoint_solve(val, vec)
            if vec.ndim == 2:
                val = np.zeros((vec.shape[0], vec.shape[1]))
                for idx in range(vec.shape[1]):
                    val1 = self.model.linearized_forward_solve(vec[:, idx])
                    val1 = self.noise.precision_times_param(self.S@val1)
                    val1 = self.model.linearized_adjoint_solve(val1, vec[:, idx])
                    val[:, idx] = np.array(val1)
            return np.array(self.M@val).squeeze()
        elif self.hessian_type == "accurate":
            if vec.ndim == 1:
                val = self.model.eval_hessian_res(vec)
            if vec.ndim == 2:
                val = np.zeros((vec.shape[0], vec.shape[1]))
                for idx in range(vec.shape[1]):
                    val1 = self.model.eval_hessian_res(vec[:, idx])
                    val[:, idx] = np.array(val1)
            return np.array(self.M@val).squeeze()

    def eval_Hessian_misfit_M(self):
        leng = self.M.shape[0]
        self.linear_ope = spsl.LinearOperator((leng, leng), matvec=self._eval_Hessian_misfit_M)
        return self.linear_ope
    
    ## K M^{-1} K as eva_Hessian_misfit_M, we also multiply M
    def eval_prior_var_inv_M(self):
        leng = self.prior.M.shape[0]
        self.linear_ope = spsl.LinearOperator((leng, leng), matvec=self.model.prior.eval_MCinv)
        return self.linear_ope
    
    ## K^{-1} M K^{-1}
    def eval_prior_var_M(self):
        leng = self.prior.M.shape[0]
        self.linear_ope = spsl.LinearOperator((leng, leng), matvec=self.model.prior.eval_CMinv)
        return self.linear_ope
        
    def eval_eigensystem(self, num_eigval, method='double_pass',
                              oversampling_factor=20, cut_val=0.9, **kwargs):
        '''
        Calculate the eigensystem of H_{misfit} v = \lambda \Gamma^{-1} v.
        (\Gamma is the prior covariance operator)
        The related matrixes (H_{misfit} and \Gamma) are not symmetric, 
        however, the standard eigen-system computing algorithm need these matrixes 
        to be symmetric. Hence, we actually compute the following problem:
                M H_{misfit} v = \lambda M \Gamma^{-1} v

        Parameters
        ----------
        num_eigval : int
            calucalte the first num_eigval number of large eigenvalues
        method : str, optional
            double_pass and scipy_eigsh can be choien. The default is 'double_pass'.
        oversampling_factor : int, optional
            To ensure an accurate calculation of the required eigenvalues. The default is 20.
        **kwargs : TYPE
            Depends on which method is employed.

        Returns
        -------
        None.
        
        The computated eignvalue will be in a descending order.
        '''
    
        if method == 'double_pass':
            Hessian_misfit = self.eval_Hessian_misfit_M()
            prior_var_inv = self.eval_prior_var_inv_M()
            prior_var = self.eval_prior_var_M()
            rs = num_eigval + oversampling_factor
            omega = np.random.randn(self.M.shape[0], rs)
            self.eigval, self.eigvec = double_pass(
                Hessian_misfit, M=prior_var_inv, Minv=prior_var, 
                r=num_eigval, l=oversampling_factor, n=self.M.shape[0],
                cutval=cut_val
            )
        elif method == 'single_pass':
            raise NotImplementedError
        elif method == 'scipy_eigsh':
            ## The eigsh function in scipy seems much slower than the "double_pass" and 
            ## "single_pass" algorithms implemented in the package. The eigsh function 
            ## is kept as a baseline for testing our implementations. 
            Hessian_misfit = self.eval_Hessian_misfit_M()
            prior_var_inv = self.eval_prior_var_inv_M()
            prior_var = self.eval_prior_var_M()
            self.eigval, self.eigvec = spsl.eigsh(
                Hessian_misfit, M=prior_var_inv, k=num_eigval+oversampling_factor, which='LM', 
                Minv=prior_var, **kwargs
            )  #  optional parameters: maxiter=maxiter, v0=v0 (initial)
            index = self.eigval > cut_val
            if np.sum(index) == num_eigval:
                print("Warring! The eigensystem may be inaccurate!")
            self.eigval = np.flip(self.eigval[index])
            self.eigvec = np.flip(self.eigvec[:, index], axis=1)
        else:
            assert False, "method should be double_pass, scipy_eigsh"

        self.ch_eigvec = self.eigvec.copy()
        ## In the above, we actually solve the H v1 = \lambda \Gamma^{-1} v1, we saved as
        ## self.CH_eigvec and self.CH_eigval
        ## However, we actually need to solve \Gamma^{1/2} H \Gamma^{1/2} v = \lambda v
        ## Notice that v1 \neq v, we have v = \Gamma^{-1/2} v
        self.eigvec = spsl.spsolve(self.M, self.prior.K@self.eigvec)
        
    def posterior_var_times_vec(self, vec):
        dr = self.eigval/(self.eigval + 1.0)
        Dr = sps.csc_matrix(sps.diags(dr))
        val1 = spsl.spsolve(self.prior.K, self.M@vec)
        val2 = self.eigvec@(Dr@(self.eigvec.T@(self.M@val1)))
        val = self.M@(val1 - val2)
        val = spsl.spsolve(self.prior.K, val)
        return np.array(val)        
        
    def generate_sample(self):
        assert hasattr(self, "mean") and hasattr(self, "eigval") and hasattr(self, "eigvec")
        n = np.random.normal(0, 1, (self.fun_dim,))
        val1 = self.eval_Minvhalf_times_param(n)
        pr = 1.0/np.sqrt(self.eigval+1.0) - 1.0
        Pr = sps.csc_matrix(sps.diags(pr))
        val2 = self.eigvec@(Pr@(self.eigvec.T@(self.M@val1)))
        val = self.M@(val1 + val2)
        val = spsl.spsolve(self.prior.K, val)
        val = self.mean + val
        return np.array(val)
        
    def pointwise_variance_field(self, xx, yy=None):
        '''
        Calculate the pointwise variance field of the posterior measure. 
        Through a simple calculations, we can find the following formula:
            c_h(xx, yy) = \Phi(xx)^T[K^{-1}MK^{-1]} - K^{-1}MV_r D_r V_r^T M K^{-1}]\Phi(yy),
        which is actually the same as the formula (5.7) in the following paper: 
            A computational framework for infinite-dimensional Bayesian inverse problems
            part I: The linearized case, with application to global seismic inversion,
            SIAM J. Sci. Comput., 2013

        Parameters
        ----------
        xx : list
            [(x_1,y_1), \cdots, (x_N, y_N)]
        yy : list
            [(x_1,y_1), \cdots, (x_M, y_M)]

        Returns
        -------
        None.

        '''
        assert hasattr(self, "eigval") and hasattr(self, "eigvec")

        if yy is None:
            yy = np.array(xx)
        SN = construct_measure_matrix(self.prior.Vh, np.array(xx))
        SM = construct_measure_matrix(self.prior.Vh, np.array(yy))
        SM = (SM.todense()).T

        # self.luK = spsl.splu(self.prior.K.tocsc())
        # val = np.zeros((SM.shape[0], SM.shape[1]))
        # for idx in range(SM.shape[1]):
        #     val[:, idx] = self.luK.solve(np.array(SM[:, idx])).squeeze()
        val = spsl.spsolve(self.prior.K.tocsc(), SM)
        dr = self.eigval/(self.eigval + 1.0)
        Dr = sps.csc_matrix(sps.diags(dr))
        val1 = self.eigvec@(Dr@(self.eigvec.T@(self.M@val)))
        val = self.M@(val - val1)
        # for idx in range(val.shape[1]):
        #     val[:, idx] = self.luK.solve(np.array(val[:, idx])).squeeze()
        val = spsl.spsolve(self.prior.K.tocsc(), val)
        val = SN@val        
        
        if sps.issparse(val):
            val = val.todense()
        
        return np.array(val)


class rMAP:
    '''
    The rMAP method is proposed in [1]. The current implementation has not further accelerate the
    speed for solving optimization and just use prior mean or specified initial (e.g., MAP estimate)
    as the initial guess. Only for Gaussain noise and Gaussaian prior assumptions,
    but the forward problem can be nonlinear. For theories and more details, see [1].

    Ref: [1] K. Wang, T. Bui-Thanh, O. Ghattas, A randomized maximum a posteriori method for posterior
    sampling of high dimensional nonlinear Bayesian inverse problems,
    SIAM Journal on Scientific Computing, 40(1), 2018, A142--A171.
    '''
    def __init__(self, comm, num_size, model, comm_size):
        assert hasattr(model, "num_dofs")
        assert hasattr(model, "smoother")
        assert hasattr(model, "prior")
        assert hasattr(model, "loss")
        assert hasattr(model, "equ_solver")
        assert hasattr(model, "noise")

        self.name = "rMAP"
        ## set parameters for different particles
        self.sample_size = num_size
        self.num_per_worker = self.sample_size // comm_size  ## every core needs to compute num_per_work optimization problems
        ## if the following situation happens, we make sure the number of samples
        ## larger than or equal to the required number of samples.
        assert self.sample_size >= comm_size
        self.model = model
        self.comm = comm

        self.optim_options = {
            "max_iter": [50, 50], "init_val": None, "info_optim": False,
            "cg_max": 100, "newton_method": 'bicgstab', "grad_smooth_degree": 1e-2,
            "if_normalize_dd": False
        }  ## "newton_method" == cg_my, bicgstab, cg, cgs

    def optimizing(self):
        num_iter_grad, num_iter_newton = self.optim_options["max_iter"]
        optim_method = self.optim_options["newton_method"]
        init_vec = self.optim_options["init_val"]
        info = self.optim_options["info_optim"]
        if init_vec is not None:
            assert len(init_vec) == self.model.num_dofs
        assert optim_method == "cg_my" or optim_method == "bicgstab"
        ## set optimizer
        optimizer = GradientDescent(model=self.model)
        optimizer.if_normalize_dd = self.optim_options["if_normalize_dd"]
        self.model.smoother.set_degree(self.optim_options["grad_smooth_degree"])
        if init_vec is None:
            init_vec = np.array(self.model.prior.mean_vec, np.float64)
            # init_vec = np.zeros(self.model.num_dofs)
        optimizer.re_init(init_vec)
        loss_pre = self.model.loss()[0]
        for itr in range(num_iter_grad):
            optimizer.descent_direction(self.model.smoother.smoothing)
            optimizer.step(method='armijo', show_step=False)
            if optimizer.converged == False:
                break
            loss = self.model.loss()[0]
            if info == True:
                print("iter = %2d/%d, loss = %.4f" % (itr + 1, num_iter_grad, loss))
            if np.abs(loss - loss_pre) < 1e-3 * loss:
                if info == True:
                    print("Iteration stoped at iter = %d" % itr)
                break
            loss_pre = loss

        ## Newton
        newton_cg = NewtonCG(model=self.model)
        newton_cg.if_normalize_dd = self.optim_options["if_normalize_dd"]
        init_fun = fem.Function(self.model.equ_solver.Vh)
        init_fun.x.array[:] = np.array(optimizer.mk.copy())
        newton_cg.re_init(init_fun.x.array)
        loss_pre = self.model.loss()[0]
        for itr in range(num_iter_newton):
            newton_cg.descent_direction(cg_max=50, method=optim_method)
            if info == True:
                print(newton_cg.hessian_terminate_info)
            newton_cg.step(method='armijo', show_step=False)
            if newton_cg.converged == False:
                break
            loss = self.model.loss()[0]
            if info == True:
                print("iter = %2d/%d, loss = %.4f" % (itr + 1, num_iter_newton, loss))
            if np.abs(loss - loss_pre) < 1e-3 * loss:
                if info == True:
                    print("Iteration stoped at iter = %d" % itr)
                break
            loss_pre = loss

        if num_iter_newton == 0:
            sol = optimizer.mk.copy()
        else:
            sol = newton_cg.mk.copy()

        return sol

    def sampling(self, init_vec=None):
        num_iter_grad, num_iter_newton = self.optim_options["max_iter"]
        optim_method = self.optim_options["newton_method"]
        info = self.optim_options["info_optim"]
        if init_vec is not None:
            assert len(init_vec) == self.model.num_dofs
        assert optim_method in ["cg_my", "bicgstab", "cg", "cgs"]
        sample_local = []
        for i in range(self.num_per_worker):
            if info == True:
                print("----------------------")
            ## save original mean
            prior_mean_ = self.model.prior.mean_vec.copy()
            noise_mean_ = self.model.noise.mean.copy()
            ## generate sample with zero mean
            epsilon = self.model.prior.generate_sample_zero_mean()
            theta = self.model.noise.generate_sample_zero_mean()
            ## set mean
            self.model.prior.set_mean(prior_mean_ + epsilon)
            self.model.noise.set_parameters(noise_mean_ + theta, self.model.noise.std_dev)
            ## set optimizer
            optimizer = GradientDescent(model=self.model)
            optimizer.if_normalize_dd = self.optim_options["if_normalize_dd"]
            self.model.smoother.set_degree(self.optim_options["grad_smooth_degree"])
            if init_vec is None:
                init_vec = np.array(self.model.prior.mean_vec, np.float64)
                # init_vec = np.zeros(self.model.num_dofs)
            optimizer.re_init(init_vec)
            loss_pre = self.model.loss()[0]
            for itr in range(num_iter_grad):
                optimizer.descent_direction(self.model.smoother.smoothing)
                optimizer.step(method='armijo', show_step=False)
                if optimizer.converged == False:
                    break
                loss = self.model.loss()[0]
                if info == True:
                    print("iter = %2d/%d, loss = %.4f" % (itr + 1, num_iter_grad, loss))
                if np.abs(loss - loss_pre) < 1e-3 * loss:
                    if info == True:
                        print("Iteration stoped at iter = %d" % itr)
                    break
                loss_pre = loss

            ## Newton
            newton_cg = NewtonCG(model=self.model)
            newton_cg.if_normalize_dd = self.optim_options["if_normalize_dd"]
            init_fun = fem.Function(self.model.equ_solver.Vh)
            init_fun.x.array[:] = np.array(optimizer.mk.copy())
            newton_cg.re_init(init_fun.x.array)
            loss_pre = self.model.loss()[0]
            for itr in range(num_iter_newton):
                newton_cg.descent_direction(cg_max=50, method=optim_method)
                if info == True:
                    print(newton_cg.hessian_terminate_info)
                newton_cg.step(method='armijo', show_step=False)
                if newton_cg.converged == False:
                    break
                loss = self.model.loss()[0]
                if info == True:
                    print("iter = %2d/%d, loss = %.4f" % (itr + 1, num_iter_newton, loss))
                if np.abs(loss - loss_pre) < 1e-3 * loss:
                    if info == True:
                        print("Iteration stoped at iter = %d" % itr)
                    break
                loss_pre = loss
            if num_iter_newton == 0:
                sample_local.append(optimizer.mk.copy())
            else:
                sample_local.append(newton_cg.mk.copy())
            self.model.prior.set_mean(prior_mean_.copy())
            self.model.noise.set_parameters(noise_mean_.copy(), self.model.noise.std_dev)

        sample_global = self.comm.gather(sample_local, root=0)
        if self.comm.rank == 0:
            samples = np.array(sample_global[0])  ## row is the number of solutions, cloumn is the number of dofs
            for sample_local in sample_global[1:]:
                samples = np.concatenate((samples, np.array(sample_local)), axis=0)
            # print(samples.shape)
            return samples
        else:
            return None


        













