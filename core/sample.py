#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 20:56:03 2022

@author: Junxiong Jia
"""

import os
import numpy as np
from scipy.special import logsumexp

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.approximate_sample import GaussianApproximate
from core.optimizer import GradientDescent, NewtonCG
from mpi4py import MPI
import matplotlib.pyplot as plt

###################################################################
class MCMCBase:
    def __init__(self, model, reduce_chain=None, save_path=None):
        assert hasattr(model, "prior")
        self.model = model
        self.prior = model.prior
        self.reduce_chain = reduce_chain
        self.save_path = save_path
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        self.chain = []
        self.acc_rate = 0.0
        self.index = 0

    def save_local(self):
        ## reduce_chain is an interger number, we will withdraw the reduce_chain number of samples
        ## to save the memory; if the path is specified, we save the chain.
        if self.reduce_chain is not None:
            if self.save_path is not None:
                if np.int64(len(self.chain)) == np.int64(self.reduce_chain):
                    np.save(self.save_path / ('sample_' + str(np.int64(self.index))), self.chain)
                    # tmp = self.chain[-1].copy()
                    # self.chain = [tmp]
                    self.chain = []
                    self.index += 1
            elif self.save_path is None:
                if np.int64(len(self.chain)) == np.int64(self.reduce_chain):
                    # tmp = self.chain[-1].copy()
                    # self.chain = [tmp]
                    self.chain = []
                    self.index += 1  ## here the index has not meaning

    def save_all(self):
        if self.save_path is not None:
            if self.reduce_chain is None:
                np.save(self.save_path / ('samples_all', self.chain))
        # else:
        #     print('\033[1;31m', end='')
        #     print("Not specify the save_path!")
        #     print('\033[0m', end='')

    def sampling(self, len_chain, callback=None, u0=None, index=None, **kwargs):
        raise NotImplementedError


class VanillaMCMC(MCMCBase):
    '''
    Ref: S. L. Cotter, G. O. Roberts, A. M. Stuart and D. White,
    MCMC Methods for Functions: Modifying Old Algorithms to Make Them Faster, Statistical Science, 2013.
    See Subsection 4.2 of the above article.
    '''
    def __init__(self, model, beta, reduce_chain=None, save_path=None):
        super().__init__(model, reduce_chain, save_path)
        self.M, self.K = model.prior.M, model.prior.K
        self.loss = model.loss_res
        self.dim = model.num_dofs
        assert hasattr(model.prior, "eval_CM_inner")
        self.prior = model.prior

        self.beta = beta
        tmp = np.sqrt(1 - beta**2)
        self.dt = (2 - 2*tmp)/(1 + tmp)

    def rho(self, x_info, y_info):
        val = 0.5*self.prior.eval_CM_inner(x_info[0])
        # print(val)
        return x_info[1] + val

    def proposal(self, x_info):
        # coef = np.sqrt(2*self.dt)
        ans1 = x_info[0]
        ans2 = self.prior.generate_sample()
        return ans1 + self.beta * ans2

    def sampling(self, len_chain=1e5, callback=None, u0=None, index=None):
        if u0 is None:
            x = self.prior.generate_sample()
        else:
            x = u0.copy()
        assert x.shape[0] == self.dim
        assert x.ndim == 1
        self.chain = [x]
        acc_num = 0
        ## index is the block of the chain saved on the hard disk
        if index == None:
            self.index = 0
        else:
            self.index = index
        x_info = [x, self.loss(x)]
        i = 1
        while i <= len_chain:
            y = self.proposal(x_info)
            y_info = [y, self.loss(y)]
            tem = self.rho(x_info, y_info) - self.rho(y_info, x_info)
            tem_acc = np.exp(min(0, tem))
            # print(tem_acc)
            if np.random.uniform() < tem_acc:
                x_info = y_info
                acc_num += 1
            self.acc_rate = acc_num / i
            self.chain.append(x_info[0])
            i += 1
            self.save_local()
            if callback is not None:
                callback([x_info, i, self.acc_rate])

        self.save_all()

class pCN(MCMCBase):
    '''
    M. Dashti, A. M. Stuart, The Bayesian Approch to Inverse Problems,
    Hankbook of Uncertainty Quantification, 2017 [Sections 5.1 and 5.2]
    '''
    def __init__(self, model, beta, reduce_chain=None, save_path=None):
        super().__init__(model, reduce_chain, save_path)
        self.M, self.K = model.prior.M, model.prior.K
        self.loss = model.loss_res
        self.dim = model.num_dofs

        tmp = np.sqrt(1 - beta**2)
        self.dt = (2 - 2*tmp)/(1 + tmp)
        self.beta = beta

    def rho(self, x_info, y_info):
        return x_info[1]

    def proposal(self, x_info):
        # dt = self.dt
        coef1 = np.sqrt(1-self.beta*self.beta)
        ans1 = x_info[0]
        ans2 = self.prior.generate_sample()
        return coef1 * ans1 + self.beta * ans2

    def sampling(self, len_chain=1e5, callback=None, u0=None, index=None):
        if u0 is None:
            x = self.prior.generate_sample()
        else:
            x = u0.copy()
        assert x.shape[0] == self.dim
        assert x.ndim == 1
        data_info = np.finfo(x[0].dtype)
        max_value = data_info.max
        min_value = data_info.min
        self.chain = [x]
        acc_num = 0
        ## index is the block of the chain saved on the hard disk
        if index == None:
            self.index = 0
        else:
            self.index = index
        x_loss = self.loss(x)
        if np.isnan(x_loss):
            x_loss = max_value
        x_info = [x, x_loss]
        i = 1
        while i <= len_chain:
            y = self.proposal(x_info)
            y_loss = self.loss(y)
            if np.isnan(y_loss):
                y_loss = max_value
            y_info = [y, y_loss]
            tem = self.rho(x_info, y_info) - self.rho(y_info, x_info)
            tem_acc = np.exp(min(0, tem))
            # print(y_loss, x_loss)
            if np.random.uniform() < tem_acc:
                x_info = [y_info[0], y_info[1]]
                acc_num += 1
            self.acc_rate = acc_num / i
            self.chain.append(x_info[0])
            i += 1
            self.save_local()
            if callback is not None:
                callback([x_info, i, self.acc_rate])

        self.save_all()

class minuspCN(MCMCBase):
    '''
    M. Dashti, A. M. Stuart, The Bayesian Approch to Inverse Problems,
    Hankbook of Uncertainty Quantification, 2017 [Sections 5.1 and 5.2]
    '''
    def __init__(self, model, beta, reduce_chain=None, save_path=None,mean=None):
        super().__init__(model, reduce_chain, save_path)
        self.M, self.K = model.prior.M, model.prior.K
        self.loss = model.loss_res
        self.dim = model.num_dofs

        tmp = np.sqrt(1 - beta**2)
        self.dt = (2 - 2*tmp)/(1 + tmp)
        self.beta = beta
        if mean is None:
            self.mean=0
        else:
            self.mean=mean

    def rho(self, x_info, y_info):
        return x_info[1]

    def proposal(self, x_info):
        # dt = self.dt
        coef1 = np.sqrt(1-self.beta*self.beta)
        ans1 = x_info[0]
        ans2 = self.prior.generate_sample()
        return self.mean-coef1 * (ans1-self.mean) + self.beta * ans2

    def sampling(self, len_chain=1e5, callback=None, u0=None, index=None):
        if u0 is None:
            x = self.prior.generate_sample()
        else:
            x = u0.copy()
        assert x.shape[0] == self.dim
        assert x.ndim == 1
        data_info = np.finfo(x[0].dtype)
        max_value = data_info.max
        min_value = data_info.min
        self.chain = [x]
        acc_num = 0
        ## index is the block of the chain saved on the hard disk
        if index == None:
            self.index = 0
        else:
            self.index = index
        x_loss = self.loss(x)
        if np.isnan(x_loss):
            x_loss = max_value
        x_info = [x, x_loss]
        i = 1
        while i <= len_chain:
            y = self.proposal(x_info)
            y_loss = self.loss(y)
            if np.isnan(y_loss):
                y_loss = max_value
            y_info = [y, y_loss]
            tem = self.rho(x_info, y_info) - self.rho(y_info, x_info)
            tem_acc = np.exp(min(0, tem))
            # print(self.rho(x_info, y_info), self.rho(y_info, x_info), y_loss, x_loss)
            if np.random.uniform() < tem_acc:
                x_info = [y_info[0], y_info[1]]
                acc_num += 1
            self.acc_rate = acc_num / i
            self.chain.append(x_info[0])
            i += 1
            self.save_local()
            if callback is not None:
                callback([x_info, i, self.acc_rate])

        self.save_all()


class pCNL(MCMCBase):
    '''
    Ref: S. L. Cotter, G. O. Roberts, A. M. Stuart and D. White,
    MCMC Methods for Functions: Modifying Old Algorithms to Make Them Faster, Statistical Science, 2013.

    Remark: The current program may not work well, which may contains some unknown buggs.
    '''
    def __init__(self, model, beta, grad_smooth_degree=1e-2, reduce_chain=None, save_path=None):
        super().__init__(model, reduce_chain, save_path)
        self.model.smoother.set_degree(grad_smooth_degree)  ## prepare for evaluating optimization
        self.prior = model.prior
        self.M, self.K = model.prior.M, model.prior.K
        self.loss = model.loss_res
        self.grad = model.eval_grad_res
        self.dim = model.num_dofs

        tmp = np.sqrt(1 - beta ** 2)
        self.dt = (2 - 2 * tmp) / (1 + tmp)

    def rho(self, x_info, y_info):
        x, grad_x, loss_x = x_info
        y = y_info[0]
        coef1, coef2, coef3, coef4 = 1, 1/2, self.dt/4, self.dt/4
        ans1 = loss_x
        ans2 = grad_x@self.M@(y-x)
        ans3 = grad_x@self.M@(x+y)
        ans4 = self.prior.eval_C(grad_x)@(self.M@grad_x)
        return coef1*ans1+coef2*ans2+coef3*ans3+coef4*ans4

    def proposal(self, x_info):
        dt = self.dt
        x, grad_x, loss_x = x_info

        coef1 = (2-dt)/(2+dt)
        coef2 = -2*dt/(2+dt)
        coef3 = np.sqrt(8*dt)/(2+dt)
        ans1 = x
        ans2 = self.prior.eval_C(grad_x)
        ans3 = self.prior.generate_sample()
        return coef1*ans1+coef2*ans2+coef3*ans3

    def sampling(self, len_chain=1e5, callback=None, u0=None, index=None):
        if u0 is None:
            x = self.prior.generate_sample()
        else:
            x = u0.copy()
        assert x.shape[0] == self.dim
        assert x.ndim == 1
        data_info = np.finfo(x[0].dtype)
        max_value = data_info.max
        min_value = data_info.min
        self.chain = [x]
        acc_num = 0
        ## index is the block of the chain saved on the hard disk
        if index == None:
            self.index = 0
        else:
            self.index = index
        grad_x = self.grad(x)
        grad_x = self.model.smoother.smoothing(grad_x)
        x_loss = self.loss()
        if np.isnan(x_loss):
            x_loss = max_value
        x_info = [x, grad_x, x_loss]
        i = 1
        while i <= len_chain:
            y = self.proposal(x_info)
            grad_y = self.grad(y)
            grad_y = self.model.smoother.smoothing(grad_y)
            y_loss = self.loss()
            if np.isnan(y_loss):
                y_loss = max_value
            y_info = [y, grad_y, y_loss]
            tem = self.rho(x_info, y_info) - self.rho(y_info, x_info)
            tem_acc = np.exp(min(0, tem))
            # print(tem_acc)
            if np.random.uniform() < tem_acc:
                x_info = [y_info[0], y_info[1], y_info[2]]
                acc_num += 1
            self.acc_rate = acc_num / i
            self.chain.append(x_info[0])
            i += 1
            self.save_local()
            if callback is not None:
                callback([x_info, i, self.acc_rate])

        self.save_all()


class Newton_pCNL(MCMCBase):
    '''
    Only valid for Gaussian prior measure

    Remark: The self.mode should be set to be "map_hessian" for the current program, for other two "init_hessian" and
    "every_step_hessian", the eigensystem cannot be calculated correctly or the gradient term will be blow up to np.nan.
    How to solve these problems and some insights of these problems should be further considered.
    For the current program, if we meet these two problems, we reduce the algorithm to be pCN.
    '''
    def __init__(self, model, dt, beta=0.01, reduce_chain=None, save_path=None):
        super().__init__(model, reduce_chain, save_path)
        assert hasattr(model.prior, 'M') and hasattr(model.prior, 'K')
        assert hasattr(model.prior, 'eval_sqrtM')
        assert hasattr(model.prior, 'eval_sqrtC')
        assert hasattr(model.prior, 'eval_C')
        assert hasattr(model.prior, 'eval_Cinv')

        self.M, self.K = model.prior.M, model.prior.K
        self.loss = model.loss_res
        self.grad = model.eval_grad_res
        self.dim = model.num_dofs

        self.dt, self.a = dt, 1 + dt / 2
        tmp = np.sqrt(1 - beta ** 2)
        self.dt_pcn = (2 - 2 * tmp) / (1 + tmp)

        self.optim_options = {
            "max_iter": [50, 50], "init_val": None, "info_optim": False,
            "cg_max": 100, "newton_method": 'bicgstab', "grad_smooth_degree": 1e-2
        }  ## "newton_method" == cg_my, bicgstab, cg, cgs
        self.model.smoother.set_degree(self.optim_options["grad_smooth_degree"])  ## prepare for evaluating optimization
        self.eigensystem_optims = {
            "cut_val": 0.1, "method": "scipy_eigsh", "num_eigval": 30,
            "hessian_type": "linear_approximate", "oversampling_factor": 20
        }  ## "method" == scipy_eigsh, double_pass; "hessian_type" == accurate, linear_approximate
        self.mode = "every_step_hessian"
        ## model == every_step_hessian, map_hessian, init_hessian
        self.eval_eigensystem_iter_max = 10

        self.if_grad = True
        self.if_eigensystem = True

    def optimizing(self, optimizer, max_iter=50, init_val=None, info=True, **kwargs):
        if init_val is None:
            init_val = np.zeros(self.dim)
        assert init_val.shape[0] == self.dim
        assert init_val.ndim == 1
        data_info = np.finfo(x[0].dtype)
        max_value = data_info.max
        min_value = data_info.min

        optimizer.re_init(init_val)

        loss_pre = self.model.loss()[0]
        if info == True:
            print("Init loss: ", loss_pre)
        for itr in range(max_iter):
            optimizer.descent_direction(**kwargs)
            optimizer.step(method='armijo', show_step=False)
            if optimizer.converged == False:
                break
            loss = self.model.loss()[0]
            if info == True:
                print("iter = %2d/%d, loss = %.4f" % (itr + 1, max_iter, loss))
            if np.abs(loss - loss_pre) < 1e-3 * loss:
                if info == True:
                    print("Iteration stoped at iter = %d" % itr)
                break
            loss_pre = loss
        return optimizer.mk.copy()

    def eval_map(self):
        self.model.smoother.set_degree(self.optim_options["grad_smooth_degree"])  ## prepare for evaluating optimization
        optimizer = GradientDescent(model=self.model)
        estimated_param = self.optimizing(
            optimizer, max_iter=self.optim_options["max_iter"][0], init_val=self.optim_options["init_val"],
            info=self.optim_options["info_optim"], smoothing=self.model.smoother.smoothing
        )
        optimizer = NewtonCG(model=self.model)
        estimated_param = self.optimizing(
            optimizer, max_iter=self.optim_options["max_iter"][1], init_val=estimated_param,
            info=self.optim_options["info_optim"], cg_max=self.optim_options["cg_max"],
            method=self.optim_options["newton_method"]
        )
        self.map_estimate = estimated_param

    def eval_eigsystem(self, param=None):
        if param is None:
            self.model.update_param(self.map_estimate, update_sol=True)
        else:
            assert param.ndim == 1 and param.shape[0] == self.model.num_dofs
            self.model.update_param(param, update_sol=True)
        gaussian_approximate = GaussianApproximate(self.model, hessian_type=self.eigensystem_optims["hessian_type"])
        cut_val = self.eigensystem_optims["cut_val"]
        gaussian_approximate.eval_eigensystem(num_eigval=self.eigensystem_optims["num_eigval"],
                                              method=self.eigensystem_optims["method"],
                                              oversampling_factor=self.eigensystem_optims["oversampling_factor"],
                                              cut_val=cut_val)
        iter_num = 1
        if len(gaussian_approximate.eigval) == 0:
            self.if_eigensystem = False
        else:
            self.if_eigensystem = True
        # print(gaussian_approximate.eigval)
        if self.if_eigensystem is True:
            while iter_num <= self.eval_eigensystem_iter_max:
                if gaussian_approximate.eigval[-1] >= 1:
                    cut_val += 10
                    gaussian_approximate.eval_eigensystem(num_eigval=self.eigensystem_optims["num_eigval"],
                                                          method=self.eigensystem_optims["method"],
                                                          oversampling_factor=self.eigensystem_optims["oversampling_factor"],
                                                          cut_val=cut_val)
                iter_num += 1

            ## C@H@eig_vecs = eig_vecs@eig_vals, here C is the prior covariance operator, H is the Hessian operator
            ## The eig_vecs and eig_vals are actually calculated by H@eig_vecs = C^{-1}@eig_vecs@eig_vals
            ## For discrete problem, we may need to make H and C^{-1} be symmetric matrixes, so we usually
            ## times M (mass matrix) on both sides.

            # self.eigvals, self.eigvecs = gaussian_approximate.eigval, gaussian_approximate.eigvec
            # self.ch_eigvecs = self.prior.eval_sqrtC(self.eigvecs)
            self.eigvals, self.ch_eigvecs = gaussian_approximate.eigval, gaussian_approximate.ch_eigvec
            self.eigvecs = self.prior.eval_sqrtCinv(self.ch_eigvecs)
            ## The above two methods seem to be identical, but the actual computational results
            ## seem to be different, which may need more understanding.
            # print(gaussian_approximate.eigval[-1])
            # assert self.eigvals[-1] < 1, "the smallest eigenvalue should be less than 1"

            self.num_eigvals = len(self.eigvals)
            self.eigvecs_lozenge = (self.eigvecs.T) @ self.M
            self.ch_eigvecs_lozenge = (self.ch_eigvecs.T) @ self.M

            # Da is used in proposal part1, 3, and D_proposal_part2 is used is part2
            self.Da = np.diag(self.eigvals / (self.eigvals + self.a))
            self.D_proposal_part2 = np.diag(self.a * np.sqrt(self.eigvals + 1) / (self.a + self.eigvals) - 1)
            self.D1 = np.diag(self.eigvals / (self.eigvals + 1))  # used in rho(x) to compute accept rate

    def preC(self, x):
        ## preconditioner, i.e. (C^-1+H)^-1
        return self.prior.eval_C(x) - self.ch_eigvecs @ (self.D1 @ (self.ch_eigvecs_lozenge @ x))

    def preC_Cinv(self, x):
        # preconditioner @(C^-1), i.e., (C^-1+H)^-1 C^{-1}
        return x - self.ch_eigvecs @ (self.D1 @ (self.ch_eigvecs_lozenge @ self.prior.eval_Cinv(x)))

    def proposal_part1(self, x):
        ##
        ans1 = (1 - self.dt / self.a) * x
        tmp1 = self.prior.eval_sqrtCinv(x)
        ans2 = self.dt / self.a * (self.Da @ (self.eigvecs_lozenge @ tmp1))
        return [ans1, ans2]

    def proposal_part2(self, n):
        ##
        coef = np.sqrt(2 * self.dt) / self.a
        tmp1 = self.prior.eval_sqrtM(n)
        ans1 = self.D_proposal_part2 @ (self.eigvecs.T @ tmp1)
        ans2 = self.prior.eval_sqrtCsqrtMinv(n)
        return [coef*ans1, coef*ans2]

    def proposal_part3(self, grad_x):
        ##
        coef = self.dt/self.a
        ans1 = self.Da @ (self.ch_eigvecs_lozenge @ grad_x)
        ans2 = -1 * self.prior.eval_C(grad_x)
        return [coef*ans1, coef*ans2]

    def proposal(self, x_info):
        n = np.random.normal(0, 1, self.dim)
        x, grad_x, loss_x = x_info
        ans11, ans12 = self.proposal_part1(x)
        ans21, ans22 = self.proposal_part2(n)
        ans31, ans32 = self.proposal_part3(grad_x)
        ## If the grad makes ans31 and ans32 to be np.nan, we drop the gradient term.
        ## The method reduces to the usual pCN algorithm.
        if np.any(np.isnan(ans31)) or np.any(np.isnan(ans32)):
            self.if_grad = False
            print('\033[1;31m', end='')
            print("Grad term is np.nan, so grad term has been deleted!")
            print('\033[0m', end='')
            return ans11 + ans22 + self.ch_eigvecs@(ans12 + ans21)
        else:
            self.if_grad = True
            return ans11 + ans22 + ans32 + self.ch_eigvecs@(ans12 + ans21 + ans31)

    def rho(self, x_info, y_info):
        x, grad_x, loss_x = x_info
        y = y_info[0]
        if self.if_grad == True:
            coef1, coef2, coef3, coef4 = 1, 1 / 2, self.dt / 4, self.dt / 4
            ans1 = loss_x
            ans2 = grad_x @ (self.M @ (y - x))
            ans3 = grad_x @ (self.M @ self.preC_Cinv(x + y))
            ans4 = self.preC(grad_x) @ self.M @ grad_x
            return coef1 * ans1 + coef2 * ans2 + coef3 * ans3 + coef4 * ans4
        elif self.if_grad == False:
            ## If the grad makes ans31 and ans32 to be np.nan, we drop the gradient term.
            ## The method reduces to the usual pCN algorithm.
            return loss_x

    def rho_pCN(self, x_info, y_info):
        return x_info[1]

    def proposal_pCN(self, x_info):
        dt = self.dt_pcn
        coef1 = (2 - dt) / (2 + dt)
        coef2 = np.sqrt(8 * dt) / (2 + dt)
        ans1 = x_info[0]
        ans2 = self.prior.generate_sample()
        return coef1 * ans1 + coef2 * ans2

    def sampling(self, len_chain=1e5, callback=None, u0=None, index=None):
        if u0 is None:
            if self.mode == "map_hessian":
                x = self.map_estimate
                data_info = np.finfo(x[0].dtype)
                max_value = data_info.max
                min_value = data_info.min
            elif self.mode == "every_step_hessian":
                x = self.prior.generate_sample()
                data_info = np.finfo(x[0].dtype)
                max_value = data_info.max
                min_value = data_info.min
            else:
                raise NotImplementedError("mode == map_hessian or every_step_hessian when u0==None")
        else:
            if self.mode == "map_hessian":
                assert u0.shape[0] == self.dim and u0.ndim == 1
                self.optim_options["init_val"] = u0
                self.eval_map()
                self.eval_eigsystem()
                if self.if_eigensystem is not True:
                    print('\033[1;31m', end='')
                    print("There is not positive eigenvalues, so pCN has been employed!")
                    print('\033[0m', end='')
                x = self.map_estimate
                data_info = np.finfo(x[0].dtype)
                max_value = data_info.max
                min_value = data_info.min
            elif self.mode == "every_step_hessian":
                x = u0.copy()
                data_info = np.finfo(x[0].dtype)
                max_value = data_info.max
                min_value = data_info.min
            elif self.mode == "init_hessian":
                assert u0.shape[0] == self.dim and u0.ndim == 1
                self.eval_eigsystem(u0)
                if self.if_eigensystem is not True:
                    print('\033[1;31m', end='')
                    print("There is not positive eigenvalues, so pCN has been employed!")
                    print('\033[0m', end='')
                x = u0.copy()
                data_info = np.finfo(x[0].dtype)
                max_value = data_info.max
                min_value = data_info.min
            else:
                raise NotImplementedError("mode == map_hessian or every_step_hessian or init_hessian")

        assert x.shape[0] == self.dim
        assert x.ndim == 1
        self.chain = [x]
        acc_num = 0
        ## index is the block of the chain saved on the hard disk
        if index == None:
            self.index = 0
        else:
            self.index = index
        if self.if_eigensystem is True:
            grad_x = self.model.smoother.smoothing(self.grad(x))
            x_loss = self.loss()
            if np.isnan(x_loss):
                x_loss = max_value
            x_info = [x, grad_x, x_loss]
            if self.mode == "every_step_hessian":
                self.eval_eigsystem(x_info[0])
                if self.if_eigensystem is not True:
                    print('\033[1;31m', end='')
                    print("There is not positive eigenvalues, so pCN has been employed!")
                    print('\033[0m', end='')
        if self.if_eigensystem is not True:
            ## if the eigensystem computation fails (no positive eigenval),
            ## we do pCN.
            x_loss = self.loss(x)
            if np.isnan(x_loss):
                x_loss = max_value
            x_info = [x, x_loss]
        i = 1
        while i <= len_chain:
            if self.if_eigensystem is True:
                y = self.proposal(x_info)
                grad_y = self.model.smoother.smoothing(self.grad(y))
                y_loss = self.loss()
                if np.isnan(y_loss):
                    y_loss = max_value
                y_info = [y, grad_y, y_loss]
                tem = self.rho(x_info, y_info) - self.rho(y_info, x_info)
                tem_acc = np.exp(min(0, tem))
            if self.if_eigensystem is not True:
                y = self.proposal_pCN(x_info)
                y_loss = self.loss(y)
                if np.isnan(y_loss):
                    y_loss = max_value
                y_info = [y, y_loss]
                tem = self.rho_pCN(x_info, y_info) - self.rho_pCN(y_info, x_info)
                tem_acc = np.exp(min(0, tem))
            # print(tem_acc)
            if np.random.uniform() < tem_acc:
                x_info = y_info
                if self.mode == "every_step_hessian":
                    self.eval_eigsystem(x_info[0])
                    if self.if_eigensystem is not True:
                        print('\033[1;31m', end='')
                        print("There is not positive eigenvalues, so pCN has been employed!")
                        print('\033[0m', end='')
                acc_num += 1
            self.acc_rate = acc_num / i
            self.chain.append(x_info[0])
            i += 1
            self.save_local()
            if callback is not None:
                callback([x_info, i, self.acc_rate])

        self.save_all()


class SMC:
    '''
    Ref: M. Dashti, A. M. Stuart, The Bayesian Approach to Inverse Problems,
    Handbook of Uncertainty Quantification, Springer, 2017 [Section 5.3]
    '''
    def __init__(self, comm, num_particles, model=None, prior=None, num_dofs=None):
        if model is not None:
            self.model = model
            self.prior=model.prior
            self.num_dofs=model.num_dofs
            assert num_particles >= comm.size

        elif prior is not None and num_dofs is not None:
            self.prior = prior
            self.num_dofs=num_dofs
        else:
            print("model is not None, or prior and num_dofs is not None!")
            
        assert num_particles >= comm.size
        self.num_particles = num_particles
        self.comm = comm
        self.phi_idx_tem=0
        
    def prepare(self, particles=None):
        # if particles is None, then particles are generated by prior samples in each kernal
        # otherwise the particles are given, and prepare() just send them to different kernals
        if self.comm.rank == 0:
            self.weights = np.ones(self.num_particles) * (1 / self.num_particles)
        else:
            self.weights = None
        
        ## decide number of each process
        self.num_per_worker = np.zeros(self.comm.size, dtype=np.int64)
        num_residual = self.num_particles % self.comm.size
        self.num_per_worker[:] = int(self.num_particles // self.comm.size)
        self.num_per_worker[:num_residual] = int(self.num_particles // self.comm.size + 1)
        
        
        self.idxs = []
        for i in range(self.comm.size):
            self.idxs.append(np.arange(self.num_per_worker[i]) + i * self.num_per_worker[i])
            
            
        if particles is None:
            particles_local = []
            for idx in range(self.num_per_worker[self.comm.rank]):
                particles_local.append(self.prior.generate_sample())
            self.particles_local = np.array(particles_local).reshape((self.num_per_worker[self.comm.rank], self.num_dofs))
        
        if particles is not None:
            self.particles=particles
            if self.comm.rank == 0:
                self.particles_local = self.particles[self.idxs[0], :]
            if self.comm.size > 1:
                for i in range(1, self.comm.size):
                    if self.comm.rank == 0:
                        particles_local_i = np.array(self.particles[self.idxs[i], :], dtype=np.float64)
                        self.comm.Send(particles_local_i, dest=i, tag=i)
                    elif self.comm.rank == i:
                        self.particles_local = np.empty((self.num_per_worker[i], self.num_dofs), dtype=np.float64)
                        self.comm.Recv(self.particles_local, source=0, tag=i)
        
        self.dtype = self.particles_local[0, 0].dtype



    def transition(self, sampler, len_chain, info_acc_rate=True, **kwargs):
        assert hasattr(sampler, 'sampling') and hasattr(sampler, 'acc_rate')
        assert hasattr(sampler, 'chain')
        self.acc_rates = np.zeros(self.num_per_worker[self.comm.rank])
        for idx in range(self.num_per_worker[self.comm.rank]):
            sampler.sampling(len_chain=len_chain, u0=self.particles_local[idx, :], **kwargs)
            if info_acc_rate == True:
                print("acc_rate = %.5f" % sampler.acc_rate)
            self.acc_rates[idx] = sampler.acc_rate
            tmp = np.array(sampler.chain[-1]).squeeze()
            assert ~np.any(np.isnan(tmp)), "particles should not contain np.nan, there must be problems when sampling!"
            # if np.any(np.isnan(tmp)):
            #     tmp = self.particles_local[idx, :]
            #     print("sampler failed to sampling!")
            self.particles_local[idx, :] = tmp.copy()

    # def gather_acc_rates(self):
    #     acc_rates_ = self.comm.gather(self.acc_rates, root=0)
    #     if self.comm.rank == 0:
    #         average_acc_rate = np.mean(acc_rates_)
    #         return average_acc_rate
    #     else:
    #         return None
    
    def gather_acc_rates(self):
        acc_rates_ = self.comm.gather(self.acc_rates, root=0)
        if self.comm.rank == 0:
            self.acc_rates = np.mean(acc_rates_)
            # print('this is gather acc_rates, acc_rates_ is ',acc_rates_)
        else:
            return None

        if self.comm.size > 1:
            for i in range(1, self.comm.size):
                if self.comm.rank == 0:              
                    acc_i=self.acc_rates
                    self.comm.Send(acc_i, dest=i, tag=i+4000)
                    
                elif self.comm.rank == i:
                    tem_acc=np.array([-1.0])
                    self.comm.Recv(tem_acc, source=0, tag=i+1000)
                    self.acc_rates=tem_acc[0]

    # def resampling(self, potential_fun):
    #     particles_ = self.comm.gather(self.particles_local, root=0)
    #     if self.comm.rank == 0:
    #         self.particles = np.array(particles_[0])  ## row is the number of solutions, cloumn is the number of dofs
    #         for tmp in particles_[1:]:
    #             self.particles = np.concatenate((self.particles, np.array(tmp)), axis=0)

    #         tmp = np.zeros(self.num_particles, dtype=np.float64)
    #         for idx in range(self.num_particles):
    #             tmp_p = potential_fun(self.particles[idx, :])
    #             ## if tmp is np.nan, it means that potential_fun(self.particles[idx, :]) is a very large number, so
    #             ## -tmp is very small number, that means tmp[idx] should be a very small number.
    #             if np.isnan(tmp_p):
    #                 tmp_p = np.finfo(self.dtype).max  ## the maximum number machine can represent.
    #             tmp[idx] = -tmp_p + np.log(self.weights[idx] + 1e-20)
    #         self.weights = np.exp(tmp - logsumexp(tmp))

    #         ## resampling (without comm.gather all of the particles to root 0, it seems complex
    #         ## for doing the resampling procedure.)
    #         idx_resample = np.random.choice(len(self.weights), len(self.weights), True, self.weights)
    #         self.particles = self.particles[idx_resample, :]
    #         self.weights = np.ones(self.num_particles) * (1 / self.num_particles)

    #         self.particles_local = self.particles[self.idxs[0], :]

    #     if self.comm.size > 1:
    #         for i in range(1, self.comm.size):
    #             if self.comm.rank == 0:
    #                 particles_local_i = np.array(self.particles[self.idxs[i], :], dtype=np.float64)
    #                 self.comm.Send(particles_local_i, dest=i, tag=i)
    #             elif self.comm.rank == i:
    #                 self.particles_local = np.empty((self.num_per_worker[i], self.model.num_dofs), dtype=np.float64)
    #                 self.comm.Recv(self.particles_local, source=0, tag=i)
    
    
    def resampling(self, potential_fun,h):
        self.phis_local = []
        for idx in range(self.num_per_worker[self.comm.rank]):
            self.phis_local.append(potential_fun(self.particles_local[idx]))
        # print(self.phis_local,self.comm.rank)
        # self.particles=self.gather_samples()
        self.gather_samples()
        phis=np.array(self.gather_phis())
        self.ESS=None
        if self.comm.rank == 0:
            tmp = -phis*h
            self.weights = np.exp(tmp - logsumexp(tmp))
            self.ESS=1/sum(self.weights**2)
            idx_resample = np.random.choice(len(self.weights), len(self.weights), True, self.weights)
            self.particles = self.particles[idx_resample, :]
            self.weights = np.ones(self.num_particles) * (1 / self.num_particles)

            self.particles_local = self.particles[self.idxs[0], :]
        if self.comm.size > 1:
            for i in range(1, self.comm.size):
                if self.comm.rank == 0:
                    particles_local_i = np.array(self.particles[self.idxs[i], :], dtype=np.float64)
                    self.comm.Send(particles_local_i, dest=i, tag=i)
                    
                    particles_i=np.array(self.particles, dtype=np.float64)
                    self.comm.Send(particles_i, dest=i, tag=i+2000)

                    ESS=self.ESS
                    self.comm.Send(ESS, dest=i, tag=i+1000)
                    
                elif self.comm.rank == i:
                    self.particles_local = np.empty((self.num_per_worker[i], self.num_dofs), dtype=np.float64)
                    self.comm.Recv(self.particles_local, source=0, tag=i)
                    
                    self.particles=np.empty((self.num_particles, self.num_dofs), dtype=np.float64)
                    self.comm.Recv(self.particles, source=0, tag=i+2000)

                    
                    tem_ESS=np.array([-1.0])
                    self.comm.Recv(tem_ESS, source=0, tag=i+1000)
                    self.ESS=tem_ESS[0]
    
    # def gather_samples(self):
    #     particles_ = self.comm.gather(self.particles_local, root=0)
    #     if self.comm.rank == 0:
    #         particles = np.array(particles_[0])  ## row is the number of solutions, cloumn is the number of dofs
    #         for tmp in particles_[1:]:
    #             particles = np.concatenate((particles, np.array(tmp)), axis=0)
    #         return particles
    #     else:
    #         return None
    
    def gather_samples(self,multiplier=None):
    # this function cannot be run in kernal 0!!!! which will cause the program to hang
        particles_ = self.comm.gather(self.particles_local, root=0)
        if self.comm.rank == 0:
            particles = np.array(particles_[0])  ## row is the number of solutions, cloumn is the number of dofs
            for tmp in particles_[1:]:
                particles = np.concatenate((particles, np.array(tmp)), axis=0)
            if multiplier is None:
                self.particles = particles
            else:
                assert multiplier.shape[0]==particles.shape[1],print('tem10 not equal n^2',multiplier.shape[0],particles.shape[1])
                self.particles = particles*multiplier
        if self.comm.size > 1:
            for i in range(1, self.comm.size):
                if self.comm.rank == 0:              
                    particles_i=np.array(self.particles, dtype=np.float64)
                    self.comm.Send(particles_i, dest=i, tag=i+3000)
                    
                elif self.comm.rank == i:
                    self.particles=np.empty((self.num_particles, self.num_dofs), dtype=np.float64)
                    self.comm.Recv(self.particles, source=0, tag=i+3000)

    def gather_phis(self):
        phis_ = self.comm.gather(self.phis_local, root=0)
        if self.comm.rank == 0:
            phis = phis_[0]  ## row is the number of solutions, cloumn is the number of dofs
            for tmp in phis_[1:]:
                phis += tmp
            # print(phis)
            return phis
        else:
            return None

    def gather_phis1(self):
        phis_ = self.comm.gather(self.phis_local1, root=0)
        if self.comm.rank == 0:
            phis = phis_[0]  ## row is the number of solutions, cloumn is the number of dofs
            for tmp in phis_[1:]:
                phis += tmp
            # print(phis)
            return phis
        else:
            return None
        
    def gather_phis2(self):
        phis_ = self.comm.gather(self.phis_local2, root=0)
        if self.comm.rank == 0:
            phis = phis_[0]  ## row is the number of solutions, cloumn is the number of dofs
            for tmp in phis_[1:]:
                phis += tmp
            # print(phis)
            return phis
        else:
            return None

class IS(MCMCBase):
    def __init__(self, model, GMM,idx_freq,idx_theta, reduce_chain=None, save_path=None):
        super().__init__(model, reduce_chain, save_path)
        self.M, self.K = model.prior.M, model.prior.K
        self.model = model
        self.dim = model.num_dofs
        
        self.prior_GMM=GMM
        
        self.idx_freq=idx_freq
        self.idx_theta=idx_theta
        
    def loss(self,u):
        return self.model.loss_res_1freq_1theta(self.idx_freq,self.idx_theta,u)
        
    def rho(self, x_info, y_info):
        return x_info[1]

    def proposal(self):
        return self.prior_GMM.generate_sample()

    def sampling(self, len_chain=1e5, callback=None, u0=None, index=None):
        if u0 is None:
            x = self.prior_GMM.generate_sample()
        else:
            x = u0.copy()
        assert x.shape[0] == self.dim
        assert x.ndim == 1
        data_info = np.finfo(x[0].dtype)
        max_value = data_info.max
        min_value = data_info.min
        self.chain = [x]
        acc_num = 0
        ## index is the block of the chain saved on the hard disk
        if index == None:
            self.index = 0
        else:
            self.index = index
        x_loss = self.loss(x)
        if np.isnan(x_loss):
            x_loss = max_value
        x_info = [x, x_loss]
        i = 1
        while i <= len_chain:
            y = self.proposal()
            y_loss = self.loss(y)
            if np.isnan(y_loss):
                y_loss = max_value
            y_info = [y, y_loss]
            tem = self.rho(x_info, y_info) - self.rho(y_info, x_info)
            tem_acc = np.exp(min(0, tem))
            # print(self.rho(x_info, y_info), self.rho(y_info, x_info), y_loss, x_loss)
            if np.random.uniform() < tem_acc:
                x_info = [y_info[0], y_info[1]]
                acc_num += 1
            self.acc_rate = acc_num / i
            self.chain.append(x_info[0])
            i += 1
            self.save_local()
            if callback is not None:
                callback([x_info, i, self.acc_rate])

        self.save_all()




class pCN_GMM(MCMCBase):
    def __init__(self, model, step, GMM, idx_freq,idx_theta,
                                         reduce_chain=None, save_path=None):
        super().__init__(model, reduce_chain, save_path)
        self.GMM = GMM
        self.idx_freq=idx_freq
        self.idx_theta=idx_theta
        
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
        self.model = model
    
    def loss(self,u):
        return self.model.loss_res_1freq_1theta(self.idx_freq,self.idx_theta,u)
    
    def a(self, u_info, v_info):
        LSE2=self.LSEj2k2(u_info,v_info)
        LSE1=self.LSEj1k1(u_info,v_info)
        Delta_LSE=LSE2-LSE1
        return np.exp(min(0,Delta_LSE))
    
    def proposal(self, u):
        tem1 = self.gamma * u
        m = self.GMM_paras['weights']@self.GMM_paras['means']@self.phis.T
        tem2 = (1-self.gamma)*m
        tem3 = self.beta * self.GMM.generate_sample()
        return tem1+tem2+tem3
    
    def sampling(self, len_chain=1e5, callback=None, u0=None, index=None):
        if u0 is None:
            x = self.GMM.generate_sample()
        else:
            x = u0.copy()
        assert x.shape[0] == self.model.num_dofs
        assert x.ndim == 1
        data_info = np.finfo(x[0].dtype)
        max_value = data_info.max
        # min_value = data_info.min
        self.chain = [x]
        acc_num = 0
        ## index is the block of the chain saved on the hard disk
        if index == None:
            self.index = 0
        else:
            self.index = index
        x_loss = self.loss(x)
        if np.isnan(x_loss):
            x_loss = max_value
        x_info = [x, x_loss]
        i = 1
        while i <= len_chain:
            y = self.proposal(x)
            y_loss = self.loss(y)
            if np.isnan(y_loss):
                y_loss = max_value
            y_info = [y, y_loss]
            tem_acc = self.a(x_info, y_info)
            if np.random.uniform() < tem_acc:
                x_info = [y_info[0], y_info[1]]
                acc_num += 1
            self.acc_rate = acc_num / i
            self.chain.append(x_info[0])
            i += 1
            self.save_local()
            if callback is not None:
                callback([x_info, i, self.acc_rate])

        self.save_all()

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



class pCN_preserve_GMM(MCMCBase):
    '''
    M. Dashti, A. M. Stuart, The Bayesian Approch to Inverse Problems,
    Hankbook of Uncertainty Quantification, 2017 [Sections 5.1 and 5.2]
    '''
    def __init__(self, model, GMM, reduce_chain=None, save_path=None):
        super().__init__(model, reduce_chain, save_path)
        self.dim = model.num_dofs
        self.GMM=GMM

    # def proposal(self, x):
    #     gamma=np.sqrt(1-self.beta*self.beta)
    #     return gamma * x + self.beta * self.prior.generate_sample()

    def sampling(self, len_chain,callback=None, u0=None, index=None):
        ## index is the block of the chain saved on the hard disk
        if index == None:
            self.index = 0
        else:
            self.index = index
        # while i <= len_chain:
        x = self.GMM.generate_sample()
        self.chain = [x]
        self.acc_rate = 1
        self.save_local()
        self.save_all()

class pCN_preserve_GMM_simple():
    def __init__(self, GMM, reduce_chain=None, save_path=None):
        self.GMM=GMM
        self.acc_rate = 0.0
        self.chain = []


    def sampling(self, len_chain,callback=None, u0=None, index=None):
        ## index is the block of the chain saved on the hard disk
        if index == None:
            self.index = 0
        else:
            self.index = index
        # while i <= len_chain:
        x = self.GMM.generate_sample()
        self.chain = [x]
        self.acc_rate = 1