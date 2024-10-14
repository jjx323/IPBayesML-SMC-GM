import numpy as np
import ufl
import dolfinx.fem.petsc as dlfp
from dolfinx import fem
import scipy.sparse.linalg as spsl
from core.misc import construct_measure_matrix, trans2scipy


class EquSolverBase:
    '''
    This class give standards for construct EquSolver class for specific equations,
    the four function name should be the same as given here.
    '''
    def __init__(self):
        ## self.Vh should be a function space generated based on the mesh
        # self.Vh = fem.FunctionSpace(mesh, ("Lagrange", 1))
        # self.sol_forward = fem.Function(self.Vh)
        # self.sol_adjoint = fem.Function(self.Vh)
        # self.sol_inc_forward = fem.Function(self.Vh)  ## solution of the incremental forward equation
        # self.sol_inc_adjoint = fem.Function(self.Vh)  ## solution of the incremental ajoint equation
        pass

    def forward_solve(self, u=None):
        raise NotImplementedError

    def adjoint_solve(self):
        raise NotImplementedError

    def inc_forward_solve(self):
        raise NotImplementedError

    def inc_adjoint_solve(self):
        raise NotImplementedError


class ModelBase:
    def __init__(self, prior, equ_solver, noise, data):
        assert hasattr(equ_solver, 'forward_solve')
        assert hasattr(prior, 'generate_sample')

        self.prior = prior
        self.equ_solver = equ_solver
        assert hasattr(equ_solver, "Vh")
        self.noise = noise
        self.coordinates = data["coordinates"]
        self.data = data["data"]
        self._init_measurement_matrix(self.coordinates)
        self.num_dofs = self.equ_solver.num_dofs
        u, v = ufl.TrialFunction(self.equ_solver.Vh), ufl.TestFunction(self.equ_solver.Vh)
        a = fem.form(ufl.inner(u, v)*ufl.dx)
        self.M_ = dlfp.assemble_matrix(a)
        self.M_.assemble()
        self.M = trans2scipy(self.M_)

    def _init_measurement_matrix(self, coordinates):
        raise NotImplementedError

    def get_data(self, w):
        ## Here we only provide a simplest example, for more complex problems such as
        ## inverse scattering problems, we may need to rewrite this function.
        assert w.shape == self.S.shape[1], "The measure matrix not in accordance with the input function"
        return np.array(self.S@w)

    def loss_res(self, u=None):
        ## calaulate the resudial loss
        raise NotImplementedError

    def loss_prior(self, u=None):
        raise NotImplementedError

    def loss(self, u=None):
        loss_prior = self.loss_prior(u)
        loss_res = self.loss_res(u)
        loss_total = loss_res + loss_prior
        return loss_total, loss_res, loss_prior

    def eval_grad_res(self, u):
        ## calaulate the gradient of the resudial term
        raise NotImplementedError

    def eval_grad_prior(self, u):
        return self.prior.eval_grad(u)

    def eval_grad(self, u):
        grad_res = self.eval_grad_res(u)
        grad_prior = self.eval_grad_prior(u)
        return grad_res + grad_prior, grad_res, grad_prior

    def eval_hessian_res(self, u):
        ## calaulate the Hessian of the resudial term
        raise NotImplementedError

    def eval_hessian_prior(self, u):
        return self.prior.eval_hessian(u)

    def eval_hessian(self, u):
        hessian_res = self.eval_hessian_res(u)
        hessian_prior = self.eval_hessian_prior(u)
        return hessian_res + hessian_prior
        # return hessian_prior

    def hessian_linear_operator(self):
        linear_ope = spsl.LinearOperator((self.num_dofs, self.num_dofs), matvec=self.eval_hessian)
        return linear_ope

    def MxHessian(self, u):
        ## Usually, algorithms need a symmetric matrix.
        ## Here, we calculate MxHessian to make a symmetric matrix.
        return self.M@self.eval_hessian(u)

    def MxHessian_linear_operator(self):
        linear_op = spsl.LinearOperator((self.num_dofs, self.num_dofs), matvec=self.MxHessian)
        return linear_op

    def precondition(self, u):
        return self.prior.precondition(u)

    def precondition_linear_operator(self):
        linear_ope = spsl.LinearOperator((self.num_dofs, self.num_dofs), matvec=self.precondition)
        return linear_ope




