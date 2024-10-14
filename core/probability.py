from mpi4py import MPI
import numpy as np
import ufl
import dolfinx as dlf
from dolfinx import la
from dolfinx import fem
import dolfinx.fem.petsc as dlfp
import ufl
from petsc4py import PETSc
import scipy.sparse.linalg as spsl
import scipy.sparse as sps
from core.eigensystem import double_pass
from core.misc import construct_measure_matrix, trans2scipy


class GaussianElliptic2:
    '''
    prior Gaussian probability measure N(mean, C)
    C^{-1/2}: an elliptic operator -\nabla(\cdot\Theta\nabla \cdot) + a(x) Id

    Ref: A computational framework for infinite-dimensional Bayesian inverse problems
    part I: The linearized case, with application to global seismic inversion,
    SIAM J. Sci. Comput., 2013
    '''
    def __init__(self, function_space, params=None, boundary="Neumann", dtype=np.float64):
        assert type(function_space) == dlf.fem.function.FunctionSpaceBase
        self.Vh = function_space
        self._input_params = params
        if params is None:
            params = {
                "theta": lambda x: 1.0 + 0.0*x[0],
                "ax": lambda x: 1.0 + 0.0*x[0],
                "mean": lambda x: 0.0*x[0]
            }
        assert len(params) == 3
        self.params = params
        self.boundary = boundary
        self.dtype = dtype
        self._init_measure()
        self.mean = fem.Function(self.Vh)
        self.mean.interpolate(self.params["mean"])
        self.mean_vec = np.array(self.mean.x.array).squeeze()
        self.sample_fun = fem.Function(self.Vh)
        self.num_dofs = len(self.mean.x.array)
        ## The following are auxillary functions used for evaluate grad, hessian
        self.temp1_fun = fem.Function(self.Vh)
        self.temp2_fun = fem.Function(self.Vh)
        self.temp3_fun = fem.Function(self.Vh)
        self.temp4_fun = fem.Function(self.Vh)
        self._rdv = fem.Function(self.Vh)
        self.K = trans2scipy(self.K_)
        self.M = trans2scipy(self.M_)
        self.has_eigensystem = False

    def _init_measure(self):
        u, v = ufl.TrialFunction(self.Vh), ufl.TestFunction(self.Vh)
        a_fun = fem.Function(self.Vh)
        a_fun.interpolate(self.params["ax"])
        theta = fem.Function(self.Vh)
        theta.interpolate(self.params["theta"])
        ## self.L is employed for generating sample functions
        self.L = dlfp.assemble_vector(fem.form(ufl.inner(fem.Constant(self.Vh.mesh, dlf.default_scalar_type(0.0)), v)*ufl.dx))
        a = ufl.inner(theta*ufl.grad(u), ufl.grad(v))*ufl.dx + a_fun*ufl.inner(u, v)*ufl.dx
        ac = fem.form(a)
        m = ufl.inner(u, v)*ufl.dx
        mc = fem.form(m)
        self.K_ = dlfp.assemble_matrix(ac)
        self.K_.assemble()
        self._init_solver_K()
        self.M_ = dlfp.assemble_matrix(mc)
        self.M_.assemble()
        self._init_solver_M()
        self._eval_sqrtM()

    def _eval_sqrtM(self):
        '''
        M^{1/2} is calculated by: sum all of the elements of one row, take squareroot, and
        then construct a diagonal matrix.
        '''
        # row_sum_vector = self.M.getRowSum()
        # self.M_half_array = np.sqrt(row_sum_vector[:])
        # row_sum_vector[:] = self.M_half_array
        # self.M_half1 = PETSc.Mat().createDiagonal(row_sum_vector)

        ## In the following code, we just employed the operations in PETSc
        self.M_half_array = self.M_.getRowSum()
        self.M_half_array.sqrtabs()
        self.M_half = PETSc.Mat().createDiagonal(self.M_half_array)

    def eval_sqrtM(self, u, out_type="numpy"):
        if type(u) == PETSc.Vec:
            self.temp1_fun.vector.pointwiseMult(self.M_half_array, u)
        else:
            assert u.shape[0] == self.num_dofs
            self.temp2_fun.x.array[:] = np.array(u)
            self.temp1_fun.vector.pointwiseMult(self.M_half_array, self.temp2_fun.vector)
        if out_type == "numpy":
            return np.array(self.temp1_fun.x.array)
        elif out_type == "Function":
            return self.temp1_fun

    def eval_sqrtMinv(self, u):
        raise NotImplementedError

    def _init_solver_K(self):
        self.solverK = PETSc.KSP().create(self.Vh.mesh.comm)
        self.solverK.setOperators(self.K_)
        self.solverK.setType(PETSc.KSP.Type.PREONLY)
        self.solverK.getPC().setType(PETSc.PC.Type.LU)

    def _init_solver_M(self):
        self.solverM = PETSc.KSP().create(self.Vh.mesh.comm)
        self.solverM.setOperators(self.M_)
        self.solverM.setType(PETSc.KSP.Type.PREONLY)
        self.solverM.getPC().setType(PETSc.PC.Type.LU)

    def set_mean(self, point_values):
        assert len(self.mean.x.array) == len(point_values)
        self.mean.x.array[:] = point_values
        self.mean_vec = np.array(self.mean.x.array).squeeze()

    def generate_sample_zero_mean(self, num=1):
        ## the solver with LU decomposition seems to only do LU decomposition once
        ## so generate multiple samples will be quick, only use for loop seems enough
        samples = np.zeros((num, self.num_dofs))
        for i in range(num):
            # self.L.array[:] = self.M_half_array[:] * np.random.normal(0, 1, (self.num_dofs,))
            self._rdv.vector.setValues(range(self.num_dofs), np.random.normal(0, 1, (self.num_dofs,)))
            self._rdv.vector.assemble()
            # self.temp1_fun.vector.pointwiseMult(self.M_half_array, self._rdv.vector)
            temp_fun = self.eval_sqrtM(self._rdv.vector, out_type="Function")
            self.solverK.solve(temp_fun.vector, self.sample_fun.vector)
            samples[i, :] = np.array(self.sample_fun.x.array, dtype=self.dtype)
        return samples.squeeze()

    def generate_sample(self, num=1):
        rv = self.generate_sample_zero_mean(num=num)
        if num > 1:
            assert rv.shape[1] == self.mean.x.array.shape[0]
        else:
            assert rv.shape[0] == self.mean.x.array.shape[0]
        val = self.mean.x.array + rv
        return val.squeeze()

    def eval_grad(self, u):
        assert u.ndim == 1  ## u must be a one-dimensional vector
        assert self.num_dofs == u.shape[0]  ## the dimension of u should be equal to num_dofs

        ## description of the calculate procedure by numpy array
        # res = u - self.mean.x.array
        # K = trans2scipy(self.K)
        # M = trans2scipy(self.M)
        # grad = (K.T)@spsl.spsolve(M, K@res)
        # self.grad_scipy = spsl.spsolve(M, grad)

        ## For using petsc4py.PETSc.Mat, see the following webset:
        ## https://petsc.org/main/petsc4py/reference/petsc4py.PETSc.Mat.html

        ## temp1 = u - mean
        self.temp1_fun.x.array[:] = np.array(u - self.mean.x.array)
        ## temp2 = K@temp1
        self.K_.mult(self.temp1_fun.vector, self.temp2_fun.vector)
        ## temp1 = M^{-1}@temp2
        self.solverM.solve(self.temp2_fun.vector, self.temp1_fun.vector)
        ## temp2 = K^T@temp1
        self.K_.multTranspose(self.temp1_fun.vector, self.temp2_fun.vector)
        ## temp1 = M^{-1}@temp2
        self.solverM.solve(self.temp2_fun.vector, self.temp1_fun.vector)

        return np.array(self.temp1_fun.x.array)

    def eval_hessian(self, u):
        assert u.ndim == 1  ## u must be a one-dimensional vector
        assert self.num_dofs == u.shape[0]  ## the dimension of u should be equal to num_dofs

        ## description of the calculate procedure by numpy array
        # temp = (self.K.T)@spsl.spsolve(self.M, self.K@u_vec)
        # temp = spsl.spsolve(self.M, temp)

        ## Implementation by petsc4py.PETSc.Mat
        self.temp1_fun.x.array[:] = np.array(u)
        ## temp2 = K@temp1
        self.K_.mult(self.temp1_fun.vector, self.temp2_fun.vector)
        ## temp1 = M^{-1}@temp2
        self.solverM.solve(self.temp2_fun.vector, self.temp1_fun.vector)
        ## temp2 = K^T@temp1
        self.K_.multTranspose(self.temp1_fun.vector, self.temp2_fun.vector)
        ## temp1 = M^{-1}@temp2
        self.solverM.solve(self.temp2_fun.vector, self.temp1_fun.vector)

        return np.array(self.temp1_fun.x.array)

    def eval_CM_inner(self, u, v=None):
        """
        evaluate (C^{-1/2}u, C^{-1/2}v)
        """
        if v is None:
            v = np.array(u)
        assert u.ndim == 1 and v.ndim == 1
        assert self.num_dofs == u.shape[0] == v.shape[0]

        # temp1 = u_vec - self.mean.x.array
        # temp2 = v_vec - self.mean.x.array
        self.temp1_fun.x.array[:] = np.array(u - self.mean.x.array)
        self.temp2_fun.x.array[:] = np.array(v - self.mean.x.array)
        ## we need to calculate: temp1@(self.K.T)@spsl.spsolve(self.M, self.K@temp2)
        ## temp3 = self.K@temp2
        self.K_.mult(self.temp2_fun.vector, self.temp3_fun.vector)
        ## temp4 = M^{-1}@temp3
        self.solverM.solve(self.temp3_fun.vector, self.temp4_fun.vector)
        ## temp3 = K.T@temp4
        self.K_.multTranspose(self.temp4_fun.vector, self.temp3_fun.vector)
        val = self.temp1_fun.vector.dot(self.temp3_fun.vector)

        return val

    def eval_MCinv(self, u):
        """
        evaluate M C^{-1}u = M M^{-1}KM^{-1}K u = KM^{-1}K u
        """
        u = u.squeeze()
        assert u.ndim == 1 or u.ndim == 2
        assert u.shape[0] == self.num_dofs
        if u.ndim == 2:
            num_ = u.shape[1]
            val = np.zeros((self.num_dofs, num_))
        else:
            num_ = 1
            val = np.zeros((self.num_dofs, 1))

        for idx in range(num_):
            if num_ > 1:
                self.temp1_fun.x.array[:] = np.array(u[:, idx])
            if num_ == 1:
                self.temp1_fun.x.array[:] = np.array(u)
            ## we need to calculate: (self.K.T)@spsl.spsolve(self.M, self.K@temp2)
            ## temp2 = self.K@temp2
            self.K_.mult(self.temp1_fun.vector, self.temp2_fun.vector)
            ## temp3 = M^{-1}@temp2
            self.solverM.solve(self.temp2_fun.vector, self.temp3_fun.vector)
            ## temp4 = K.T@temp3
            self.K_.multTranspose(self.temp3_fun.vector, self.temp4_fun.vector)
            val[:, idx] = np.array(self.temp4_fun.x.array)

        return val.squeeze()

    def eval_sqrtCsqrtMinv(self, u):
        '''
        evaluate K^{-1} M^{1/2} u
        '''
        u = u.squeeze()
        assert u.ndim == 1 or u.ndim == 2
        assert u.shape[0] == self.num_dofs
        if u.ndim == 2:
            num_ = u.shape[1]
            val = np.zeros((self.num_dofs, num_))
        else:
            num_ = 1
            val = np.zeros((self.num_dofs, 1))

        for idx in range(num_):
            if num_ > 1:
                self.temp1_fun.x.array[:] = np.array(u[:, idx])
            if num_ == 1:
                self.temp1_fun.x.array[:] = np.array(u)
            self.solverK.solve(self.eval_sqrtM(self.temp1_fun.x.array, out_type="Function").vector, self.temp2_fun.vector)
            val[:, idx] = np.array(self.temp2_fun.x.array)

        return val.squeeze()

    def eval_Cinv(self, u):
        """
        evaluate C^{-1}u = M^{-1}KM^{-1}K u
        """
        u = u.squeeze()
        assert u.ndim == 1 or u.ndim == 2
        assert u.shape[0] == self.num_dofs
        if u.ndim == 2:
            num_ = u.shape[1]
            val = np.zeros((self.num_dofs, num_))
        else:
            num_ = 1
            val = np.zeros((self.num_dofs, 1))

        for idx in range(num_):
            if num_ > 1:
                self.temp1_fun.x.array[:] = np.array(u[:, idx])
            if num_ == 1:
                self.temp1_fun.x.array[:] = np.array(u)
            self.K_.mult(self.temp1_fun.vector, self.temp2_fun.vector)
            self.solverM.solve(self.temp2_fun.vector, self.temp1_fun.vector)
            self.K_.mult(self.temp1_fun.vector, self.temp2_fun.vector)
            self.solverM.solve(self.temp2_fun.vector, self.temp1_fun.vector)
            val[:, idx] = np.array(self.temp1_fun.x.array)

        return val.squeeze()

    def eval_CMinv(self, u):
        """
        evaluate K^{-1} M K^{-1} u
        """
        u = u.squeeze()
        assert u.ndim == 1 or u.ndim == 2
        assert u.shape[0] == self.num_dofs
        if u.ndim == 2:
            num_ = u.shape[1]
            val = np.zeros((self.num_dofs, num_))
        else:
            num_ = 1
            val = np.zeros((self.num_dofs, num_))

        for idx in range(num_):
            if num_ > 1:
                self.temp1_fun.x.array[:] = np.array(u[:, idx])
            if num_ == 1:
                self.temp1_fun.x.array[:] = np.array(u)
            self.solverK.solve(self.temp1_fun.vector, self.temp2_fun.vector)
            self.M_.mult(self.temp2_fun.vector, self.temp1_fun.vector)
            self.solverK.solve(self.temp1_fun.vector, self.temp2_fun.vector)
            val[:, idx] = np.array(self.temp2_fun.x.array)

        return val.squeeze()

    def eval_C(self, u):
        """
        evaluate K^{-1} M K^{-1} M u
        """
        u = u.squeeze()
        assert u.ndim == 1 or u.ndim == 2
        assert u.shape[0] == self.num_dofs
        if u.ndim == 2:
            num_ = u.shape[1]
            val = np.zeros((self.num_dofs, num_))
        else:
            num_ = 1
            val = np.zeros((self.num_dofs, 1))

        for idx in range(num_):
            if num_ > 1:
                self.temp1_fun.x.array[:] = np.array(u[:, idx])
            if num_ == 1:
                self.temp1_fun.x.array[:] = np.array(u)
            self.M_.mult(self.temp1_fun.vector, self.temp2_fun.vector)
            self.solverK.solve(self.temp2_fun.vector, self.temp1_fun.vector)
            self.M_.mult(self.temp1_fun.vector, self.temp2_fun.vector)
            self.solverK.solve(self.temp2_fun.vector, self.temp1_fun.vector)
            val[:, idx] = np.array(self.temp1_fun.x.array)

        return val.squeeze()

    def eval_sqrtC(self, u):
        """
        evaluate K^{-1} M u
        """
        u = u.squeeze()
        assert u.ndim == 1 or u.ndim == 2
        assert u.shape[0] == self.num_dofs
        if u.ndim == 2:
            num_ = u.shape[1]
            val = np.zeros((self.num_dofs, num_))
        else:
            num_ = 1
            val = np.zeros((self.num_dofs, 1))

        for idx in range(num_):
            if num_ > 1:
                self.temp1_fun.x.array[:] = np.array(u[:, idx])
            if num_ == 1:
                self.temp1_fun.x.array[:] = np.array(u)
            self.M_.mult(self.temp1_fun.vector, self.temp2_fun.vector)
            self.solverK.solve(self.temp2_fun.vector, self.temp1_fun.vector)
            val[:, idx] = np.array(self.temp1_fun.x.array)

        return val.squeeze()

    def eval_sqrtCinv(self, u):
        '''
        evaluate M^{-1} K u
        '''
        u = u.squeeze()
        assert u.ndim == 1 or u.ndim == 2
        assert u.shape[0] == self.num_dofs
        if u.ndim == 2:
            num_ = u.shape[1]
            val = np.zeros((self.num_dofs, num_))
        else:
            num_ = 1
            val = np.zeros((self.num_dofs, 1))

        for idx in range(num_):
            if num_ > 1:
                self.temp1_fun.x.array[:] = np.array(u[:, idx])
            if num_ == 1:
                self.temp1_fun.x.array[:] = np.array(u)
            self.K_.mult(self.temp1_fun.vector, self.temp2_fun.vector)
            self.solverM.solve(self.temp2_fun.vector, self.temp1_fun.vector)
            val[:, idx] = np.array(self.temp1_fun.x.array)

        return val.squeeze()

    def precondition(self, u):
        # # temp = spsl.spsolve(self.K, self.M@spsl.spsolve((self.K).T, self.M@m_vec))
        # ## Usually, algorithms need a symmetric matrix.
        # ## Here, we drop the last M in prior, e.g.,
        # ## For GaussianElliptic2, we calculate K^{-1}MK^{-1}m_vec instead of K^{-1}MK^{-1}M m_vec
        # # temp = spsl.spsolve(self.K, self.M@spsl.spsolve(self.K, self.M@m_vec))
        # if self.use_LU == False:
        #     temp = spsl.spsolve(self.K, self.M@spsl.spsolve(self.K, m_vec))
        # elif self.use_LU == True:
        #     temp = self.luK.solve(m_vec)
        #     temp = self.M@temp
        #     temp = self.luK.solve(temp)
        # else:
        #     raise NotImplementedError("use_LU must be True or False")
        # return np.array(temp)
        assert u.ndim == 1
        assert u.shape[0] == self.num_dofs
        self.temp1_fun.x.array[:] = np.array(u)

        self.solverK.solve(self.temp1_fun.vector, self.temp2_fun.vector)
        self.M_.mult(self.temp2_fun.vector, self.temp1_fun.vector)
        self.solverK.solve(self.temp1_fun.vector, self.temp2_fun.vector)
        return np.array(self.temp2_fun.x.array)

        # self.M_.mult(self.temp1_fun.vector, self.temp2_fun.vector)
        # self.solverK.solve(self.temp2_fun.vector, self.temp1_fun.vector)
        # self.M_.mult(self.temp1_fun.vector, self.temp2_fun.vector)
        # self.solverK.solve(self.temp2_fun.vector, self.temp1_fun.vector)
        # self.M_.mult(self.temp1_fun.vector, self.temp2_fun.vector)
        # return np.array(self.temp2_fun.x.array)

    def pointwise_variance_field(self, xx, yy=None):
        '''
        This function evaluate the pointwise variance field in a finite element discretization

        Parameters
        ----------
        xx : list
            [(x_1,y_1), \cdots, (x_N, y_N)]
        yy : list
            [(x_1,y_1), \cdots, (x_M, y_M)]

        Returns: variance field c(xx, yy), a matrix NxM
        -------
        None.

        '''

        if yy is None:
            yy = np.copy(xx)

        SN = construct_measure_matrix(self.Vh, np.array(xx))
        SM = construct_measure_matrix(self.Vh, np.array(yy))
        SM = (SM.todense()).T

        ## Here SM and SN are matrixes, I am not sure the PETSc can solve equations AX=B with B to be a matrix.
        ## So as a temporary solution, I convert the matrixes to matrixes in scipy and use
        ## scipy.sparse.linalg.spsolve as the linear equation solver.
        # self.luK = spsl.splu(self.K.tocsc())
        # val = np.zeros((SM.shape[0], SM.shape[1]))
        # for idx in range(val.shape[1]):
        #     val[:, idx] = self.luK.solve(np.array(SM[:, idx])).squeeze()
        val = spsl.spsolve(self.K.tocsr(), SM)
        val = self.M@val
        # for idx in range(val.shape[1]):
        #     val[:, idx] = self.luK.solve(np.array(val[:, idx])).squeeze()
        val = spsl.spsolve(self.K.tocsr(), val)
        val = SN@val

        if sps.issparse(val):
            val = val.todense()
        return np.array(val)

    def eval_MC(self, u):
        # def eval_MKinvMKinvM_times_param(self, u):
        u = u.squeeze()
        assert u.ndim == 1 or u.ndim == 2
        assert u.shape[0] == self.num_dofs
        if u.ndim == 2:
            num_ = u.shape[1]
            val = np.zeros((self.num_dofs, num_))
        else:
            num_ = 1
            val = np.zeros((self.num_dofs, 1))

        for idx in range(num_):
            if num_ > 1:
                self.temp1_fun.x.array[:] = np.array(u[:, idx])
            if num_ == 1:
                self.temp1_fun.x.array[:] = np.array(u)
            self.M_.mult(self.temp1_fun.vector, self.temp2_fun.vector)
            self.solverK.solve(self.temp2_fun.vector, self.temp1_fun.vector)
            self.M_.mult(self.temp1_fun.vector, self.temp2_fun.vector)
            self.solverK.solve(self.temp2_fun.vector, self.temp1_fun.vector)
            self.M_.mult(self.temp1_fun.vector, self.temp2_fun.vector)
            val[:, idx] = np.array(self.temp2_fun.x.array)

        return val.squeeze()

    def _eval_MC(self):
    # def _eval_MKinvMKinvM_times_param(self):
        self.linear_ope = spsl.LinearOperator((self.num_dofs, self.num_dofs), matvec=self.eval_MC)
        return self.linear_ope

    def eval_Minv(self, u):
        u = u.squeeze()
        assert u.ndim == 1 or u.ndim == 2
        assert u.shape[0] == self.num_dofs
        if u.ndim == 2:
            num_ = u.shape[1]
            val = np.zeros((self.num_dofs, num_))
        else:
            num_ = 1
            val = np.zeros((self.num_dofs, num_))

        u = u.reshape(self.num_dofs, num_)
        for idx in range(num_):
            self.temp1_fun.x.array[:] = u[:, idx]
            self.solverM.solve(self.temp1_fun.vector, self.temp2_fun.vector)
            val[:, idx] = np.array(self.temp2_fun.x.array)

        return val.squeeze()

    def _eval_Minv(self):
        self.linear_ope = spsl.LinearOperator((self.num_dofs, self.num_dofs), matvec=self.eval_Minv)
        return self.linear_ope

    def eval_eigensystem(self, num_eigval, method='double_pass', oversampling_factor=20, low_val=0.1, **kwargs):
        '''
        Calculate the eigensystem of C v = \lambda v, where C is the covariance operator
        Through FEM discretization, C = A^{-2} = K^{-1}MK^{-1}M, so we need to calculate
        MK^{-1}MK^{-1}M v = \lambda M v, with MK^{-1}MK^{-1}M be a symmetric matrix.
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
            # MC = self._eval_MKinvMKinvM_times_param()
            MC = self._eval_MC()
            rs = num_eigval + oversampling_factor
            omega = np.random.randn(self.M.shape[0], rs)
            self.eigval, self.eigvec = double_pass(
                MC, M=self.M, Minv=self._eval_Minv(),
                r=num_eigval, l=oversampling_factor, n=self.M.shape[0]
            )
        elif method == 'single_pass':
            raise NotImplementedError
        elif method == 'scipy_eigsh':
            ## The eigsh function in scipy seems much slower than the "double_pass" and
            ## "single_pass" algorithms implemented in the package. The eigsh function
            ## is kept as a baseline for testing our implementations.
            # MC = self._eval_MKinvMKinvM_times_param()
            MC = self._eval_MC()
            self.eigval, self.eigvec = spsl.eigsh(
                MC, M=self.M, k=num_eigval + oversampling_factor, which='LM',
                Minv=self._eval_Minv(), **kwargs
            )  # optional parameters: maxiter=maxiter, v0=v0 (initial)
            index = self.eigval > low_val
            if np.sum(index) == num_eigval:
                print("Warring! The eigensystem may be inaccurate!")
            self.eigval = np.flip(self.eigval[index])
            self.eigvec = np.flip(self.eigvec[:, index], axis=1)
        else:
            assert False, "method should be double_pass, scipy_eigsh"

        self.has_eigensystem = True


def simple_sqrt_eig_vals(params=None):
    if params is None:
        params = {
            "theta": lambda x: 1.0 + 0.0*x[0],
            "ax": lambda x: 1.0 + 0.0*x[0],
            "mean": lambda x: 0.0*x[0]
        }
    a=params['ax']
    theta=params['theta']
    tem_mn=[]
    for i in range(10):
        for j in range(10):
            tem_mn.append(i**2+j**2)
    tem_mn=np.sort(np.array(tem_mn))
    eig_vals=a([1,2])+theta([1,2])*tem_mn*np.pi**2
    eig_vals=1/eig_vals
    # eig_vals=eig_vals**2
    return eig_vals 






