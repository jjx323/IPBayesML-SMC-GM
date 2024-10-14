from mpi4py import MPI
import numpy as np
import scipy
import scipy.sparse as sp
import dolfinx
import dolfinx as dlf
from dolfinx import fem, la, geometry
import dolfinx.fem.petsc as dlfp
from petsc4py import PETSc
import ufl
import cupy as cp
import cupyx.scipy.sparse as csps


def construct_measure_matrix(V, points):
    '''
    Input:
        V: function space produced by dolfinx.fem.FunctionSpace
        points: input points, e.g.,
                points = [[x1_1, x1_2, x1_3], [x2_1, x2_2, x2_3]] stands for
                there are two measured points with coordinates:
                x1=(x1_1,x1_2,_x1_3), x2=(x2_1,x2_2,x2_3)
    Output:
        S: a scipy.sparse.csr_matrix

    Remark:
    The following code is based on the codes provided by Jorgen Dokken at the website:
    https://fenicsproject.discourse.group/t/how-to-construct-a-sparse-matrix-used-for-calculating-function-values-at-multiple-points/13081/4
    '''
    nx, dim = points.shape
    assert dim == 3, "The points should be shape Nx3 (N is the number of points)"
    mesh = V.mesh
    sdim = V.element.space_dimension
    # Determine what process owns a point and what cells it lies within
    # _, _, owning_points, cells = dolfinx.cpp.geometry.determine_point_ownership(
    #     mesh._cpp_object, points, 1e-6)
    ## The extral parameter 1e-6 seems not allowed
    _, _, owning_points, cells = dolfinx.cpp.geometry.determine_point_ownership(
        mesh._cpp_object, points,1e-6)
    owning_points = np.asarray(owning_points).reshape(-1, 3)

    # Pull owning points back to reference cell
    mesh_nodes = mesh.geometry.x
    cmap = mesh.geometry.cmaps[0]
    ref_x = np.zeros((len(cells), mesh.geometry.dim),
                     dtype=mesh.geometry.x.dtype)
    for i, (point, cell) in enumerate(zip(owning_points, cells)):
        geom_dofs = mesh.geometry.dofmap[cell]
        ref_x[i] = cmap.pull_back(point.reshape(-1, 3), mesh_nodes[geom_dofs])

    # Create expression evaluating a trial function (i.e. just the basis function)
    u = ufl.TrialFunction(V)
    num_dofs = V.dofmap.dof_layout.num_dofs * V.dofmap.bs
    if len(cells) > 0:
        # NOTE: Expression lives on only this communicator rank
        expr = dolfinx.fem.Expression(u, ref_x, comm=MPI.COMM_SELF)
        # values = expr.eval(mesh, np.asarray(cells))
        ## the command np.asarry(cells) seems leading to some errors
        values = expr.eval(mesh, cells)

        # Strip out basis function values per cell
        basis_values = values[:num_dofs:num_dofs*len(cells)]
    else:
        basis_values = np.zeros(
            (0, num_dofs), dtype=dolfinx.default_scalar_type)

    rows = np.zeros(nx*sdim, dtype='int')
    cols = np.zeros(nx*sdim, dtype='int')
    vals = np.zeros(nx*sdim)
    for k in range(nx):
        jj = np.arange(sdim*k, sdim*(k+1))
        rows[jj] = k
        # Find the dofs for the cell
        cols[jj] = V.dofmap.cell_dofs(cells[k])
        vals[jj] = basis_values[0][sdim*k:sdim*(k+1)]

    ij = np.concatenate((np.array([rows]), np.array([cols])), axis=0)
    S = sp.csr_matrix((vals, ij), shape=(nx, V.tabulate_dof_coordinates().shape[0]))
    return S

# def construct_measure_matrix(V, points):
#     '''
#     Input:
#         V: function space produced by dolfinx.fem.FunctionSpace
#         points: input points, e.g.,
#                 points = [[x1_1,x2_1], [x1_2,x2_2], [x1_3,x2_3]] stands for
#                 there are two measured points with coordinates:
#                 x1=(x1_1,x1_2,_x1_3), x2=(x2_1,x2_2,x2_3)
#     Output:
#         S: a scipy.sparse.csr_matrix
#     '''
#     assert points.shape[0] == 3, "The points should be shape 3xN"
#     domain = V.mesh
#     bb_tree = geometry.bb_tree(domain, domain.topology.dim)
#     cells = []
#     points_on_proc = []
#     cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
#     colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
#     for i, point in enumerate(points.T):
#         if len(colliding_cells.links(i)) > 0:
#             points_on_proc.append(point)
#             cells.append(colliding_cells.links(i)[0])
#         else:
#             print("There may be points lay outside of the domain!")
#     points_on_proc = np.array(points_on_proc, dtype=np.float64)
#     ## The following way may be not so elegant, further optimization need to be done
#     ut = fem.Function(V)
#     S = sp.lil_matrix((len(points_on_proc), len(ut.x.array)))
#     ut.x.array[:] = 0.0
#     ut.x.array[0] = 1.0
#     S[:, 0] = ut.eval(points_on_proc, cells).squeeze()
#     for i in range(1, ut.x.array.shape[0]):
#         ut.x.array[i-1] = 0.0
#         ut.x.array[i] = 1.0
#         S[:, i] = ut.eval(points_on_proc, cells).squeeze()
#     return S.tocsr()


def assemble_matrix_scipy(a, bcs=None, dtype=np.float64):
    '''
    Input:
        a: a bilinear form defined by ufl
        bc: dirichletbc
        dtype: specify the data type of the assemble
    Output:
        As: a scipy.sparse.csr_matrix
    '''
    ac = fem.form(a, dtype=dtype)
    if bcs is None:
        A = fem.assemble_matrix(ac, block_mode=la.BlockMode.expanded)
    else:
        A = fem.assemble_matrix(ac, bcs, block_mode=la.BlockMode.expanded)
    A.scatter_reverse()
    assert A.block_size == [1, 1]
    As = scipy.sparse.csr_matrix((A.data, A.indices, A.indptr))
    
    return As


def assemble_scipy(a, L, bcs=None, dtype=np.float64):
    '''
    Input:
        a: a bilinear form defined by ufl
        L: a linear form defined by ufl
        bc: dirichletbc
        dtype: specify the data type of the assemble
    Output:
        As: a scipy.sparse.csr_matrix
        b.array: an array in numpy
    '''
    ac = fem.form(a, dtype=dtype)
    if bcs is None:
        A = fem.assemble_matrix(ac, block_mode=la.BlockMode.expanded)
    else:
        A = fem.assemble_matrix(ac, bcs, block_mode=la.BlockMode.expanded)
    A.scatter_reverse()
    assert A.block_size == [1, 1]
    As = scipy.sparse.csr_matrix((A.data, A.indices, A.indptr))

    b = fem.assemble_vector(fem.form(L, dtype=dtype))
    if bcs is not None:
        fem.apply_lifting(b.array, [ac], bcs=[bcs])
        b.scatter_reverse(la.InsertMode.add)
        fem.set_bc(b.array, bcs)

    return As, b.array


def trans2scipy(A, gpu=False):
    '''
    Input:
        A: a matrix generated from dolfinx.fem.assemble_matrix or a
    Output:
        As: a scipy.sparse.csr_matrix
    '''
    if type(A) == dolfinx.la.MatrixCSR:
        if gpu == False:
            As = scipy.sparse.csr_matrix((A.data, A.indices, A.indptr))
        elif gpu == True:
            As = csps.csr_matrix((A.data, A.indices, A.indptr))
    elif type(A) == PETSc.Mat:
        indptr, indices, data = A.getValuesCSR()
        if gpu == False:
            As = scipy.sparse.csr_matrix((data, indices, indptr), shape=A.getSize())
        elif gpu == True:
            data, indices, indptr = cp.array(data), cp.array(indices), cp.array(indptr)
            As = csps.csr_matrix((data, indices, indptr), shape=A.getSize())
    else:
        raise TypeError("A should be dolfinx.la.MatrixCSR or petsc4py.PETSc.Mat")
    return As


def trans2petsc(A):
    '''
    Input:
        A: a sparse matrix with format scipy.sparse.csr_matrix
    Output:
        Ap: a matirix with format PETSc.Mat()
    '''
    assert type(A) == scipy.sparse.csr_matrix, "A must be of type scipy.sparse.csr_matrix"

    Ap = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    Ap.assemble()
    return Ap


def error_compare(ue, uh, space_type=None):
    '''
    Input:
        ue: the exact solution, can be a function defined by ufl
        uh: the solution solved by FEM
        space_type: e.g. ("Lagrange", 5)
    Output:
        error_L2: L2 error
        error_max: Maximum error
        msh: the mesh of the function uh.function_space.mesh
    '''
    msh = uh.function_space.mesh
    if space_type == None:
        V2 = uh.function_space
    else:
        V2 = fem.FunctionSpace(msh, space_type)
    uex = fem.Function(V2)
    uex.interpolate(ue)

    L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
    error_local = fem.assemble_scalar(L2_error)
    error_L2 = np.sqrt(msh.comm.allreduce(error_local, op=MPI.SUM))

    Vh = uh.function_space
    ueh = fem.Function(Vh)
    ueh.interpolate(ue)
    error_max = np.max(np.abs(ueh.x.array-uh.x.array))

    return error_L2, error_max, msh


def project(u, Vh):
    '''
    Input:
        u: fem.Function with fine mesh
        V: fem.Function with rough mesh
    Output:
        v: fem.Function with rough mesh

    Remark: This function is not so efficient
    '''
    v = fem.Function(Vh)
    v.interpolate(u, nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
        v.function_space.mesh._cpp_object,
        v.function_space.element,
        u.function_space.mesh._cpp_object,1e-14))
    return v


class Projector:
    '''
    Solve v = u, where u is a known function by FEM algorithm.
    '''
    def __init__(self, Vh):
        '''
        Vh is a function space
        '''
        self.Vh = Vh
        u_, v_ = ufl.TrialFunction(Vh), ufl.TestFunction(Vh)
        a = ufl.inner(u_, v_) * ufl.dx
        A = dlfp.assemble_matrix(fem.form(a))
        A.assemble()
        self.solverA = PETSc.KSP().create(Vh.mesh.comm)
        self.solverA.setOperators(A)
        self.solverA.setType(PETSc.KSP.Type.PREONLY)
        self.solverA.getPC().setType(PETSc.PC.Type.LU)

    def project(self, u):
        v_ = ufl.TestFunction(self.Vh)
        b = u * v_ * ufl.dx
        b = dlfp.assemble_vector(fem.form(b))
        v = fem.Function(self.Vh)
        self.solverA.solve(b, v.vector)
        return v

class Smoother:
    def __init__(self, Vh, degree=0.0):
        self.Vh = Vh
        self.fun = fem.Function(self.Vh)
        self.sfun = fem.Function(self.Vh)
        self.degree = degree
        self.set_degree(degree)

    # def init_new(self):
    #     nx = 100
    #     mesh = dlf.mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)
    #     Vh = fem.FunctionSpace(mesh, ("Lagrange", 1))
    #     smoother = Smoother(Vh, degree=self.degree)
    #     return smoother

    def set_degree(self, degree):
        u, v = ufl.TrialFunction(self.Vh), ufl.TestFunction(self.Vh)
        degree = fem.Constant(self.Vh.mesh, dolfinx.default_scalar_type(degree))
        a = ufl.inner(u, v)*ufl.dx + ufl.inner(degree*ufl.grad(u), ufl.grad(v))*ufl.dx
        A = dlfp.assemble_matrix(fem.form(a))
        A.assemble()
        self.solver = PETSc.KSP().create(self.Vh.mesh.comm)
        self.solver.setOperators(A)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)

    def smoothing(self, fun_vec):
        assert fun_vec.shape[0] == self.fun.x.array.shape[0]
        self.fun.x.array[:] = np.array(fun_vec)
        v = ufl.TestFunction(self.Vh)
        b = fem.form(ufl.dot(self.fun, v)*ufl.dx)
        b = dlfp.assemble_vector(b)
        self.solver.solve(b, self.sfun.vector)
        return np.array(self.sfun.x.array)


def print_my(*string, end=None, color=None):
    if color == 'red':
        print('\033[1;31m', end='')
        if end == None: print(*string)
        else: print(*string, end=end)
        print('\033[0m', end='')
    elif color == None:
        print(*string, end=end)
    elif color == 'blue':
        print('\033[1;34m', end='')
        if end == None: print(*string)
        else: print(*string, end=end)
        print('\033[0m', end='')
    elif color == 'green':
        print('\033[1;32m', end='')
        if end == None: print(*string)
        else: print(*string, end=end)
        print('\033[0m', end='')

