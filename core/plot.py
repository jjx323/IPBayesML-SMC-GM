from dolfinx import plot
import scipy as sp
import numpy as np
import dolfinx as dlf
import pyvista
import scienceplots
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm
import matplotlib.ticker as ticker
# import vtk
# from vtk.util.numpy_support import vtk_to_numpy

plt.style.use('science')
plt.style.use(['science','ieee'])

# plt.rcParams.update({
#   "text.usetex": True,
#   "font.family": "serif"
# })

# rcParams['text.latex.preamble'] = r'\usepackage{amssymb} \usepackage{amsmath} \usepackage{amsthm} \usepackage{mathtools}'

def plot_mesh(domain, show=True, path=""):
    tdim = domain.topology.dim
    pyvista.start_xvfb()
    topology, cell_type, geometry = plot.vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_type, geometry)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    if show:
        plotter.show()
    else:
        plotter.save_graphic(path)
        plotter.close()

        
def plot_fun2d(u, nx=None, show=True, path="", view="2d", grid_on=False, package="pyvista"):
    assert package in {"pyvista", "matplotlib"}

    if package == "pyvista":
        V = u.function_space
        u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
        u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
        u_grid.point_data["u"] = u.x.array.real
        u_grid.set_active_scalars("u")
        if show is False:
            off_screen = True
        else:
            off_screen = False
        if view == "2d":
            u_plotter = pyvista.Plotter(off_screen=off_screen)
            u_plotter.add_mesh(u_grid, show_edges=grid_on)
            u_plotter.view_xy()
            if show:
                u_plotter.show()
            else:
                u_plotter.save_graphic(path)
                u_plotter.close()
        elif view == "3d":
            warped = u_grid.warp_by_scalar()
            plotter2 = pyvista.Plotter(off_screen=off_screen)
            plotter2.add_mesh(warped, show_edges=True, show_scalar_bar=True)
            if show:
                plotter2.show()
            else:
                plotter2.save_graphic(path)
                plotter2.close()
    elif package == "matplotlib":
        if nx is None:
            N = u.x.array.shape[0]
        else:
            N = nx
        
        coords = u.function_space.tabulate_dof_coordinates()[:, 0:2]
        minx = np.min(coords[:, 0])
        maxx = np.max(coords[:, 1])
        
        x_grid = np.linspace(minx, maxx, N)
        y_grid = np.linspace(minx, maxx, N)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        val = u.x.array
        # u_fun = sp.interpolate.NearestNDInterpolator(coords, val)
        u_fun = sp.interpolate.LinearNDInterpolator(coords, val)
        # u_fun = sp.interpolate.CloughTocher2DInterpolator(coords, val)
        with plt.style.context('science'):
            fig=plt.figure()
            ax = fig.add_subplot(111)
            
            ## No tick marks on the axis.
            ax.tick_params(axis='x', which='both', length=0)
            ax.tick_params(axis='y', which='both', length=0)
            ###################################################
            
            plt.contourf(X, Y, u_fun(X, Y), cmap=cm.jet, levels=50)
            
            ### Up to 6 ticks on the colorbar.
            cb=plt.colorbar()
            cb.locator = ticker.MaxNLocator(nbins=7)  
            cb.update_ticks()
            #############################
            
            # plt.xlabel(r'$x$', fontsize=20)
            # plt.ylabel(r'$y$', fontsize=20)
            # plt.title(r'$T_{LU}$', fontsize=20)
            plt.savefig(path, dpi=600, bbox_inches='tight')
            plt.close()















