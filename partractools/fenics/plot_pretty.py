import argparse
import os
from itertools import product

import dolfin as df
import numpy as np
from mpi4py import MPI

from helpers import DolfinParams, Timestamps, mpi_max, mpi_min, PeriodicBC, parse_element, Btm


def parse_args():
    parser = argparse.ArgumentParser(description="Plot pretty flow field")
    parser.add_argument("input_file", type=str, help="Input")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # rad = 0.35

    params = DolfinParams(args.input_file)

    dirname = os.path.dirname(args.input_file)
    meshfile = os.path.join(dirname, params["mesh"])

    print(os.listdir(dirname))
    params_sim = DolfinParams(os.path.join(dirname, "params.dat"))
    rad = params_sim["rad"]

    mesh = df.Mesh()
    with df.HDF5File(mesh.mpi_comm(), meshfile, "r") as h5f:
        h5f.read(mesh, "mesh", False)

    timestampsfile = os.path.join(dirname, params["timestamps"])
    ts = Timestamps(timestampsfile)

    #for t, filename in ts.items():
    #    print(t, filename)

    dim = mesh.geometry().dim()
    x = mesh.coordinates()

    x_max = mpi_max(x)
    x_min = mpi_min(x)

    # print(dim)

    pbc = PeriodicBC(
        [params["periodic_x"], params["periodic_y"], params["periodic_z"]],
        dim, x_min, x_max)

    pressure_element = parse_element(params["pressure_space"])
    velocity_element = parse_element(params["velocity_space"])

    V_el = df.VectorElement(velocity_element[0], mesh.ufl_cell(),
                            velocity_element[1])
    P_el = df.FiniteElement(pressure_element[0], mesh.ufl_cell(),
                            pressure_element[1])

    V = df.FunctionSpace(mesh, V_el, constrained_domain=pbc)
    S = df.FunctionSpace(mesh, P_el, constrained_domain=pbc)
    Sc = df.FunctionSpace(mesh, P_el)
    u_ = df.Function(V, name="u")

    # psi stuff
    psi = df.TrialFunction(Sc)
    v = df.TestFunction(Sc)
    n = df.FacetNormal(mesh)

    vort = df.curl(u_)

    a = df.dot(df.grad(v), df.grad(psi)) * df.dx
    L = v * vort * df.dx + v * (n[1] * u_[0] - n[0] * u_[1]) * df.ds
    psi_ = df.Function(Sc, name="psi")
    bcs = [
        df.DirichletBC(
            Sc, df.Constant(0.),
            "on_boundary && sqrt(x[0]*x[0] + x[1]*x[1]) < {rad} + DOLFIN_EPS_LARGE"
            .format(rad=rad))
    ]

    facets = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    Btm(x_min, x_max).mark(facets, 1)

    xdmff = df.XDMFFile(mesh.mpi_comm(), os.path.join(dirname, "psi.xdmf"))
    xdmff.parameters["rewrite_function_mesh"] = False
    xdmff.parameters["functions_share_mesh"] = True

    for i in range(len(ts.ts)):
        tsi = ts[i]

        with df.HDF5File(mesh.mpi_comm(), tsi[1], "r") as h5f:
            h5f.read(u_, "u")

        # u_y = df.assemble(u_[1] * df.ds(1, domain=mesh, subdomain_data=facets))
        #print("u_y:", u_y)

        A = df.assemble(a)
        b = df.assemble(L)
        [bc.apply(A, b) for bc in bcs]
        df.solve(A, psi_.vector(), b, "gmres", "hypre_amg")

        xdmff.write(psi_, float(ts[i][0]))
        #xdmff.write(u_, float(ts[i][0]))
    xdmff.close()

    for i in range(len(ts.ts)):
        tsi = ts[i]

        with df.HDF5File(mesh.mpi_comm(), tsi[1], "r") as h5f:
            h5f.read(u_, "u")

        u_y = df.assemble(u_[1] * df.ds(1, domain=mesh, subdomain_data=facets))

        A = df.assemble(a)
        b = df.assemble(L)
        [bc.apply(A, b) for bc in bcs]
        df.solve(A, psi_.vector(), b, "gmres", "hypre_amg")

        import matplotlib.pyplot as plt
        import matplotlib.tri as tri
        xy = mesh.coordinates()
        triang = tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())
        fig, ax = plt.subplots(1, 1)
        C = psi_.compute_vertex_values(mesh)
        ax.tripcolor(triang, C * 0)
        c = ax.tricontour(triang, C, np.arange(-5, 5, 1) * u_y)
        #from inspect import getmembers
        #pts = np.vstack(c.allsegs[0])
        #ids_along = np.linalg.norm(pts, axis=1) < (rad + df.DOLFIN_EPS_LARGE)
        # pts_along = pts[ids_along, :]
        # pta, ptb = pts_along[0], pts_along[-1]
        # pts_sep = pts[np.logical_not(ids_along), :]
        # ax.scatter(pta[0], pta[1], color="red")
        # ax.scatter(ptb[0], ptb[1], color="blue")

        plt.show()
