import argparse
import os

import dolfin as df
import matplotlib.pyplot as plt

from helpers import DolfinParams, Timestamps, mpi_max, mpi_min, PeriodicBC, parse_element, Btm, mesh2triang


def parse_args():
    parser = argparse.ArgumentParser(description="Plot pretty flow field")
    parser.add_argument("input_file", type=str, help="Input")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # rad = 0.35

    params = DolfinParams(args.input_file)

    dirname = os.path.dirname(args.input_file)
    imgfolder = os.path.join(dirname, "Images")
    if not os.path.exists(imgfolder):
        os.makedirs(imgfolder)

    meshfile = os.path.join(dirname, params["mesh"])


    print(os.listdir(dirname))
    params_sim = DolfinParams(os.path.join(dirname, "params.dat"))
    rad = params_sim["rad"]

    mesh = df.Mesh()
    with df.HDF5File(mesh.mpi_comm(), meshfile, "r") as h5f:
        h5f.read(mesh, "mesh", False)

    timestampsfile = os.path.join(dirname, params["timestamps"])
    ts = Timestamps(timestampsfile)

    dim = mesh.geometry().dim()
    x = mesh.coordinates()

    x_max = mpi_max(x)
    x_min = mpi_min(x)

    pbc = PeriodicBC(
        [params["periodic_x"], params["periodic_y"], params["periodic_z"]],
        dim, x_min, x_max)

    pressure_element = parse_element(params["pressure_space"])
    velocity_element = parse_element(params["velocity_space"])
    phase_field_element = parse_element(params["phase_field_space"])

    V_el = df.VectorElement(velocity_element[0], mesh.ufl_cell(),
                            velocity_element[1])
    P_el = df.FiniteElement(pressure_element[0], mesh.ufl_cell(),
                            pressure_element[1])
    C_el = df.FiniteElement(phase_field_element[0], mesh.ufl_cell(),
                            phase_field_element[1])

    V = df.FunctionSpace(mesh, V_el, constrained_domain=pbc)
    P = df.FunctionSpace(mesh, P_el, constrained_domain=pbc)
    C = df.FunctionSpace(mesh, C_el, constrained_domain=pbc)

    u_ = df.Function(V, name="u")
    p_ = df.Function(P, name="p")
    c_ = df.Function(C, name="c")

    facets = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    Btm(x_min, x_max).mark(facets, 1)

    nv = mesh.num_vertices()
    triang = mesh2triang(mesh)

    levels = [-10, 0, 10]
    colors = ['#ffffff', '#a0c0ff']

    for i in range(len(ts.ts)):
        tsi = ts[i]

        with df.HDF5File(mesh.mpi_comm(), tsi[1], "r") as h5f:
            #h5f.read(u_, "u")
            #h5f.read(p_, "p")
            h5f.read(c_, "phi")
        #uv = u_.compute_vertex_values(mesh)
        #u = [uv[i * nv: (i+1) * nv] for i in range(dim)]
        #p = p_.compute_vertex_values(mesh)

        c = c_.compute_vertex_values(mesh)

        fig, ax = plt.subplots(1, 1)
        ax.set_aspect("equal")
        ax.set_facecolor('lightgray')
        #ax.tick_params(left=False, bottom=False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        #ax_[0].tripcolor(triang, u[0], shading="gouraud")
        #ax_[1].tripcolor(triang, u[1], shading="gouraud")
        #ax_[2].tripcolor(triang, p, shading="gouraud")
        ax.tricontourf(triang, c, levels=levels, colors=colors)

        plt.savefig(os.path.join(imgfolder, "phase_{:06d}.png".format(i)))
        plt.close()
        #plt.show()

