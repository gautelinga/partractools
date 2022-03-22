from types import CellType
from analyze_eulerian_timeseries import *
from felbm2fem import make_folder_safe, get_fluid_domain, make_xdmf_mesh

import matplotlib.pyplot as plt
import meshio

from mpi4py import MPI


def parse_args():
    parser = argparse.ArgumentParser(description="FEM interpolated FELBM data")
    parser.add_argument("folder", type=str, help="Folder")
    parser.add_argument("-t0", type=float, default=None, help="")
    parser.add_argument("-t1", type=float, default=None, help="")
    parser.add_argument("-axis", type=int, default=0, help="Axis")
    parser.add_argument("--show", action="store_true", help="Show plot")
    parser.add_argument("--phase_by_density", action="store_true", help="Show plot")
    args = parser.parse_args()
    return args

def load_mesh(nx, ny, is_fluid):
    mesh = None
    if rank == 0:
        nodes = np.array([(i, j) for j, i in itertools.product(range(ny+1), range(nx+1))], dtype=float)
        elems = np.array([(i + j*(nx+1), i+1 + j*(nx+1), i+1 + (j+1)*(nx+1), i + (j+1)*(nx+1)) 
                          for j, i in itertools.product(range(ny), range(nx))], dtype=int)
        
        elems = elems[is_fluid, :]
        used_nodes = np.unique(elems)
        map_ids = np.zeros(used_nodes.max()+1, dtype=int)
        for i, j in zip(used_nodes, range(len(used_nodes))):
            map_ids[i] = j
        nodes = nodes[used_nodes, :]
        elems = map_ids[elems]
        faces = []

        mesh = [elems, faces, nodes]

    return mesh


if __name__ == "__main__":
    args = parse_args()

    felbm_folder = args.folder
    analysisfolder = os.path.join(felbm_folder, "Analysis")
    make_folder_safe(analysisfolder)

    timestamps = select_timestamps(felbm_folder, args.t0, args.t1)

    is_fluid_xy, (nx, ny, nz) = get_fluid_domain(felbm_folder)
    is_fluid = is_fluid_xy.flatten()

    #print("nx = {}, ny = {}".format(nx, ny))

    rho = np.zeros_like(is_fluid_xy, dtype=float)
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    #p = np.zeros_like(rho)
    ux_ = ux.flatten()[is_fluid]
    uy_ = np.zeros_like(ux_)

    mesh = load_mesh(nx, ny, is_fluid, analysisfolder)

    print(mesh)
    