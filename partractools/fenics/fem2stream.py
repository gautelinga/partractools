import argparse
import meshio
import numpy as np
import matplotlib.pyplot as plt
import os
from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def parse_args():
    parser = argparse.ArgumentParser(description="Convert from xdmf to FELBM data")
    parser.add_argument("xdmffile", type=str, help="XDMF file")
    parser.add_argument("-t0", type=float, default=None, help="")
    parser.add_argument("-t1", type=float, default=None, help="")
    parser.add_argument("--show", action="store_true", help="Show plot")
    args = parser.parse_args()
    return args

def make_folder_safe(analysisfolder):
    if rank == 0 and not os.path.exists(analysisfolder):
        os.makedirs(analysisfolder)

if __name__ == "__main__":
    args = parse_args()

    basefolder = os.path.dirname(args.xdmffile)
    outputfolder = os.path.join(basefolder, "Interpolated2")
    make_folder_safe(outputfolder)

    UxUy_ = np.loadtxt(os.path.join(basefolder, "UxUy.dat"))

    with meshio.xdmf.TimeSeriesReader(args.xdmffile) as reader:
        points, cells = reader.read_points_cells()
        n = list((points.max(axis=0)-points.min(axis=0)).astype(int))
        # assert(n[2] == 1)
        id2ij = (points - np.outer(np.ones(len(points)), points.min(axis=0))).astype(int)
        is_fluid = np.zeros(n, dtype=bool)

        ids_ij = np.zeros(n, dtype=int)
        for k, ij in enumerate(id2ij):
            if ij[0] < n[0] and ij[1] < n[1]:
                is_fluid[ij[0], ij[1]] = True
                ids_ij[ij[0], ij[1]] = k
        is_solid = np.logical_not(is_fluid)

        #exit()
        psi = np.zeros_like(is_fluid, dtype=float) #.flatten()
        ids = ids_ij[is_fluid]
        
        psi_out = np.zeros((n[1], n[0]))
        is_solid_out = np.zeros((n[1], n[0]), dtype=int)

        Y, X = np.meshgrid(range(n[1]), range(n[0]))

        """
        prmfile = open(os.path.join(outputfolder, "felbm_params.dat"), "w")
        prmfile.write("timestamps=timestamps.dat\n")
        prmfile.write("is_solid_file=is_solid.h5\n")
        prmfile.write("ignore_pressure=true\n")
        prmfile.write("ignore_density=true\n")
        prmfile.write("ignore_uz=true\n")
        prmfile.write("boundary_mode=sharp\n")
        prmfile.close()
        """

        is_solid_out[:, :] = np.logical_not(is_fluid.T)
        with h5py.File(os.path.join(outputfolder, "is_solid.h5"), "w") as h5f:
            h5f.create_dataset("is_solid", data=is_solid_out.reshape((n[0], n[1], 1)))

        tsfile = open(os.path.join(outputfolder, "timestamps.dat"), "w")
        for k in range(reader.num_steps):
            t, point_data, cell_data = reader.read_data(k)
            locname = "output_{}.h5".format(k)
            h5fname = os.path.join(outputfolder, locname)
            tsfile.write("{} {}\n".format(t, locname))
            psi_ = point_data["psi"]
            psi[is_fluid] = psi_[ids, 0]

            Ux, Uy = UxUy_[k, 0], UxUy_[k, 1]
            # psi[:, :] += - Uy * X + Ux * Y

            psi[is_fluid] -= psi[is_fluid].mean()

            psi[is_solid] = 0
            psi = np.ma.masked_where(is_solid, psi)

            fig, ax = plt.subplots(1, 2)

            ax[0].plot(range(n[1]), psi[0, :])
            ax[0].plot(range(n[1]), psi[-1, :])
            ax[1].imshow(psi)
            plt.show()

            psi_out[:, :] = psi.T

            #with h5py.File(h5fname, "w") as h5f:
            #    h5f.create_dataset("u_x", data=ux_out.reshape((n[0], n[1], 1)))
            #    h5f.create_dataset("u_y", data=uy_out.reshape((n[0], n[1], 1)))
                
        tsfile.close()