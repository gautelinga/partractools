import argparse
import os

import matplotlib.pyplot as plt
from matplotlib import tri
import h5py
import numpy as np

from mpi4py.MPI import COMM_WORLD
mpi_rank = COMM_WORLD.Get_rank()
mpi_size = COMM_WORLD.Get_size()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot pretty flow field")
    parser.add_argument("input_folder", type=str, help="Input folder")
    parser.add_argument("--skip", type=int, default=1, help="Skip")
    parser.add_argument("--size", type=float, default=4.0, help="Size")
    parser.add_argument("--start", type=int, default=None, help="Start id")
    parser.add_argument("--stop", type=int, default=None, help="Stop id")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tsfolder = os.path.join(args.input_folder, "Timeseries")
    imgfolder = os.path.join(tsfolder, "Images")
    if not os.path.exists(imgfolder):
        os.makedirs(imgfolder)

    phi_file = os.path.join(tsfolder, "phi_from_tstep_0.h5")

    with h5py.File(phi_file, "r") as h5f:
        xy = np.array(h5f["Mesh/0/mesh/geometry"])
        cells = np.array(h5f["Mesh/0/mesh/topology"])
        t_ = [int(k) for k in h5f["VisualisationVector"].keys()]
    
    Lx = xy[:, 0].max()-xy[:, 0].min()
    Ly = xy[:, 1].max()-xy[:, 1].min()

    triang = tri.Triangulation(xy[:, 0], xy[:, 1], cells)

    t_ = sorted(t_)

    start = 0 if args.start is None else args.start
    stop = len(t_) if args.stop is None else args.stop

    tsteps = list(enumerate(t_))[start:stop:args.skip]

    levels = [-10, 0, 10]
    colors = ['#ffffff', '#a0c0ff']

    c = np.zeros(len(xy))

    for i, tstep in tsteps[mpi_rank::mpi_size]:
        if mpi_rank == 0:
            print(f"Step: {i}, Time = {tstep} / {t_[-1]}")
        with h5py.File(phi_file, "r") as h5f:
            c[:] = np.array(h5f["VisualisationVector/{}".format(tstep)]).flatten()

        fig, ax = plt.subplots(1, 1, figsize=((args.size), (Ly/Lx * args.size)))
        ax.set_aspect("equal")
        ax.set_facecolor('lightgray')
        #ax.tick_params(left=False, bottom=False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        #ax_[0].tripcolor(triang, u[0], shading="gouraud")
        #ax_[1].tripcolor(triang, u[1], shading="gouraud")
        #ax_[2].tripcolor(triang, p, shading="gouraud")
        ax.tricontourf(triang, c, levels=levels, colors=colors)

        plt.tight_layout()
        plt.savefig(os.path.join(imgfolder, "phase_{:06d}.png".format(i)))
        plt.close()
        #plt.show()

    # h5f.close()