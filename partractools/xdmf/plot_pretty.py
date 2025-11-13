import argparse
import os

import matplotlib.pyplot as plt
from matplotlib import tri
import h5py
import numpy as np

from partractools.common.utils import GenParams, mpi_root, mpi_rank, mpi_size, parse_xdmf, get_first_one, mpi_min, mpi_max, mpi_comm


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

    tsfolder = args.input_folder
    imgfolder = os.path.join(tsfolder, "Images")
    if mpi_root and not os.path.exists(imgfolder):
        os.makedirs(imgfolder)
    mpi_comm.Barrier()

    phi_xdmffile = os.path.join(tsfolder, "phi_from_tstep_0.xdmf")

    dsets, topology_address, geometry_address = parse_xdmf(phi_xdmffile, True)
    dsets = dict(dsets)

    with h5py.File(topology_address[0], "r") as h5f:
        cells = h5f[topology_address[1]][:]

    with h5py.File(geometry_address[0], "r") as h5f:
        xy = h5f[geometry_address[1]][:]
    
    t_ = sorted(dsets.keys())
    it_ = list(range(len(t_)))
    
    Lx = xy[:, 0].max()-xy[:, 0].min()
    Ly = xy[:, 1].max()-xy[:, 1].min()

    triang = tri.Triangulation(xy[:, 0], xy[:, 1], cells)

    start = 0 if args.start is None else args.start
    stop = len(t_) if args.stop is None else args.stop

    tsteps = list(enumerate(it_))[start:stop:args.skip]

    levels = [-10, 0, 10]
    colors = ['#ffffff', '#a0c0ff']

    c = np.zeros(len(xy))

    for i, tstep in tsteps[mpi_rank::mpi_size]:
        if mpi_root:
            print(f"Step: {i}, Time = {t_[tstep]} / {t_[-1]}")
        phi_file, dset_loc = dsets[t_[tstep]]
        with h5py.File(phi_file, "r") as h5f:
            c[:] = h5f[dset_loc][:].flatten()

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

    if mpi_root:
        print("now run:")
        print(f"ffmpeg -framerate 25 -i {imgfolder}/phase_%06d.png -c:v libx264 -pix_fmt yuv420p {imgfolder}/out.mp4")

    # h5f.close()
