import argparse
import os

import matplotlib.pyplot as plt
from matplotlib import tri
import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot pretty flow field")
    parser.add_argument("input_folder", type=str, help="Input folder")
    parser.add_argument("--skip", type=int, default=1, help="Skip")
    parser.add_argument("--size", type=float, default=4.0, help="Size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tsfolder = os.path.join(args.input_folder, "Timeseries")
    imgfolder = os.path.join(tsfolder, "Images")
    if not os.path.exists(imgfolder):
        os.makedirs(imgfolder)

    h5f = h5py.File(os.path.join(tsfolder, "phi_from_tstep_0.h5"), "r")
    
    xy = np.array(h5f["Mesh/0/mesh/geometry"])
    cells = np.array(h5f["Mesh/0/mesh/topology"])
    Lx = xy[:, 0].max()-xy[:, 0].min()
    Ly = xy[:, 1].max()-xy[:, 1].min()

    triang = tri.Triangulation(xy[:, 0], xy[:, 1], cells)

    tsteps = [int(k) for k in h5f["VisualisationVector"].keys()]
    tsteps = sorted(tsteps)[::args.skip]

    levels = [-10, 0, 10]
    colors = ['#ffffff', '#a0c0ff']

    c = np.zeros(len(xy))

    for i, tstep in enumerate(tsteps):
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

    h5f.close()