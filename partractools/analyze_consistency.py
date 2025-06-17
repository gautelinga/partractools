#!/usr/bin/env python3
import argparse
import os
from turtle import color
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from partractools.common.utils import Params, find_params, get_timeseries, read_params, get_folders

def parse_args():
    parser = argparse.ArgumentParser(description="Make elongation pdf from filaments")
    parser.add_argument("folder", type=str, help="Folder")
    parser.add_argument("--skip", type=int, default=1, help="Skip")
    return parser.parse_args()


def main():
    args = parse_args()

    folder = args.folder
    d = 1.0

    analysis_folder = os.path.join(folder, "Analysis")
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)

    images_folder = os.path.join(folder, "Images")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    ts_ = []
    posf_ = []

    params = Params(folder)
    t0 = params.get_tmin()

    ts, posf = get_timeseries(folder)
    ts_ = np.array(ts)

    t_ = ts_[::args.skip]

    ux_mean_ = np.zeros_like(t_)
    uy_mean_ = np.zeros_like(t_)
    cell_type_ = [np.zeros_like(t_) for i in range(3)]

    w_mean_ = np.zeros_like(t_)
    S_mean_ = np.zeros_like(t_)

    for it, t in enumerate(t_):
        posft, grp = posf[t]

        cell_type = None
        with h5py.File(posft, "r") as h5f:
            ux = np.array(h5f[grp + "/u"][:, 0])
            uy = np.array(h5f[grp + "/u"][:, 1])

            w = h5f[grp + "/w"][:, 0]
            S = h5f[grp + "/S"][:, 0]

            #if grp + "/cell_type" in h5f:
            cell_type = np.array(h5f[grp + "/cell_type"][:])

        ux_mean_[it] = ux.mean()
        uy_mean_[it] = uy.mean()

        w_mean_[it] = w.mean()
        S_mean_[it] = S.mean()

        if cell_type is not None:
            for i in range(len(cell_type_)):
                cell_type_[i][it] = np.sum(cell_type == i)

    U = uy_mean_[0]
    usign = np.sign(U)
    U = abs(U)

    tfactor = U/d

    fig, ax = plt.subplots(1, 4)
    ax[0].plot(t_ * tfactor, np.array(uy_mean_ / U * usign), label=r"$<u_y>$")
    ax[1].plot(t_ * tfactor, ux_mean_ / U, label=r"$<u_x>$")
    #ax0.plot(t_ * tfactor, U * np.ones_like(t_), label=r"$U$")
    #ax[0].legend()
    for i in range(len(cell_type_)):
        ax[2].plot(t_ * tfactor, cell_type_[i], label=f"cell_type$={i}$")
    ax[2].legend()

    ax[3].plot(t_ * tfactor, uy_mean_ / U * usign * cell_type_[0][0] / cell_type_[0])

    fig2, ax2 = plt.subplots(1, 2)

    ax2[0].plot((t_ - t_[0])* tfactor, w_mean_, label=r"w")
    ax2[1].plot((t_ - t_[0]) * tfactor, S_mean_ / tfactor, label=r"S")

    fig2.legend()

    plt.show()


if __name__ == "__main__":
    main()