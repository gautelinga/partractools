import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
from partractools.common.utils import Params, get_h5data_location, compute_lines
from matplotlib import colors as mcolors
from scipy.interpolate import UnivariateSpline, make_interp_spline
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Convergence in time")
    parser.add_argument("folder", type=str, help="Folder")
    parser.add_argument("-t0", type=float, default=0.0, help="t0")
    parser.add_argument("--show", action="store_true", help="Show plot")
    parser.add_argument("-Dm", type=float, default=1e-2, help="Diffusivity")
    parser.add_argument("-Nrw", type=int, default=100000, help="Number of random walkers")
    parser.add_argument("-dt_min", type=float, default=0., help="smallest dt")
    parser.add_argument("--recompute", action="store_true", help="Force recompute")
    args = parser.parse_args()
    return args

def build_data(args):
    data_ = dict()
    #print(os.listdir(args.folder))
    for subfolder in os.listdir(args.folder):
        subf = os.path.join(args.folder, subfolder, '0')
        if not os.path.exists(subf):
            continue
        params = Params(subf)
        if not params.exists():
            continue
        #
        # print(subf, os.listdir(subf))
        t0 = params.get_tmin()
        Dm = float(params.get("Dm", t0))
        dt = float(params.get("dt", t0))
        Nrw = int(params.get("Nrw", t0))

        if Nrw != args.Nrw or Dm != args.Dm:
            continue

        print(Nrw, Dm, dt)

        posf = get_h5data_location(subf)
        ts = list(sorted(posf.keys()))
        t_ = np.array(ts)
        nt = len(t_)

        z_mean_ = np.zeros(nt)
        z_var_ = np.zeros(nt)

        for it, t in enumerate(t_):
            posft, cat = posf[t]
            with h5py.File(posft, "r") as h5f:
                pts = np.array(h5f[cat]["points"])[:, :]
            z_mean_[it] = pts[:, 2].mean()
            z_var_[it] = pts[:, 2].var()
        
        popt_z_mean = np.polyfit(t_[nt//2:], z_mean_[nt//2:], 1)
        popt_z_var = np.polyfit(t_[nt//2:], z_var_[nt//2:], 1)
        u_mean = popt_z_mean[0]
        D_eff = popt_z_var[0]/2

        if args.show:
            fig, ax = plt.subplots(1, 2)
            ax[0].plot(t_, z_mean_)
            ax[1].plot(t_, z_var_)
            ax[0].plot(t_, u_mean*t_)
            ax[1].plot(t_, 2*D_eff*t_)
            plt.show()

        data_[dt] = [u_mean, D_eff]
    return data_

if __name__ == "__main__":
    args = parse_args()

    pklfname = os.path.join(args.folder, "tmp.pkl")
    if os.path.exists(pklfname) and not args.recompute:
        data_ = pickle.load(open(pklfname, "rb"))
    else:
        data_ = build_data(args)
        pickle.dump(data_, open(pklfname, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    dt_ = np.array(sorted(list(data_.keys())))
    dt_ = dt_[dt_ > args.dt_min]

    D_eff_ = np.array([data_[dt][1] for dt in dt_])
    u_mean_ = np.array([data_[dt][0] for dt in dt_])
    D_eff_ex = args.Dm * ( 1 + 1./48/args.Dm**2)

    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    fig.set_tight_layout("tight")
    ax[0].plot(dt_, (D_eff_-D_eff_ex)/D_eff_ex, '*-', label="data")
    ax[0].plot(dt_, 0.1*dt_, label="$\sim \Delta t$")
    ax[0].semilogx()
    ax[0].semilogy()
    ax[0].set_xlabel(r"$\Delta t$")
    ax[0].set_ylabel(r"$(D_{\parallel}-D_{\parallel, \rm ex})/D_{\parallel, \rm ex}$")
    ax[0].legend()

    ax[1].plot(dt_, 1-u_mean_, '*-', label="data")
    ax[1].plot(dt_, 0.02*dt_, label="$\sim \Delta t$")
    ax[1].semilogx()
    ax[1].semilogy()
    ax[1].set_xlabel(r"$\Delta t$")
    ax[1].set_ylabel(r"$(u_{\rm ex}-u)/u_{\rm ex}$")
    ax[1].legend()

    plt.savefig(os.path.join(args.folder, "convergence_dt.pdf"))
    if args.show:
        plt.show()
    plt.close()