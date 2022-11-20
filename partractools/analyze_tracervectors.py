#!/usr/bin/env python3
import argparse
import os
from turtle import color
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from partractools.common.utils import Params, find_params, get_timeseries, read_params, get_folders

def gaussian(x, mu,sig):
    return 1./(np.sqrt(2.*np.pi)*sig) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def lognormal(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig*x) * np.exp(-np.power(np.log(x) - mu, 2.) / (2 * np.power(sig, 2.)))


def calc_moments(data, w):
    w /= w.sum()
    assert all(w >= 0)
    data_mean = np.sum(data * w)  # since w.sum() == 1
    data_var = np.sum(
        (data - data_mean)**2 *
        w)  # OK for large N, how to calculate sample mean with weights?
    data_std = np.sqrt(data_var)
    return data_mean, data_var, data_std


def calc_hist(data, data_mean, data_std, brange, nstd, bins):
    if brange is None:
        brange = (data_mean - nstd * data_std,
                  data_mean + nstd * data_std)
    elif isinstance(brange, str):
        x0, x1 = [float(a) for a in brange.split(":")]
        brange = (x0, x1)

    hist, bin_edges = np.histogram(data,
                                   density=True,
                                   bins=bins,
                                   range=brange)
    x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return x, hist

def autocorr(x):
    r2=np.fft.ifft(np.abs(np.fft.fft(x))**2).real
    return r2

def main():
    parser = argparse.ArgumentParser(
        description="Make elongation pdf from filaments")
    parser.add_argument("folder", type=str, help="Folder")
    parser.add_argument("-t", type=str, default="0.0", help="t")
    parser.add_argument("-d", type=float, default="1", help="Bead diameter")
    parser.add_argument("-neq", type=float, default=0, help="Equilibration time (in units of t_adv)")
    parser.add_argument("--range", type=str, default=None, help="t")
    parser.add_argument("-bins", type=int, default=100, help="Number of bins")
    parser.add_argument("--show", action="store_true", help="Show")
    parser.add_argument("--save", action="store_true", help="Save")
    parser.add_argument("--terminal",
                        action="store_true",
                        help="Print to terminal")
    parser.add_argument("--output", action="store_true", help="Output")
    parser.add_argument("-nstd",
                        type=int,
                        default=5,
                        help="Number of stds to include in data")
    parser.add_argument("--single",
                        action="store_true",
                        help="Do it on a single one")
    parser.add_argument("--nolog", action="store_true", help="No logarithm")
    parser.add_argument("--tol",
                        type=float,
                        default=0.0,
                        help="Tolerance for removing outliers")
    parser.add_argument("--weights", type=str, default="dl0", help="Weights")
    parser.add_argument("--skip", type=int, default=1, help="Skip")

    args = parser.parse_args()

    possible_fields = [["u", "Vector", "Node"],  ["c", "Scalar", "Node"],
                       ["p", "Scalar", "Node"],  ["rho", "Scalar", "Node"],
                       ["H", "Scalar", "Node"],  ["n", "Vector", "Node"],
                       ["dA", "Scalar", "Face"], ["dA0", "Scalar", "Face"],
                       ["dl", "Scalar", "Edge"], ["dl0", "Scalar", "Edge"]]

    folders = get_folders(args.folder)
    nfld = len(folders)
    if nfld == 0:
      folders = [args.folder]

    analysis_folder = os.path.join(args.folder, "Analysis")
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)

    images_folder = os.path.join(args.folder, "Images")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    ts_ = []
    posf_ = []
    for ifolder, folder in enumerate(folders):
        params = Params(folder)
        t0 = params.get_tmin()

        ts, posf = get_timeseries(folder)

        ts = np.array(ts)

        inn = args.t.split(":")
        if len(inn) == 1:
            t0 = inn[0]
            t1 = ts[-1]
        elif len(inn) == 2:
            t0, t1 = inn
        else:
            print("Wrong input")
            exit()
        t0 = float(t0)
        t1 = float(t1)

        ts = ts[ts >= t0]
        ts = ts[ts <= t1]

        ts_.append(ts)
        posf_.append(posf)

    t_ = ts_[0][::args.skip]

    Lx = 1
    Ly = np.sqrt(3)

    ux_mean_ = np.zeros_like(t_)
    uy_mean_ = np.zeros_like(t_)
    for it, t in enumerate(t_):
        ux_ = []
        uy_ = []
        for posf in posf_:
            posft, grp = posf[t]
            with h5py.File(posft, "r") as h5f:
                ux = np.array(h5f[grp + "/u"][:, 0])
                uy = np.array(h5f[grp + "/u"][:, 1])

            ux_.append(ux)
            uy_.append(uy)
        ux = np.hstack(ux_)
        uy = np.hstack(uy_)

        ux_mean_[it] = ux.mean()
        uy_mean_[it] = uy.mean()

    U = uy_mean_.mean()

    tfactor = U/args.d

    #fig0, ax0 = plt.subplots(1, 1)
    #ax0.plot(t_ * tfactor, uy_mean_, label=r"$<u_y>$")
    #ax0.plot(t_ * tfactor, ux_mean_, label=r"$<u_x>$")
    #ax0.plot(t_ * tfactor, U * np.ones_like(t_), label=r"$U$")
    #ax0.legend()
    #plt.show()

    print("t_adv =", 1./tfactor)
    # exclude the first ~5 advective times for equilibration
    #t_ = t_[t_ >= args.neq * 1./tfactor]

    S_ = [[] for _ in t_]
    w_ = [[] for _ in t_]

    NS = 100
    St_ = np.zeros((NS, len(t_)))

    tlast = t_[-1]
    posft, grp = posf[tlast]
    with h5py.File(posft, "r") as h5f:
        wlast = np.array(h5f[grp + "/w"][:, 0])
        print(f"wmax={wlast.max()}")
        print(f"wmin={wlast.min()}")
        print(f"wmed={np.median(wlast)}")
    h, bins = np.histogram(wlast, bins=256, density=True)
    x = 0.5*(bins[1:]+bins[:-1])

    p0 = [wlast.mean(), wlast.std()]

    popt, pcov = curve_fit(gaussian, x, h, p0)
    plt.plot(x, h)
    plt.plot(x, gaussian(x, *popt))
    plt.show()

    #exit()
    for it0, t0 in enumerate(t_):
        #dl0_ = []
        #n0_ = []
        w0_ = []
        S0_ = []
        for posf in posf_:
            posft, grp = posf[t0]
            with h5py.File(posft, "r") as h5f:
                #dl0 = np.array(h5f[grp + "/dl"][:, 0])/np.array(h5f[grp + "/dl0"][:, 0])
                #n0 = np.array(h5f[grp + "/doublings"][:, 0])
                w0 = np.array(h5f[grp + "/w"][:, 0])
                S0 = np.array(h5f[grp + "/S"][:, 0])
                x = np.array(h5f[grp + "/points"][:, :])
                n = np.array(h5f[grp + "/n"][:, :])
            w0_.append(w0)
            S0_.append(S0)
        w0 = np.hstack(w0_)
        S0 = np.hstack(S0_)
        w_[it0].append(w0)
        S_[it0].append(S0)
        St_[:, it0] = S0[:NS]

        if args.show:
            fig, ax = plt.subplots(1, 2)
            xx = np.remainder(x[:, 0] + 0.5*Lx, Lx) - 0.5*Lx
            yy = np.remainder(x[:, 1] + 0.5*Ly, Ly) - 0.5*Ly
            c1 = ax[0].quiver(xx, yy, n[:, 0], n[:, 1], S0)
            fig.colorbar(c1)
            ax[1].hist(S0, bins=256, density=True)
            plt.show()

    fig, ax = plt.subplots(1, 2)
    for iS in range(2):
        ax[0].plot(t_, St_[iS, :])
        ax[1].plot(t_[100:], autocorr(St_[iS, 100:]))
    plt.show()

    S_ = [np.concatenate(Ss) for Ss in S_]

    w_ = [np.concatenate(ws) for ws in w_]

    w_mean = np.zeros_like(t_)
    w_var = np.zeros_like(t_)
    rho_mean = np.zeros_like(t_)

    S_mean = np.zeros_like(t_)

    for it, t in enumerate(t_):
        S_mean[it] = S_[it].mean()
        w = w_[it]
        
        rho_mean[it] = np.exp(w).mean()
        w_mean[it] = w.mean()
        w_var[it] = w.var()
        w_std = np.sqrt(w_var[it])

        x_w, hist_w = calc_hist(w, w_mean[it], w_std,
                                (w_mean[it] - args.nstd*w_std, w_mean[it] + args.nstd*w_std), args.nstd, args.bins)

    Scov = np.zeros((len(t_), len(t_)))
    for it, _ in enumerate(t_):
        for jt, _ in enumerate(t_):
            Scov[it, jt] = np.mean((S_[it] - S_mean[it])*(S_[jt] - S_mean[jt]))
    
    fig, ax = plt.subplots(1, 1)
    c = ax.pcolormesh(t_, t_, Scov)
    fig.colorbar(c)
    plt.show()

    dt = t_[1]-t_[0]
    int_Scov = np.array([Scov[it, :it].sum()*dt for it, _ in enumerate(t_)])
    plt.plot(t_, int_Scov)
    plt.show()

    if False:
        if args.show or args.save:
            fig, ax2 = plt.subplots(1, 1)

            var = "\\log(\\rho)"
            ax2.plot(x_w, hist_w, 'o')
            ax2.set_xlabel("$" + var + "$")
            ax2.set_ylabel("$P(" + var + ")$")
            nn = args.nstd
            xx = np.linspace(w_mean[it]-nn*w_std, w_mean[it]+nn*w_std, 1000)
            ax2.plot(xx, gaussian(xx, w_mean[it], w_std))

            if args.save:
                plt.savefig(os.path.join(images_folder, "rho_pdfs_t{}.png".format(t)))
            if args.show:
                plt.show()

    N = len(t_)
    tau = t_ - t_[0]

    popt_mean = np.polyfit(tau[:N//2] * tfactor, w_mean[:N//2], 1)
    popt_var = np.polyfit(tau[:N//2] * tfactor, w_var[:N//2], 1)

    print("mean: ", popt_mean)
    print("var:  ", popt_var)

    fig, ax = plt.subplots(1, 1)

    ax.plot(tau[:N//1] * tfactor, np.log(rho_mean[:N//1]), label='log(<rho>)')
    ax.plot(tau[:N//1] * tfactor, w_mean[:N//1], label='<log(rho)>')
    ax.plot(tau[:N//1] * tfactor, w_var[:N//1], label='Var(log(rho))')
    ax.plot(tau[:N//1] * tfactor, tau[:N//1] * tfactor * popt_mean[0], label="mean fit")
    ax.plot(tau[:N//1] * tfactor, tau[:N//1] * tfactor * popt_var[0], label="var fit")

    ax.legend()

    fig2, ax2 = plt.subplots(1, 1)
    ax2.plot(tau * tfactor, S_mean)

    plt.show()



if __name__ == "__main__":
    main()