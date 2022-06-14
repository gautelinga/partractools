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

def main():
    parser = argparse.ArgumentParser(
        description="Make elongation pdf from filaments")
    parser.add_argument("folder", type=str, help="Folder")
    parser.add_argument("-t", type=str, default="0.0", help="t")
    parser.add_argument("-d", type=float, default="16", help="Bead diameter")
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

    elongfilename = os.path.join(analysis_folder, "elongdata.dat")

    #params_ = []
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

    plt.plot(t_ * tfactor, uy_mean_)
    plt.plot(t_ * tfactor, ux_mean_)
    plt.plot(t_ * tfactor, U * np.ones_like(t_))
    plt.show()

    print("t_adv =", 1./tfactor)
    # exclude the first ~5 advective times for equilibration
    t_ = t_[t_ >= args.neq * 1./tfactor]

    rho_mean_ = np.zeros((len(t_), len(t_)))
    rho_var_ = np.zeros_like(rho_mean_)
    logrho_mean_ = np.zeros_like(rho_mean_)
    logrho_var_ = np.zeros_like(rho_mean_)

    logrho_ = [[] for _ in t_]

    for it0, t0 in enumerate(t_):
        #dl0_ = []
        #n0_ = []
        logelong0_ = []
        for posf in posf_:
            posft, grp = posf[t0]
            with h5py.File(posft, "r") as h5f:
                #dl0 = np.array(h5f[grp + "/dl"][:, 0])/np.array(h5f[grp + "/dl0"][:, 0])
                #n0 = np.array(h5f[grp + "/doublings"][:, 0])
                logelong0 = np.array(h5f[grp + "/logelong"][:, 0])
            #dl0_.append(dl0)
            #n0_.append(n0)
            logelong0_.append(logelong0)
        #dl0 = np.hstack(dl0_)
        #n0 = np.hstack(n0_)
        logelong0 = np.hstack(logelong0_)

        for it1, t1 in enumerate(t_):
            if it1 < it0:
                continue

            if it1 % 10 == 0:
                print("Computing rho(t1 | t0) where t1 = {} > t0 = {} \t\t({}/{})".format(t1, t0, len(t_)*it0 + it1, len(t_)**2))

            #dl1_ = []
            #n1_ = []
            logelong1_ = []
            for posf in posf_:
                posft, grp = posf[t1]
                with h5py.File(posft, "r") as h5f:
                    #dl1 = np.array(h5f[grp + "/dl"][:, 0])/np.array(h5f[grp + "/dl0"][:, 0])
                    #n1 = np.array(h5f[grp + "/doublings"][:, 0])
                    logelong1 = np.array(h5f[grp + "/logelong"][:, 0])
                #dl1_.append(dl1)
                #n1_.append(n1)
                logelong1_.append(logelong1)
            #dl1 = np.hstack(dl1_)
            #n1 = np.hstack(n1_)
            logelong1 = np.hstack(logelong1_)

            #rho = 
            logrho = logelong1 - logelong0 #0*np.log(dl1/dl0) + (n1 - n0) #* np.log(2)

            if it0 != it1:
                logrho = logrho[logrho != 0.0]
            logrho_[it1-it0].append(logrho)

            rho = np.exp(logrho)

            rho_mean = rho.mean()
            rho_mean_[it1, it0] = rho_mean

            rho_var_[it1, it0] = rho.var()
            logrho_mean = logrho.mean()
            logrho_var = logrho.var()
            logrho_mean_[it1, it0] = logrho_mean
            logrho_var_[it1, it0] = logrho_var
            logrho_std = np.sqrt(logrho_var)

            if False:
                x_logrho, hist_logrho = calc_hist(logrho, logrho_mean, logrho_std,
                                                (logrho_mean - args.nstd*logrho_std, logrho_mean + args.nstd*logrho_std), args.nstd, args.bins)

                if args.show or args.save:
                    fig, ax2 = plt.subplots(1, 1)

                    ax2.set_title("\\rho( {}| {})".format(t1, t0))
                    var = "\\log(\\rho)"
                    ax2.plot(x_logrho, hist_logrho, 'o')
                    ax2.set_xlabel("$" + var + "$")
                    ax2.set_ylabel("$P(" + var + ")$")
                    nn = args.nstd
                    xx = np.linspace(logrho_mean-nn*logrho_std, logrho_mean+nn*logrho_std, 1000)
                    ax2.plot(xx, gaussian(xx, logrho_mean, logrho_std))

                    if args.save:
                        plt.savefig(os.path.join(images_folder, "rho_pdfs_t{}.png".format(t)))
                    if args.show:
                        plt.show()
                    plt.close()


        #elongdata = np.vstack((t_, np.log(rho_mean_), np.log(rho_var_), np.log(rho_var_)/2, logrho_mean_, logrho_var_, np.sqrt(logrho_var_))).T
        #print(elongdata.shape)
        #np.savetxt(elongfilename, elongdata)

    logrho_ = [np.concatenate(logrhos) for logrhos in logrho_]
    lr_mean = np.zeros_like(t_)
    lr_var = np.zeros_like(t_)
    r_mean = np.zeros_like(t_)
    N = [len(logrho) for logrho in logrho_]
    for it, t in enumerate(t_):
        logrho = logrho_[it]
        logrho_mean = logrho.mean()
        logrho_var = logrho.var()
        logrho_std = np.sqrt(logrho_var)
        
        r_mean[it] = np.exp(logrho).mean()
        lr_mean[it] = logrho_mean
        lr_var[it] = logrho_var

        x_logrho, hist_logrho = calc_hist(logrho, logrho_mean, logrho_std,
                                          (logrho_mean - args.nstd*logrho_std, logrho_mean + args.nstd*logrho_std), args.nstd, args.bins)
        
    #if False:
        if args.show or args.save:
            fig, ax2 = plt.subplots(1, 1)

            var = "\\log(\\rho)"
            ax2.plot(x_logrho, hist_logrho, 'o')
            ax2.set_xlabel("$" + var + "$")
            ax2.set_ylabel("$P(" + var + ")$")
            nn = args.nstd
            xx = np.linspace(logrho_mean-nn*logrho_std, logrho_mean+nn*logrho_std, 1000)
            ax2.plot(xx, gaussian(xx, logrho_mean, logrho_std))

            if args.save:
                plt.savefig(os.path.join(images_folder, "rho_pdfs_t{}.png".format(t)))
            if args.show:
                plt.show()

    N = len(t_)
    tau = t_ - t_[0]

    popt_mean = np.polyfit(tau[:N//2] * tfactor, lr_mean[:N//2], 1)
    popt_var = np.polyfit(tau[:N//2] * tfactor, lr_var[:N//2], 1)

    print("mean: ", popt_mean)
    print("var:  ", popt_var)

    plt.plot(tau[:N//2] * tfactor, np.log(r_mean[:N//2]), label='log(<rho>)')
    plt.plot(tau[:N//2] * tfactor, lr_mean[:N//2], label='<log(rho)>')
    plt.plot(tau[:N//2] * tfactor, lr_var[:N//2], label='Var(log(rho))')
    plt.plot(tau[:N//2] * tfactor, tau[:N//2] * tfactor * popt_mean[0])
    plt.plot(tau[:N//2] * tfactor, tau[:N//2] * tfactor * popt_var[0])

    plt.legend()
    plt.show()

    if args.show or args.save:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        ax1.plot(tau * tfactor, np.log(r_mean[:]), "k-.")
        ax2.plot(tau * tfactor, lr_mean[:], "k-.")
        ax3.plot(tau * tfactor, lr_var[:], "k-.")
        for it0 in range(len(t_)):
            tau = t_[it0+1:] - t_[it0]
            ax1.plot(tau * tfactor, np.log(rho_mean_[it0+1:, it0]), label='log(<rho>)')
            #ax.plot(tau, np.log(rho_var_[1:, it0]), label='log(Var(rho))')
            ax2.plot(tau * tfactor, logrho_mean_[it0+1:, it0], label='<log(rho)>')
            ax3.plot(tau * tfactor, logrho_var_[it0+1:, it0], label='Var(log(rho))')
            #tau = t_[1:] - t_[0]
            #ax.plot(tau, logrho_mean_[1:, it0], label='<log(rho)>')
            #ax.plot(tau, logrho_var_[1:, it0], label='Var(log(rho))')
        #plt.legend()
        #ax.set_yscale("log")
        if args.save:
            plt.savefig(os.path.join(images_folder, "elong_t.png".format(t)))
        if args.show:
            plt.show()



if __name__ == "__main__":
    main()