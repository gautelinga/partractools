import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
from partractools.common.utils import Params, get_h5data_location, compute_lines
from matplotlib import colors as mcolors
from scipy.interpolate import UnivariateSpline, make_interp_spline


def parse_args():
    parser = argparse.ArgumentParser(description="Plot velocity")
    parser.add_argument("folder", type=str, help="Folder")
    parser.add_argument("-t0", type=float, default=0.0, help="t0")
    parser.add_argument("--show", action="store_true", help="Show plot")
    parser.add_argument("-s0", type=float, default=0.1, help="initial thickness")
    parser.add_argument("-Npts", type=int, default=100000, help="Number of points to interpolate to")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    params = Params(args.folder)
    t0 = params.get_tmin()
    params.get("Lx", t0)

    Lx = float(params.get("Lx", t0))
    Ly = float(params.get("Ly", t0))
    Lz = float(params.get("Lz", t0))
    L = [Lx, Ly, Lz]

    posf = get_h5data_location(args.folder)

    imgfolder = os.path.join(args.folder, "Images")
    statsfolder = os.path.join(args.folder, "Statistics")
    if not os.path.exists(imgfolder):
        os.makedirs(imgfolder)

    ts = list(sorted(posf.keys()))
    print(ts)
    assert(args.t0 in ts)
    tq = [args.t0]

    for it, t in enumerate(ts):
        posft, cat = posf[t]
        with h5py.File(posft, "r") as h5f:
            pts = np.array(h5f[cat]["points"])[:, :2]
            edges = np.array(h5f[cat]["edges"])
            dl = np.array(h5f[cat]["dl"])
            dl0 = np.array(h5f[cat]["dl0"])
            c = np.array(h5f[cat]["c"])
        rho = dl/dl0


        lines = compute_lines(pts, edges)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for line, closed in lines:
            new_edges = np.array(line, dtype=int)
            x0 = pts[new_edges[:, 0], :]
            x1 = pts[new_edges[:, 1], :]
            dl_ = dl[new_edges[:, 2], 0]
            dl0_ = dl0[new_edges[:, 2], 0]
            logs_ = np.log(args.s0 * dl0_ / dl_)

            x_jnt = np.zeros((x0.shape[0]+1, x0.shape[1]))
            x_jnt[:-1, :] = x0
            x_jnt[-1, :] = x1[-1, :]
            
            logs_jnt = np.zeros(x0.shape[0]+1)
            logs_jnt[0] = logs_[0]
            logs_jnt[-1] = logs_[-1]
            logs_jnt[1:-1] = 0.5*(logs_[:-1]+logs_[1:])

            distance = np.cumsum( np.sqrt(np.sum( np.diff(x_jnt, axis=0)**2, axis=1 )) )
            distance = np.insert(distance, 0, 0) #/distance[-1]

            # build splines for each coord
            splines = [UnivariateSpline(distance, coords, k=3, s=.001) for coords in x_jnt.T]
            logsspline = make_interp_spline(distance, logs_jnt)
            dsplines = [spline.derivative(1) for spline in splines]
            ddsplines = [spline.derivative(2) for spline in splines]
            #alpha = np.linspace(0, 1, 100000)
            alpha  = np.linspace(0, distance[-1], args.Npts)
            x_fit = np.vstack([np.array(spl(alpha)) for spl in splines]).T
            logs_fit = np.array(logsspline(alpha))
            s_fit = np.exp(logs_fit)

            t_fit = np.vstack([np.array(dspl(alpha)) for dspl in dsplines]).T
            tabs = np.linalg.norm(t_fit, axis=1)
            t_fit[:, 0] /= tabs
            t_fit[:, 1] /= tabs
            n_fit = np.zeros_like(t_fit)
            n_fit[:, 0] = -t_fit[:, 1]
            n_fit[:, 1] = t_fit[:, 0]

            tt_fit = np.vstack([np.array(ddspl(alpha)) for ddspl in ddsplines]).T
            R_fit = 1./np.linalg.norm(tt_fit, axis=1)

            s_eff = np.copy(s_fit)
            ids = s_fit > args.s0 # R_fit
            s_eff[ids] = args.s0 # R_fit[s_fit > R_fit/2]/2

            x = np.hstack([x_fit[:, 0] + s_eff/2 * n_fit[:, 0],
                        x_fit[::-1, 0] - s_eff[::-1]/2 * n_fit[::-1, 0]])
            y = np.hstack([x_fit[:, 1] + s_eff/2 * t_fit[:, 1],
                        x_fit[::-1, 1] - s_eff[::-1]/2 * n_fit[::-1, 1]])

            ax.fill(x, y)

        ax.set_xlim(-Lx/2, Lx/2)
        ax.set_ylim(-Ly/2, Ly/2)
        ax.set_aspect('equal')
        plt.tight_layout()

        plt.savefig(os.path.join(imgfolder, "thickens_{:06d}.png".format(it)))
        if args.show:
            plt.show()
        plt.close()