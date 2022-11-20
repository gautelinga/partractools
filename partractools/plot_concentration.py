import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
from partractools.common.utils import Params
from matplotlib.collections import LineCollection, CircleCollection
from matplotlib import colors as mcolors
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import make_lsq_spline, make_interp_spline
from matplotlib.patches import Circle
from plot_thickens import get_h5data_location, compute_lines

def parse_args():
    parser = argparse.ArgumentParser(description="Plot velocity")
    parser.add_argument("folder", type=str, help="Folder")
    parser.add_argument("-t0", type=float, default=0.0, help="t0")
    parser.add_argument("--show", action="store_true", help="Show plot")
    parser.add_argument("-s0", type=float, default=0.1, help="initial thickness")
    parser.add_argument("-Dm", type=float, default=0., help="Diffusion coefficient")
    parser.add_argument("-Dl_max", type=float, default=0.05, help="Maximum reconstruction interval")
    #parser.add_argument("-Npts", type=int, default=100000, help="Number of points to interpolate to")
    args = parser.parse_args()
    return args


def ellipse(nx, ny):
    x, y = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx))
    return x**2 + y**2 <= 1 + 1./np.sqrt(nx*ny)

def gaussian_ellipse(x, y, t, n, Dl_t, Dl_n):
    arg = (x * t[0] + y * t[1])**2/Dl_t**2 + (x * n[0] + y * n[1])**2/Dl_n**2
    return np.exp(-arg) #/factor

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
    if (args.t0 in ts):
        tq = [args.t0]
    else:
        tq = ts

    for it, t in enumerate(tq):
        posft, cat = posf[t]
        with h5py.File(posft, "r") as h5f:
            pts = np.array(h5f[cat]["points"])[:, :2]
            edges = np.array(h5f[cat]["edges"])
            dl = np.array(h5f[cat]["dl"])
            dl0 = np.array(h5f[cat]["dl0"])
            c = np.array(h5f[cat]["c"])
            tau = np.array(h5f[cat]["tau"])
        #x0 = np.remainder(data_0[:, 0].flatten(), L[0])
        #y0 = np.remainder(data_0[:, 1].flatten(), L[1])
        rho = dl/dl0

        Dl = min(np.mean(1./rho * np.sqrt(args.s0**2 + 4 * args.Dm * tau)), args.Dl_max)
        print(f"Dl={Dl}")
        # edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5)]

        lines = compute_lines(pts, edges)

        Nx = 1000
        ddx = Lx/Nx
        Ny = round(Ly / ddx)
        ddy = Ly/Ny
        X, Y = np.meshgrid(np.linspace(-Lx/2, Lx/2, Nx), np.linspace(-Ly/2, Ly/2, Ny))
        C = np.zeros_like(X)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for line, closed in lines:
            new_edges = np.array(line, dtype=int)
            x0 = pts[new_edges[:, 0], :]
            x1 = pts[new_edges[:, 1], :]
            dl_ = dl[new_edges[:, 2], 0]
            dl0_ = dl0[new_edges[:, 2], 0]
            logs_ = np.log(args.s0 * dl0_ / dl_)
            logtau_ = np.log(tau[new_edges[:, 2], 0])

            x_jnt = np.zeros((x0.shape[0]+1, x0.shape[1]))
            x_jnt[:-1, :] = x0
            x_jnt[-1, :] = x1[-1, :]
            
            logs_jnt = np.zeros(x0.shape[0]+1)
            logs_jnt[0] = logs_[0]
            logs_jnt[-1] = logs_[-1]
            logs_jnt[1:-1] = 0.5*(logs_[:-1]+logs_[1:])

            logtau_jnt = np.zeros_like(logs_jnt)
            logtau_jnt[0] = logtau_[0]
            logtau_jnt[-1] = logtau_[-1]
            logtau_jnt[1:-1] = 0.5*(logtau_[1:]+logtau_[:-1])

            distance = np.cumsum( np.sqrt(np.sum( np.diff(x_jnt, axis=0)**2, axis=1 )) )
            distance = np.insert(distance, 0, 0) #/distance[-1]
            Nseg = round(distance[-1]/Dl)
            Dl_loc = distance[-1]/Nseg

            # build splines for each coord
            splines = [UnivariateSpline(distance, coords, k=3, s=.001) for coords in x_jnt.T]
            logsspline = make_interp_spline(distance, logs_jnt, k=1)
            logtauspline = make_interp_spline(distance, logtau_jnt, k=1)
            
            dsplines = [spline.derivative(1) for spline in splines]
            ddsplines = [spline.derivative(2) for spline in splines]
            #alpha = np.linspace(0, 1, 100000)
            alpha  = np.linspace(0, distance[-1], Nseg+1)
            x_fit = np.vstack([np.array(spl(alpha)) for spl in splines]).T
            logs_fit = np.array(logsspline(alpha))
            s_fit = np.exp(logs_fit)
            logtau_fit = np.array(logtauspline(alpha))
            tau_fit = np.exp(logtau_fit)
            #plt.plot(distance, (s_jnt))
            #plt.plot(alpha, s_fit)
            #plt.plot(alpha, logtau_fit)
            #
            #plt.show()

            t_fit = np.vstack([np.array(dspl(alpha)) for dspl in dsplines]).T
            tabs = np.linalg.norm(t_fit, axis=1)
            t_fit[:, 0] /= tabs
            t_fit[:, 1] /= tabs
            n_fit = np.zeros_like(t_fit)
            n_fit[:, 0] = -t_fit[:, 1]
            n_fit[:, 1] = t_fit[:, 0]

            tt_fit = np.vstack([np.array(ddspl(alpha)) for ddspl in ddsplines]).T
            R_fit = 1./np.linalg.norm(tt_fit, axis=1)

            #plt.plot(alpha, R_fit)
            #plt.show()

            #ax.plot(x_jnt[:, 0], x_jnt[:, 1], '.')
            #plt.plot(x_fit[:, 0], x_fit[:, 1])

            #plt.show()
            s_eff = np.copy(s_fit)
            ids = s_fit > args.s0 # R_fit
            #not_ids = np.logical_not(ids)
            s_eff[ids] = args.s0 # R_fit[s_fit > R_fit/2]/2
            tau_eff = np.copy(tau_fit)
            #tau_eff[ids] = (s_fit[ids]/args.s0)**2

            #ax.plot(x, y, '.')
            #xx = x_fit[:, :]
            #ss = s_fit[:]
            #circles = CircleCollection(ss**2, offsets=xx, transOffset=ax.transData, color='green')
            #ax.add_collection(circles)
            #for xi, ssi in zip(xx, ss):
            #    ax.add_artist(Circle(xy=(xi[0], xi[1]), radius=min(ssi/2, args.s0/5), color='green'))
            factor_fit = np.sqrt(1 + 4*args.Dm/args.s0**2*tau_eff)
            rmax = 4 * s_eff[:] * factor_fit[:]
            rmax[rmax < Dl_loc] = Dl_loc

            curved = R_fit < s_eff[:] * factor_fit[:]
            #nstep = Dl_loc
            for i in range(5):
                sub = np.logical_or(curved[2*i:], curved[:len(curved)-2*i])
                curved[i:len(curved)-i] = sub

            print("curved:", sum(curved))

            #for i, (x, t, n, s, factor) in enumerate(zip(x_fit[:, :], t_fit[:, :], n_fit[:, :], s_eff, factor_fit)):
            for i in range(len(x_fit)):
                # ax.add_artist(Circle(xy=(x, y), radius=s))
                ixm = max(0, int((x_fit[i, 0] - rmax[i] + Lx/2)/ddx))
                ixp = min(Nx, int((x_fit[i, 0] + rmax[i] + Lx/2)/ddx))
                iym = max(0, int((x_fit[i, 1] - rmax[i] + Ly/2)/ddy))
                iyp = min(Ny, int((x_fit[i, 1] + rmax[i] + Ly/2)/ddy))
                # if iym >= 0 and iyp < C.shape[0]:
                #     #C_loc = ellipse(ixp-ixm, iyp-iym)
                #     #print(ixm, ixp, iym, iyp, C_loc.shape)

                if not curved[i]:
                    Dl_t = Dl_loc
                    Dl_n = s_fit[i] * factor_fit[i]
                else:
                    Dl_t = Dl_n = np.sqrt(Dl_loc * s_fit[i] * factor_fit[i])
                C[iym:iyp, ixm:ixp] += gaussian_ellipse(X[iym:iyp, ixm:ixp]-x_fit[i, 0], Y[iym:iyp, ixm:ixp]-x_fit[i, 1],
                                                        t_fit[i, :], n_fit[i, :], Dl_t, Dl_n)/factor_fit[i]
                #C += gaussian(X-x[0], Y-x[1], t, n, s, factor, Dl_loc)

            C /= 1.7726
            C[C > 1] = 1
            # plt.imshow(C)
            # plt.show()

            #plt.scatter(x_fit[:, 0], x_fit[:, 1], s=s_fit)

            #ax.quiver(x_fit[:, 0], x_fit[:, 1], t_fit[:, 0], t_fit[:, 1], s_fit[:], angles='xy', scale_units='xy', scale=100)
            
            #line_segments = LineCollection(segments,
            #                               linewidths=lw,
            #                               linestyles='solid',
            #                               color="blue")

            c = ax.pcolormesh(X, Y, C, cmap="gist_gray_r")
            #    if curved[i]:
            #ax.scatter(x_fit[curved, 0], x_fit[curved, 1])

            #plt.colorbar(c)
            c.set_clim(0,0.5)
            #ax.add_collection(line_segments)
            #ax.set_xlim(-Lx/2, Lx/2)
            #ax.set_ylim(-Ly/2, Ly/2)
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.axis('off')

            plt.savefig(os.path.join(imgfolder, "conc_{:06d}.png".format(it)))
            if args.show:
                plt.show()
            plt.close()


    if False:
        line_segments = LineCollection(segments,
                                    linewidths=lw,
                                    linestyles='solid',
                                    color="blue")

        fig, ax = plt.subplots(1, 1, figsize=(4, 6))

        ax.add_collection(line_segments)

        ax.set_xlim(-Lx/2, Lx/2)
        ax.set_ylim(-Ly/2, Ly/2)
        ax.set_aspect('equal')

        plt.tight_layout()
        if args.show:
            plt.show()