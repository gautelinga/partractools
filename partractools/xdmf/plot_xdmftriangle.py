import argparse
import os

import matplotlib.pyplot as plt
from matplotlib import tri
from matplotlib.collections import LineCollection
import h5py
import numpy as np

from partractools.common.utils import Params, get_h5data_location, GenParams, mpi_root, mpi_rank, mpi_size, parse_xdmf, get_first_one, mpi_min, mpi_max


def parse_args():
    parser = argparse.ArgumentParser(description="Plot pretty Lagrangian and Eulerian dynamics")
    parser.add_argument("input_folder", type=str, help="Input folder")
    parser.add_argument("--skip", type=int, default=1, help="Skip")
    parser.add_argument("--size", type=float, default=20.0, help="Size")
    parser.add_argument("--start", type=int, default=None, help="Start id")
    parser.add_argument("--stop", type=int, default=None, help="Stop id")
    parser.add_argument("-s0", type=float, default=3.0, help="Initial thickness (if applicable)")
    parser.add_argument("--arrows", action="store_true", help="Plot with arrows (if applicable)")
    parser.add_argument("--phasesep", type=float, default=0, help="Separate solute in phases (>/< 0)")
    return parser.parse_args()

def trim_path(path):
    size = len(path) + 1
    while len(path) < size:
        size = len(path)
        path = path.replace("//", "/")
    return path

def get_tE_loc(tE_, tL):
    it = np.searchsorted(tE_, tL)
    if it == 0:
        tE_loc_ = [tE_[it]]
        w_ = [1.]
    elif it >= len(tE_):
        tE_loc_ = [tE_[-1]]
        w_ = [1.]
    else:
        tE_loc_ = [tE_[it-1], tE_[it]]
        alpha_t = (tL-tE_[it-1])/(tE_[it]-tE_[it-1])
        w_ = [1-alpha_t, alpha_t]
    return tE_loc_, w_

def compute_lines(edges):
    """ edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5)] """
    v2e = dict()
    for iedge, edge in enumerate(edges):
        for iv in edge:
            if iv in v2e:
                v2e[iv].add(iedge)
            else:
                v2e[iv] = set([iedge])

    lines = []
    #edge_ids = []
    while len(v2e):
        line_loc = []
        #edge_loc = []
        iv0 = get_first_one(v2e)
        #line_loc.append(iv0)
        next_edges = v2e.pop(iv0) #.pop()
        while len(next_edges) > 1:
            next_edges.pop()
        iv = iv0
        closed = False
        while (len(next_edges) == 1):
            ie = next_edges.pop()
            next_nodes = set(edges[ie])
            next_nodes.remove(iv)
            assert(len(next_nodes) == 1)
            iv_prev = iv
            iv = next_nodes.pop()
            line_loc.append((iv_prev, iv, ie))
            #lines.append((iv_prev, iv, ie))
            #line_loc.append(iv)
            #edge_loc.append(ie)
            if iv == iv0:
                closed = True
                break
            next_edges = set(v2e.pop(iv))
            next_edges.remove(ie)
        #line_nodes.append(line_loc)
        #edge_ids.append(edge_loc)
        lines.append(line_loc)
    return lines

def mpi_get_logrho_minmax(posf, tstepsL):
    logrho_min = 0.
    logrho_max = 0.
    for _, tL in tstepsL[mpi_rank::mpi_size]:
        # Lagrangian
        posft, cat = posf[tL]
        with h5py.File(posft, "r") as h5f:
            logrho = np.log(h5f[cat]["dl"][:] / h5f[cat]["dl0"][:])
            logrho_max = max(logrho_max, logrho.max())
            logrho_min = min(logrho_min, logrho.min())
    logrho_max = mpi_max(logrho_max)
    logrho_min = mpi_min(logrho_min)
    return logrho_min, logrho_max

def mpi_get_field_minmax(field, posf, tstepsL):
    w_min = np.inf
    w_max = -np.inf
    for _, tL in tstepsL[mpi_rank::mpi_size]:
        posft, cat = posf[tL]
        with h5py.File(posft, "r") as h5f:
            w = h5f[cat][field][:]
            w_max = max(w_max, w.max())
            w_min = min(w_min, w.min())
    w_max = mpi_max(w_max)
    w_min = mpi_min(w_min)
    return w_min, w_max

def has_field(field, posft, cat):
    with h5py.File(posft, "r") as h5f:
        return field in h5f[cat]


if __name__ == "__main__":
    args = parse_args()

    params = Params(args.input_folder)
    t0 = params.get_tmin()

    folder = os.path.normpath(params["folder"])
    rootpath = os.path.normpath(os.path.join(args.input_folder, os.path.join(*[".."]*len(folder.split("/")))))

    df_params = GenParams(os.path.join(rootpath, "dolfin_params.dat"))

    posf = get_h5data_location(args.input_folder)

    tsfolder = rootpath
    imgfolder = os.path.join(args.input_folder, "Images")
    if mpi_root and not os.path.exists(imgfolder):
        os.makedirs(imgfolder)

    u_xdmffile = os.path.join(tsfolder, df_params["u"])
    phi_xdmffile = os.path.join(tsfolder, "phi_from_tstep_0.xdmf")  # update later

    dsetsE, topology_address, geometry_address = parse_xdmf(u_xdmffile, True)
    dsetsE = dict(dsetsE)

    dsetsE_phi = dict(parse_xdmf(phi_xdmffile, False))

    with h5py.File(topology_address[0], "r") as h5f:
        cells = h5f[topology_address[1]][:]

    with h5py.File(geometry_address[0], "r") as h5f:
        xy = h5f[geometry_address[1]][:]
    
    tE_ = sorted(dsetsE.keys())
    assert(np.linalg.norm(np.array(tE_)-np.array(sorted(dsetsE_phi.keys()))) < 1e-9)

    xy_max = xy.max(axis=0)
    xy_min = xy.min(axis=0)
    Lx = xy_max[0]-xy_min[0]
    Ly = xy_max[1]-xy_min[1]

    triang = tri.Triangulation(xy[:, 0], xy[:, 1], cells)

    tE_ = sorted(tE_)
    tL_ = sorted(posf.keys())

    start = 0 if args.start is None else args.start
    stop = len(tL_) if args.stop is None else args.stop

    tstepsL = list(enumerate(tL_))[start:stop:args.skip]

    levels = [-10, 0, 10]
    colors = ['#ffffff', '#a0c0ff']

    c = np.zeros(len(xy))
    c_1 = np.zeros_like(c)

    has_edges = has_field("edges", *posf[tL_[0]])

    if has_edges:
        logrho_min, logrho_max = mpi_get_logrho_minmax(posf, tstepsL)
    else:
        w_min, w_max = mpi_get_field_minmax("w", posf, tstepsL)

    for i, tL in tstepsL[mpi_rank::mpi_size]:
        if mpi_rank == 0:
            print(f"Step: {i}, Time = {tL} / {tL_[-1]}")
        
        tE_loc_, w_loc_ = get_tE_loc(tE_, tL)

        # Eulerian
        c[:] = 0.
        for tE, w in zip(tE_loc_, w_loc_):
            if w > 1e-7:
                phi_file, dset_loc = dsetsE_phi[tE]
                with h5py.File(phi_file, "r") as h5f:
                    c[:] = w * h5f[dset_loc][:].flatten()

        # Lagrangian
        posft, cat = posf[tL]
        with h5py.File(posft, "r") as h5f:
            pts = h5f[cat]["points"][:][:, :2]
            phi = h5f[cat]["rho"][:, 0]
            if has_edges:
                edges = h5f[cat]["edges"][:]
                dl = h5f[cat]["dl"][:]
                dl0 = h5f[cat]["dl0"][:]
            else:
                w = h5f[cat]["w"][:][:, 0]
                n = h5f[cat]["n"][:]
            #cc = h5f[cat]["c"][:]

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

        #if args.phasesep != 0:
        #    c_intp = tri.LinearTriInterpolator(triang, c)
        #    c_intp_vals = c_intp(pts[:, 0], pts[:, 1])
        
        if has_edges:
            cvnorm = plt.Normalize(logrho_min, logrho_max)

            lines = compute_lines(edges)
            segs = []
            lws = []
            cvs = []
            for line in lines:
                for v1, v2, ie in line:
                    x1 = pts[v1, :]
                    x2 = pts[v2, :]
                    inside_domain = True
                    if args.phasesep < 0:
                        inside_domain = phi[v1] < 0 and phi[v2] < 0
                        #    inside_domain = c_intp_vals[v1] < 0 and c_intp_vals[v2] < 0
                    elif args.phasesep > 0:
                        inside_domain = phi[v1] > 0 and phi[v2] > 0
                        #    inside_domain = c_intp_vals[v1] > 0 and c_intp_vals[v2] > 0

                    if inside_domain:
                        rho = dl[ie, 0] / dl0[ie, 0]

                        cell_1 = tuple([int(ind) for ind in np.floor( (x1[:] - xy_min[:]) / (xy_max[:] - xy_min[:]))])
                        cell_2 = tuple([int(ind) for ind in np.floor( (x2[:] - xy_min[:]) / (xy_max[:] - xy_min[:]))])
                        cells = set({cell_1, cell_2})

                        for indi, indj in cells:
                            dx = np.array([indi*Lx, indj*Ly])
                            segs.append([x1-dx, x2-dx])
                            lws.append(args.s0/rho)
                            cvs.append(np.log(rho))

            lc = LineCollection(segs, cmap='inferno', norm=cvnorm, zorder=3) #, linewidths=lws) # linewidths=lwidths)
            lc.set_array(cvs)
            #lc.set_linewidth(lws)
            #lc.set_linewidth(args.s0)
            ax.add_collection(lc)

            #if args.phasesep < 0:
            #    ax.tricontourf(triang, c, levels=levels[:-1], colors=colors[:-1], zorder=10)
            #elif args.phasesep > 0:
            #    ax.tricontourf(triang, c, levels=levels[1:], colors=colors[1:], zorder=10)
        else:
            # print(pts.shape, w.shape)
            wnorm = plt.Normalize(w_min, w_max)

            cell_x = np.floor((pts[:, 0] - xy_min[0]) / Lx )
            cell_y = np.floor((pts[:, 1] - xy_min[1]) / Ly )
            X = pts[:, 0] - cell_x * Lx
            Y = pts[:, 1] - cell_y * Ly

            if args.arrows:
                ax.quiver(X, Y, n[:, 0], n[:, 1], w, norm=wnorm, scale=1.0 / 0.1, cmap="inferno",
                          pivot='mid', units="x", headwidth=0, headlength=0, headaxislength=0)
            else:
                ax.scatter(X, Y, c=w, marker=".", norm=wnorm, s=4, cmap="inferno")

        plt.tight_layout()
        plt.savefig(os.path.join(imgfolder, "phase_{:06d}.png".format(i)))
        plt.close()
        #plt.show()

    if mpi_root:
        print("now run:")
        print(f"ffmpeg -framerate 25 -i {imgfolder}/phase_%06d.png -c:v libx264 -pix_fmt yuv420p {imgfolder}/out.mp4")
    # h5f.close()
