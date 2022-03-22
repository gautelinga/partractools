from types import CellType
from analyze_eulerian_timeseries import *
from felbm2fem import make_folder_safe, get_fluid_domain, make_xdmf_mesh

import matplotlib.pyplot as plt
import meshio

from mpi4py import MPI

import scipy as sp
from scipy import sparse
from scipy import optimize as opt
from scipy import linalg

def parse_args():
    parser = argparse.ArgumentParser(description="FVM interpolated FELBM data")
    parser.add_argument("folder", type=str, help="Folder")
    parser.add_argument("-t0", type=float, default=None, help="")
    parser.add_argument("-t1", type=float, default=None, help="")
    parser.add_argument("-axis", type=int, default=0, help="Axis")
    parser.add_argument("--show", action="store_true", help="Show plot")
    parser.add_argument("--phase_by_density", action="store_true", help="Show plot")
    args = parser.parse_args()
    return args

def load_mesh(nx, ny, is_fluid):
    mesh = None
    if rank == 0:
        nodes = np.array([(i, j) for j, i in itertools.product(range(ny+1), range(nx+1))], dtype=float)
        elems = np.array([(i + j*(nx+1), i+1 + j*(nx+1), i+1 + (j+1)*(nx+1), i + (j+1)*(nx+1)) 
                          for j, i in itertools.product(range(ny), range(nx))], dtype=int)
        
        elems = elems[is_fluid, :]
        used_nodes = np.unique(elems)
        map_ids = np.zeros(used_nodes.max()+1, dtype=int)
        for i, j in zip(used_nodes, range(len(used_nodes))):
            map_ids[i] = j
        nodes = nodes[used_nodes, :]
        elems = map_ids[elems]
        faces = []

        mesh = [elems, faces, nodes]

    return mesh


if __name__ == "__main__":
    args = parse_args()

    felbm_folder = args.folder
    analysisfolder = os.path.join(felbm_folder, "Analysis")
    make_folder_safe(analysisfolder)

    timestamps = select_timestamps(felbm_folder, args.t0, args.t1)

    is_fluid_xy, (nx, ny, nz) = get_fluid_domain(felbm_folder)
    is_fluid_xy = is_fluid_xy #.T
    is_fluid = is_fluid_xy.flatten()

    #print("nx = {}, ny = {}".format(nx, ny))

    rho = np.zeros_like(is_fluid_xy, dtype=float)
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    #p = np.zeros_like(rho)

    nx = 3
    ny = 5

    ubar = np.ones((ny, nx))
    vbar = np.ones((ny, nx))

    #print(ubar, vbar)

    has_extra = False

    diagonals_Px = [0.5*np.ones(nx * ny), 0.5*np.ones(nx*ny-1)]
    diagonals_Py = [0.5*np.ones(nx * ny), 0.5*np.ones(nx*ny-nx), 0.5*np.ones(nx)]

    diagonals_Gx = [-np.ones(nx * ny), np.ones(nx*ny-1)]
    diagonals_Gy = [-np.ones(nx * ny), np.ones(nx*ny-nx), np.ones(nx)]

    Px = sparse.lil_matrix(sparse.diags(diagonals_Px, [0, -1]))
    for j in range(0, ny):
        Px[nx * j, nx * (j+1) - 1] = 0.5
    for j in range(1, ny):
        Px[nx * j, nx * j - 1] = 0.
    Py = sparse.lil_matrix(sparse.diags(diagonals_Py, [0, -nx, nx*ny-nx]))

    Gx = sparse.lil_matrix((ny*nx, (ny+1)*(nx+1)+2*has_extra))
    for diagonal, offset in zip(diagonals_Gx, [0, 1]):
        for i, val in enumerate(diagonal):
            Gx[i, i + offset] = val
    for j in range(0, ny):
        Gx[nx * (j+1) - 1, nx * j ] = 1.0
    for j in range(1, ny):
        Gx[nx * j-1, nx * j] = 0.
        #pass

    Gy = sparse.lil_matrix((ny*nx, (ny+1)*(nx+1)+2*has_extra))
    for diagonal, offset in zip(diagonals_Gy, [0, nx, -nx*ny+nx]):
        ioffset, joffset = 0, 0
        if offset < 0:
            ioffset = -offset
        else:
            joffset = offset
        for i, val in enumerate(diagonal):
            Gy[i + ioffset, i + joffset] = val

    R = sparse.lil_matrix(((ny+1)*(nx+1)+2*has_extra, ny * nx + 2*has_extra))
    for j in range(ny):
        for i in range(nx):
            R[i + j * (nx + 1), i + j * nx] = 1.
    for i in range(nx):
        R[i + (nx + 1) * ny, i] = 1
        if has_extra:
            R[i + (nx + 1) * ny, ny * nx + 1] = 1
    for j in range(ny):
        R[nx + j * (nx+1), j * nx] = 1
        if has_extra:
            R[nx + j * (nx+1), ny * nx] = 1
    R[nx + ny + nx * ny, 0] = 1
    if has_extra:
        R[nx + ny + nx * ny, ny * nx] = 1
        R[nx + ny + nx * ny, ny * nx + 1] = 1
        R[nx + ny + nx * ny + 1, ny * nx] = 1
        R[nx + ny + nx * ny + 2, ny * nx + 1] = 1

    RD = sparse.lil_matrix((ny*nx+2*has_extra, ny*nx - 1 + 2*has_extra))
    for i in range(nx*ny-1+2*has_extra):
        RD[i+1, i] = 1

    Mx = Gx #* R * RD
    My = Gy #* R * RD

    yy, xx = np.meshgrid(range(nx+1), range(ny+1))
    psi_test = xx - yy

    utest = Gy * psi_test.flatten()

    plt.imshow(utest.reshape((ny, nx)))
    plt.show()

    duy = Gx 


    A = sparse.bmat([[My], [Mx]])
    b = np.hstack((ubar.flatten(), vbar.flatten()))

    print(A.shape, b.shape)

    #A2 = 2 * My.T * My
    #B2 = 2 * Mx.T * Mx

    #a = 2 * My.T * ubar.flatten()
    #b = 2 * Mx.T * vbar.flatten()
    
    z, res, rank, s = sparse.linalg.lsqr(A, b)[:4]

    print(z)

    #print(A2.shape, B2.shape, a.shape, b.shape)

    """
    #Pu = sparse.lil_matrix(Pu)
    Pu1 = np.zeros((nx * ny, nx * ny))
    Pv1 = np.zeros((nx * ny, nx * ny))
    Gx1 = np.zeros(Gx.shape)
    Gy1 = np.zeros(Gx.shape)
    for i in range(nx):
        for j in range(ny):
            for k in range(nx):
                for l in range(ny):
                    Gx1[i + nx * j, k + nx * l] = ( l == j ) * ( - ( k == i ) + ( k == ((i+1) % nx) ) )
                    Gy1[i + nx * j, k + nx * l] = ( - ( l == j ) + ( l == ((j+1) % ny) ) ) * ( i == k )
                    #Pu1[i + nx * j, k + nx * l] = 0.5 * ( l == j ) * ( ( k == i ) + ( k == ((i-1) % nx) ) )
                    #Pv1[i + nx * j, k + nx * l] = 0.5 * ( ( l == j ) + ( l == ((j-1) % ny) ) ) * ( i == k )
    #print(np.linalg.norm(Pu1 - Pu.toarray()))
    #print(np.linalg.norm(Pv1 - Pv.toarray()))
    print(np.linalg.norm(Gx1 - Gx.toarray()))
    print(np.linalg.norm(Gy1 - Gy.toarray()))

    """

    u0 = np.zeros((ny, nx))
    v0 = np.zeros((ny, nx))
    for it, timestamp in enumerate(timestamps):
        t = timestamp[0]
        h5filename = timestamp[1]
        mpi_print(t, h5filename)
        with h5py.File(os.path.join(felbm_folder, h5filename), "r") as h5f:
            u0[:, :] = np.array(h5f["u_x"]).reshape((nz, ny, nx))[nz // 2, :, :]
            v0[:, :] = np.array(h5f["u_y"]).reshape((nz, ny, nx))[nz // 2, :, :]

        ubar = Px.dot(u0.flatten())
        vbar = Py.dot(v0.flatten())

        a = 2 * My.T * ubar
        b = 2 * Mx.T * vbar
        x = sparse.linalg.spsolve(A2+B2, a+b)
        print(x)

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(ubar.reshape((ny, nx)))
        ax[1].imshow(vbar.reshape((ny, nx))-0.5*v0)
        ax[2].imshow(v0)
        plt.show()


    # mesh = load_mesh(nx, ny, is_fluid, analysisfolder)

    # print(mesh)

    #nx = 13
    #ny = 7

    #A = np.zeros((nx * ny, nx * ny))
    #B = np.zeros((nx * ny, nx * ny))
    #Y = np.zeros((nx * ny, nx * ny))
    #Z = np.zeros((nx * ny, nx * ny))

    print("Dimensions:", nx, ny)

