from types import CellType
from analyze_eulerian_timeseries import *
from felbm2fem import make_folder_safe, get_fluid_domain, make_xdmf_mesh

import matplotlib.pyplot as plt
import meshio

from mpi4py import MPI

import scipy as sp
from scipy import sparse
from scipy import optimize as opt

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
    is_fluid = is_fluid_xy.flatten()

    #print("nx = {}, ny = {}".format(nx, ny))

    rho = np.zeros_like(is_fluid_xy, dtype=float)
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    #p = np.zeros_like(rho)

    # mesh = load_mesh(nx, ny, is_fluid, analysisfolder)

    # print(mesh)

    #nx = 13
    #ny = 7

    #A = np.zeros((nx * ny, nx * ny))
    #B = np.zeros((nx * ny, nx * ny))
    #Y = np.zeros((nx * ny, nx * ny))
    #Z = np.zeros((nx * ny, nx * ny))

    diagonals_A = [np.ones(nx * ny), np.ones(nx*ny-1)]
    A = 0.5 * sparse.lil_matrix(sparse.diags(diagonals_A, [0, 1]))
    for j in range(ny):
        A[nx * (j + 1) - 1, nx * j] += 0.5
    for j in range(ny-1):
        A[nx * (j + 1) - 1, nx * (j + 1)] -= 0.5

    diagonals_B = [np.ones(nx * ny), np.ones(nx * ny - nx), np.ones(nx)]
    B = 0.5 * sparse.lil_matrix(sparse.diags(diagonals_B, [0, nx,-nx*ny+nx]))
    #for i in range(nx):
    #    B[ny * (i + 1) - 1, ny * i] += 0.5
    #for i in range(nx-1):
    #    B[ny * (i + 1) - 1, ny * (i+1)] -= 0.5

    diagonals_Y = [-1 * np.ones(nx*ny), np.ones(nx*ny-1)]
    Y = sparse.lil_matrix(sparse.diags(diagonals_Y, [0, 1]))
    for j in range(ny):
        Y[nx * (j + 1) - 1, nx * j] += 1
    for j in range(ny-1):
        Y[nx * (j + 1) - 1, nx * (j + 1)] -= 1

    diagonals_Z = [-np.ones(nx*ny), np.ones(nx * ny - nx), np.ones(nx)]
    Z = sparse.lil_matrix(sparse.diags(diagonals_Z, [0, nx, -nx*ny+nx]))
    
    #for i in range(nx):
    #    Z[ny * (i + 1) - 1, ny * i] += 1
    #for i in range(nx-1):
    #    Z[ny * (i + 1) - 1, ny * (i+1)] -= 1

    u_is_fluid_xy = np.ones((ny, nx), dtype=bool)
    v_is_fluid_xy = np.ones((ny, nx), dtype=bool)

    for i in range(nx):
        for j in range(ny):
            if not is_fluid_xy[j, i]:
                u_is_fluid_xy[j, i] = False
                u_is_fluid_xy[j, (i+1) % nx] = False
                v_is_fluid_xy[j, i] = False
                v_is_fluid_xy[(j+1) % ny, i] = False

    u_is_fluid = u_is_fluid_xy.flatten()
    v_is_fluid = v_is_fluid_xy.flatten()

    #u_is_fluid[:] = True
    #v_is_fluid[:] = True

    """
    AB = sparse.csc_matrix(sparse.bmat([[A, None], [None, B]]))
    DivFreeMat = sparse.hstack((Y, Z))
    nilvec = np.zeros(nx*ny)

    res = DivFreeMat.dot(np.ones(2*nx*ny))
    print(DivFreeMat.shape, res.shape)

    constraint = opt.LinearConstraint(DivFreeMat, lb=nilvec, ub=nilvec)
    b = np.ones(2*nx*ny)
    x0 = np.zeros(2*nx*ny)

    def fun(x):
        return np.linalg.norm(AB.dot(x) - b)**2

    res = opt.minimize(fun, x0, constraints=(constraint))

    print(res.x)
    """

    """
    A2 = np.zeros((nx * ny, nx * ny))
    B2 = np.zeros((nx * ny, nx * ny))
    Y2 = np.zeros((nx * ny, nx * ny))
    Z2 = np.zeros((nx * ny, nx * ny))

    for i in range(nx):
        for j in range(ny):
            for k in range(nx):
                for l in range(ny):
                    A2[ i + nx * j, k + nx * l] = 0.5 * ( ( (i + 1) % nx == k) + (i == k) ) * (j == l)
                    B2[ i + nx * j, k + nx * l] = 0.5 * (i == k) * ( ( (j + 1) % ny == l ) + (j == l))
                    Y2[ i + nx * j, k + nx * l] = ( ( (i+1) % nx == k ) - ( i == k) ) * ( j == l )
                    Z2[ i + nx * j, k + nx * l] = ( i == k ) * ( ( (j + 1) % ny == l ) - ( j == l ) )
    #print(A)
    #print(abs(A.toarray() - A2))
    #print(abs(B.toarray() - B2))
    #print(abs(Y.toarray() - Y2))
    #print(abs(Z.toarray() - Z2))
    print(abs(A.toarray() - A2).max())
    print(abs(B.toarray() - B2).max())
    print(abs(Y.toarray() - Y2).max())
    print(abs(Z.toarray() - Z2).max())
    #print(2 * np.matmul(A.T, A))
    #print(2 * np.matmul(B.T, B))
    """

    A = sparse.csc_matrix(A)
    B = sparse.csc_matrix(B)
    Y = sparse.csc_matrix(Y)
    Z = sparse.csc_matrix(Z)
    
    #A = sparse.lil_matrix(A)
    A = A[:, u_is_fluid]
    B = B[:, v_is_fluid]
    Yn = Y[:, u_is_fluid][u_is_fluid, :]
    Zn = Z[:, v_is_fluid][v_is_fluid, :]
    Yt = Y[:, v_is_fluid][v_is_fluid, :]
    Zt = Z[:, u_is_fluid][u_is_fluid, :]

    #print(A.shape)

    gamma = 1.0

    # M = sparse.csc_matrix(sparse.bmat([[2 * A.T * A, None, Y.T], [None, 2*B.T*B, Z.T], [Y, Z, None]]))
    AA = 2 * A.T * A + gamma * 2 * (Yn.T * Yn + Zt.T * Zt)
    #print(AA.shape)
    #AA = AA[:, u_is_fluid][u_is_fluid, :]

    BB = 2 * B.T * B + gamma * 2 * (Zn.T * Zn + Yt.T * Yt)
    #BB = BB[:, v_is_fluid][v_is_fluid, :]
    
    #Z = Z[:, u_is_fluid][u_is_fluid, :]
    #Y = Y[:, v_is_fluid][v_is_fluid, :]

    # AB = sparse.csc_matrix(sparse.bmat([[AA, None], [None, BB]]))
    ABC = sparse.csc_matrix(sparse.bmat([[AA, None, Yn.T], [None, BB, Zn.T], [Yn, Zn, None]]))

    """
    print(ABC.shape)

    import petsc4py
    petsc4py.init()
    from petsc4py import PETSc
    ABC = sparse.csr_matrix(ABC)

    M = PETSc.Mat()
    M.create(PETSc.COMM_WORLD)
    M.setSizes(ABC.shape)
    M.setUp()

    r0, r1 = M.getOwnershipRange()    
    M.createAIJ(size=ABC.shape, csr=(ABC.indptr[r0:r1+1]-ABC.indptr[r0],
                                     ABC.indices[ABC.indptr[r0]:ABC.indptr[r1]],
                                     ABC.data[ABC.indptr[r0]:ABC.indptr[r1]]))
    M.setUp()
    M.assemble()

    xuvz, buvz = M.getVecs()

    ksp = PETSc.KSP().create()
    ksp.setType('cg')
    ksp.getPC().setType('none')
    ksp.setOperators(M)
    ksp.setFromOptions()

    ksp.max_it = 100
    ksp.rtol = 1e-5
    ksp.atol = 0

    xx, yy = np.meshgrid(np.linspace(0., 1, nx), np.linspace(0., 1., ny))

    u0 = np.sin(2*np.pi*yy)
    v0 = np.sin(2*np.pi*xx)

    aa = 2 * A.T * u0.flatten()
    bb = 2 * B.T * v0.flatten()

    uu = np.zeros(nx*ny)
    vv = np.zeros(nx*ny)
    xa, info = sparse.linalg.cg(AA, aa)
    uu = A.dot(xa)
    xb, info = sparse.linalg.cg(BB, bb)
    vv = B.dot(xb)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(uu.reshape(ny, nx))
    ax2.imshow(vv.reshape(ny, nx))
    plt.show()
    """


    """
    ab = np.hstack((aa, bb))

    x = sparse.linalg.spsolve(AB, ab)
    uvintp = AB.dot(x)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(uvintp[:nx*ny].reshape(ny, nx))
    ax2.imshow(uvintp[nx*ny:].reshape(ny, nx))
    plt.show()
    #b[:2*nx*ny] = np.random.rand(1, 2*nx*ny)
    n = sum(u_is_fluid)
    abc = np.hstack((aa, bb, np.zeros(n)))
    x = np.zeros(3*n)

    nu = sum(u_is_fluid)
    nv = sum(v_is_fluid)

    x = sparse.linalg.spsolve(ABC, abc)

    u0back = np.zeros(nx*ny)
    v0back = np.zeros(nx*ny)
    u0back[:] = A.dot(x[:n])
    v0back[:] = B.dot(x[n:2*n])
    u0back2 = A.dot(xuvz[:n])
    v0back2 = B.dot(xuvz[n:2*n])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(u0back.reshape((ny, nx)))
    ax2.imshow(v0back.reshape((ny, nx)))
    ax3.imshow(u0back2.reshape((ny, nx)))
    ax4.imshow(v0back2.reshape((ny, nx)))
    plt.show()
    """

    n = sum(u_is_fluid)
    u0 = np.zeros((ny, nx))
    v0 = np.zeros((ny, nx))
    abc = np.zeros(3*n)
    x = np.zeros(3*n)

    uo = np.zeros(ny * nx)
    vo = np.zeros(ny * nx)
    for it, timestamp in enumerate(timestamps):
        t = timestamp[0]
        h5filename = timestamp[1]
        mpi_print(t, h5filename)
        with h5py.File(os.path.join(felbm_folder, h5filename), "r") as h5f:
            u0[:, :] = np.array(h5f["u_x"]).reshape((nz, ny, nx))[nz // 2, :, :]
            v0[:, :] = np.array(h5f["u_y"]).reshape((nz, ny, nx))[nz // 2, :, :]
        abc[:n] = 2 * A.T * u0.flatten()
        abc[n:2*n] = 2 * B.T * v0.flatten()
        #x[:], info = sparse.linalg.cg(ABC, abc)
        x[:] = sparse.linalg.spsolve(ABC, abc)
        uo[:] = A.dot(x[:n]) # .reshape((ny, nx))
        vo[:] = B.dot(x[n:2*n]) # .reshape((ny, nx))
        u2 = uo**2 + vo**2

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
        ax1.imshow(uo.reshape((ny, nx)))
        ax2.imshow(vo.reshape((ny, nx)))
        ax3.imshow(np.log(u2).reshape((ny, nx)))

        uf = np.zeros((ny, nx)).flatten()
        vf = np.zeros((ny, nx)).flatten()
        uf[u_is_fluid] = x[:n]
        vf[v_is_fluid] = x[n:2*n]
        ax4.imshow(uf.reshape((ny, nx)))
        ax5.imshow(vf.reshape((ny, nx)))
        ufabs = np.sqrt(uf**2 + vf**2)
        # ax6.imshow(np.log(ufabs).reshape((ny, nx)))
        ufxy = uf.reshape((ny, nx))
        vfxy = vf.reshape((ny, nx))

        ufm = np.zeros_like(ufxy)
        vfm = np.zeros_like(vfxy)
        ufm[:, 1:] = ufxy[:, :-1]
        ufm[:, 0] = ufxy[:, -1]
        vfm[1:, :] = vfxy[:-1, :]
        vfm[0, :] = vfxy[-1, :]
        div = (ufxy-ufm + vfxy-vfm)
        print((div/vfm).flatten()[is_fluid])
        ax6.imshow(div)

        plt.show()