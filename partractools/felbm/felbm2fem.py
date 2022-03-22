from numpy import unique
from sympy import assemble_partfrac_list, construct_domain
from analyze_eulerian_timeseries import *

import matplotlib.pyplot as plt
import dolfin as df
import itertools
import meshio


def parse_args():
    parser = argparse.ArgumentParser(description="FEM interpolated FELBM data")
    parser.add_argument("folder", type=str, help="Folder")
    parser.add_argument("-t0", type=float, default=None, help="")
    parser.add_argument("-t1", type=float, default=None, help="")
    parser.add_argument("-axis", type=int, default=0, help="Axis")
    parser.add_argument("--show", action="store_true", help="Show plot")
    parser.add_argument("--phase_by_density", action="store_true", help="Show plot")
    args = parser.parse_args()
    return args


class PBC(df.SubDomain):
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly
        df.SubDomain.__init__(self)

    def inside(self, xin, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        x = xin + 0.5
        return bool(
            (df.near(x[0], 0.) or df.near(x[1], 0.))
            and (not (df.near(x[0], self.Lx) or df.near(x[1], self.Ly)))
            and on_boundary)

    def map(self, x, y):
        if df.near(x[0], self.Lx-0.5) and df.near(x[1], self.Ly-0.5):
            y[0] = x[0] - self.Lx
            y[1] = x[1] - self.Ly
        elif df.near(x[0], self.Lx-0.5):
            y[0] = x[0] - self.Lx
            y[1] = x[1]
        else:  # near(x[2], Lz/2.):
            y[0] = x[0]
            y[1] = x[1] - self.Ly

class Wall(df.SubDomain):
    def __init__(self, is_fluid_xy):
        self.is_fluid_xy = is_fluid_xy
        self.ny, self.nx = is_fluid_xy.shape
        print(self.ny, self.nx)
        super().__init__()

    def inside(self, xin, on_boundary):
        x = xin + 0.5
        if bool(x[0] > df.DOLFIN_EPS_LARGE and x[0] < self.nx - df.DOLFIN_EPS_LARGE and
                x[1] > df.DOLFIN_EPS_LARGE and x[1] < self.ny - df.DOLFIN_EPS_LARGE):
            return on_boundary
        elif on_boundary:
            return x[0].is_integer() and x[1].is_integer()
        
def make_xdmf_mesh(nx, ny, is_fluid, tmpfilename, cell_type="quad", nsub=1):
    if rank == 0:
        nodes = np.array([(i-0.5, j-0.5) for j, i in itertools.product(range(nsub*ny+1), range(nx+1))], dtype=float)
        elems = np.array([(i + j*(nx+1), i+1 + j*(nx+1), i+1 + (j+1)*(nx+1), i + (j+1)*(nx+1)) 
                          for j, i in itertools.product(range(ny), range(nx))], dtype=int)
        
        elems = elems[is_fluid, :]
        used_nodes = np.unique(elems)
        map_ids = np.zeros(used_nodes.max()+1, dtype=int)
        for i, j in zip(used_nodes, range(len(used_nodes))):
            map_ids[i] = j
        nodes = nodes[used_nodes, :]
        elems = map_ids[elems]

        if cell_type == "triangle":
            elems_split = []
            for elem in elems:
                elems_split.append(elem[[0, 2, 1]])
                elems_split.append(elem[[0, 2, 3]])
            elems = np.array(elems_split)

        m = meshio.Mesh(nodes, [(cell_type, elems)])
        m.write(tmpfilename)

    comm.Barrier()


def load_mesh(nx, ny, is_fluid, analysisfolder, cell_type="quad", nsub=1):
    tmpfilename = os.path.join(analysisfolder, "foo.xdmf")
    make_xdmf_mesh(nx, ny, is_fluid, tmpfilename, cell_type, nsub)

    mesh = df.Mesh()
    with df.XDMFFile(tmpfilename) as xdmff:
        xdmff.read(mesh)

    return mesh


def refine_near_obstacles(mesh, is_fluid_xy, dd):
    cell_marker = df.MeshFunction("bool", mesh, mesh.topology().dim())
    cell_marker.set_all(False)
    for cell in df.cells(mesh):
        px = cell.midpoint().array()
        ix, iy = int(np.floor(px[0]+0.5)), int(np.floor(px[1]+0.5))
        is_close_to_solid = False
        for di in range(-dd, dd+1):
            for dj in range(-dd, 2):
                if di**2 + dj**2 <= dd**2:
                    ixpd = (ix + di) % nx
                    iypd = (iy + dj) % ny
                    if not is_fluid_xy[iypd, ixpd]:
                        is_close_to_solid = True
                        break
        cell_marker[cell] = is_close_to_solid
    return df.refine(mesh, cell_marker)


if __name__ == "__main__":
    args = parse_args()

    felbm_folder = args.folder
    timestamps = select_timestamps(felbm_folder, args.t0, args.t1)

    analysisfolder = os.path.join(felbm_folder, "Analysis")
    make_folder_safe(analysisfolder)

    is_fluid_xy, (nx, ny, nz) = get_fluid_domain(felbm_folder)
    is_fluid = is_fluid_xy.flatten()

    #print("nx = {}, ny = {}".format(nx, ny))

    rho = np.zeros_like(is_fluid_xy, dtype=float)
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    #p = np.zeros_like(rho)
    ux_ = ux.flatten()[is_fluid]
    uy_ = np.zeros_like(ux_)

    mesh_q = load_mesh(nx, ny, is_fluid, analysisfolder)
    mesh = load_mesh(nx, ny, is_fluid, analysisfolder, "triangle")
    #mesh = df.refine(df.refine(mesh))
    
    print(is_fluid_xy.shape)

    mesh_fine = df.refine(df.refine(mesh))
    #mesh_fine = refine_near_obstacles(mesh, is_fluid_xy, 2)
    #mesh_fine = refine_near_obstacles(mesh_fine, is_fluid_xy, 1)

    pbc = PBC(nx, ny)

    subd = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    subd_fine = df.MeshFunction("size_t", mesh_fine, mesh_fine.topology().dim()-1, 0)

    wall = Wall(is_fluid_xy)
    wall.mark(subd, 1)
    wall.mark(subd_fine, 1)
    pbc.mark(subd, 2)
    pbc.mark(subd_fine, 2)
    with df.XDMFFile(mesh.mpi_comm(), os.path.join(analysisfolder, "subd.xdmf")) as xdmff:
        xdmff.write(subd)
        #xdmff.write(subd_fine)

    S0q = df.FunctionSpace(mesh_q, "DG", 0)
    S0 = df.FunctionSpace(mesh, "DG", 0, constrained_domain=pbc)

    #dof_coord_loc = np.array(S0.tabulate_dof_coordinates()-0.0, dtype=int)
    #print(dof_coord_loc)
   
    dof_coord_glob = np.array([(i, j) for j, i in itertools.product(range(ny), range(nx))], dtype=int) #[is_fluid, :]
    #print(dof_coord_glob)
    coord2glob = dict()
    for i, coord in enumerate(dof_coord_glob):
        coord2glob[tuple(coord)] = i
    loc2glob = dict()
    dofmap = S0.dofmap()
    for cell in df.cells(mesh):
        dof_index = dofmap.cell_dofs(cell.index())[0]
        px = cell.midpoint().array()
        ix, iy = np.floor(px[0]+0.5), np.floor(px[1]+0.5)
        loc2glob[dof_index] = coord2glob[(ix, iy)]
    
    keys = np.array(list(loc2glob.keys()))
    assert(keys.max()+1 == len(keys) )
    loc2glob = np.array([loc2glob[i] for i in range(keys.max()+1)])

    #loc2glob = np.array([coord2glob[tuple(coord)] for coord in dof_coord_loc], dtype=int)


    ux0 = df.Function(S0, name="ux0")
    uy0 = df.Function(S0, name="uy0")
    u0 = df.as_vector((ux0, uy0))

    #u_el = df.FiniteElement("RT", mesh.ufl_cell(), 1)
    u_el = df.VectorElement("CG", mesh.ufl_cell(), 1)
    V = df.FunctionSpace(mesh, u_el, constrained_domain=pbc)
    #V = df.FunctionSpace(mesh, "RT", 1)
    s_el = df.FiniteElement("CG", mesh.ufl_cell(), 1)
    S = df.FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    u_ = df.Function(V, name="u")
    phi_ = df.Function(S, name="phi")
    phi = df.TrialFunction(S)
    q = df.TestFunction(S)
    noslip = df.DirichletBC(V, df.Constant((0., 0.)), subd, 1)

    # Scott-Vogelius interpolation?

    u_1 = df.Function(V)

    #+ df.inner(df.grad(u), df.grad(v)) * df.dx \

    r = 1.0
    a = df.dot(u, v) * df.dx + r * df.div(u) * df.div(v) * df.dx
    L = df.dot(u0, v) * df.dx #+ df.div(u_1) * df.div(v) * df.dx

    #a_u, L_u = df.lhs(F_u), df.rhs(F_u)
    problem_u = df.LinearVariationalProblem(a, L, u_, bcs=noslip)
    solver_u = df.LinearVariationalSolver(problem_u)
    solver_u.parameters["linear_solver"] = "gmres"
    solver_u.parameters["preconditioner"] = "hypre_amg"
    solver_u.parameters["krylov_solver"]["relative_tolerance"] = 1e-12
    solver_u.parameters["krylov_solver"]["monitor_convergence"] = False
    # + df.inner(df.grad(uw-u0_), df.grad(vw)) * df.dx \

    if True:
        u_el = df.VectorElement("CG", mesh_fine.ufl_cell(), 1)
        p_el = df.FiniteElement("DG", mesh_fine.ufl_cell(), 0)
        W = df.FunctionSpace(mesh_fine, df.MixedElement([u_el, p_el]), constrained_domain=pbc)
        w_ = df.Function(W)
        u0_ = df.Function(W.sub(0).collapse())
        uw, pw = df.TrialFunctions(W)
        vw, qw = df.TestFunctions(W)
        Fw = df.dot(uw - u0_, vw) * df.dx \
            - df.div(vw) * pw * df.dx \
            - df.div(uw) * qw * df.dx
        noslipw = df.DirichletBC(W.sub(0), df.Constant((0., 0.)), subd_fine, 1)

        a = df.lhs(Fw)  # df.dot(df.grad(phi), df.grad(q)) * df.dx
        L = df.rhs(Fw)  # q * df.div(u_) * df.dx
        problem = df.LinearVariationalProblem(a, L, w_, bcs=noslipw)
        solver = df.LinearVariationalSolver(problem)
        solver.parameters["linear_solver"] = "gmres"
        solver.parameters["preconditioner"] = "hypre_amg"
        solver.parameters["krylov_solver"]["absolute_tolerance"] = 1e-12
        solver.parameters["krylov_solver"]["relative_tolerance"] = 1e-12
        solver.parameters["krylov_solver"]["monitor_convergence"] = True

    #A = df.assemble(a)
    #solver = df.KrylovSolver(A, "gmres")
    # Create vector that spans the null space
    #null_vec = df.Vector(w_.vector())
    #S.dofmap().set(null_vec, 1.0)
    #null_vec *= 1.0/null_vec.norm("l2")

    # Create null space basis object and attach to Krylov solver
    #null_space = df.VectorSpaceBasis([null_vec])
    #df.as_backend_type(A).set_nullspace(null_space)

    #isFluid.vector()[:] = ux  # is_fluid_xy.flatten()
    #with df.XDMFFile(mesh.mpi_comm(), os.path.join(analysisfolder, "isFluid.xdmf")) as xdmff:
    #    xdmff.write(isFluid)
    xdmff = df.XDMFFile(mesh.mpi_comm(), os.path.join(analysisfolder, "data.xdmf"))
    xdmff.parameters["rewrite_function_mesh"] = False
    xdmff.parameters["flush_output"] = True
    xdmff.parameters["functions_share_mesh"] = True

    xdmff_fine = df.XDMFFile(mesh.mpi_comm(), os.path.join(analysisfolder, "data_fine.xdmf"))
    xdmff_fine.parameters["rewrite_function_mesh"] = False
    xdmff_fine.parameters["flush_output"] = True
    xdmff_fine.parameters["functions_share_mesh"] = True

    mpi_print("Test")

    t_ = np.array([t for t, _ in timestamps])

    #istart = (rank * len(timestamps)) // size
    #istop = ((rank + 1) * len(timestamps)) // size
    #timestamps_block = list(enumerate(timestamps))[istart:istop]
    timestamps_block = list(enumerate(timestamps))

    comm.Barrier()
    if rank == 0:
        pbar = tqdm(total=len(timestamps_block))

    Sc = df.FunctionSpace(mesh_fine, s_el)
    psi_c = df.Function(Sc, name="psi")
    X = df.interpolate(df.Expression("x[0]", degree=1), Sc)
    Y = df.interpolate(df.Expression("x[1]", degree=1), Sc)

    # psi stuff
    psi = df.TrialFunction(Sc)
    q = df.TestFunction(Sc)
    n = df.FacetNormal(mesh)

    vort = df.curl(u_)

    a = df.dot(df.grad(q), df.grad(psi)) * df.dx
    L = q * vort * df.dx + q * (n[1] * u_[0] - n[0] * u_[1]) * df.ds
    psi_ = df.Function(Sc, name="psi")

    A = df.assemble(a)
    b = df.assemble(L)

    krylov_prm = dict(
            monitor_convergence=False,
            report=False,
            error_on_nonconvergence=False,
            nonzero_initial_guess=True,
            maximum_iterations=200,
            relative_tolerance=1e-8,
            absolute_tolerance=1e-8),

    null_vec = df.Vector(psi_.vector())
    Sc.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0 / null_vec.norm('l2')
    Aa = df.as_backend_type(A)
    null_space = df.VectorSpaceBasis([null_vec])
    Aa.set_nullspace(null_space)
    Aa.null_space = null_space

    psi_prec = df.PETScPreconditioner("hypre_amg")
    psi_sol = df.PETScKrylovSolver("gmres", psi_prec)
    #print(phi_sol.parameters)
    #psi_sol.parameters.update(krylov_prm)
    psi_sol.set_reuse_preconditioner(True)

    UxUy_ = []

    for it, timestamp in timestamps_block:
        t = timestamp[0]
        h5filename = timestamp[1]
        mpi_print(t, h5filename)
        with h5py.File(os.path.join(felbm_folder, h5filename), "r") as h5f:
            #p[:, :] = np.array(h5f["pressure"]).reshape((nz, ny, nx))[nz // 2, :, :]
            #rho[:, :] = np.array(h5f["density"]).reshape((nz, ny, nx))[nz // 2, :, :]
            ux[:, :] = np.array(h5f["u_x"]).reshape((nz, ny, nx))[nz // 2, :, :]
            uy[:, :] = np.array(h5f["u_y"]).reshape((nz, ny, nx))[nz // 2, :, :]
        ux0.vector().set_local(ux.flatten()[loc2glob])
        uy0.vector().set_local(uy.flatten()[loc2glob])

        xdmff.write(ux0, t)
        xdmff.write(uy0, t)

        # TODO: only works when there is a mass density difference between phases!!
        #rho_mean = rho[is_fluid_xy].mean()
        #phase1 = np.logical_and(rho < rho_mean, is_fluid_xy)
        #phase2 = np.logical_and(np.logical_not(phase1), is_fluid_xy)
        # u.assign(df.project(u0, V=V, bcs=noslip, solver_type="gmres", preconditioner_type="amg"))

        #"""
        max_iter = 1
        iter = 0
        div_u_norm = 1.
        u_1.vector()[:] = 0.
        while iter < max_iter and div_u_norm > 1e-10:
            # solve and update w
            solver_u.solve()
            u_1.vector().axpy(-r, u_.vector())

            div_u_norm = np.sqrt(df.assemble(df.div(u_) * df.div(u_) * df.dx))
            print("iter={}: norm(div u) = {}".format(iter, div_u_norm))
            iter += 1
        #"""

        df.LagrangeInterpolator.interpolate(u0_, u_)
        solver.solve()
        uw_, pw_ = w_.split(deepcopy=True)
        uw_.rename("u", "tmp")
        div_u_norm = np.sqrt(df.assemble(df.div(uw_) * df.div(uw_) * df.dx))
        print("norm(div u) = {}".format(div_u_norm))
        #u_.assign(uw_)

        """
        """
        Ux = df.assemble(u_[0] * df.dx)/(nx*ny)
        Uy = df.assemble(u_[1] * df.dx)/(nx*ny)
        UxUy_.append([Ux, Uy])

        print("Ux={}, Uy={}".format(Ux, Uy))

        #[bc.apply(A, b) for bc in bcs]
        #df.solve(A, psi_.vector(), b, "gmres", "hypre_amg")
        #df.normalize(psi_.vector())
        
        b[:] = df.assemble(L)
        null_space.orthogonalize(b)
        psi_sol.solve(A, psi_.vector(), b)
        df.normalize(psi_.vector())

        #df.LagrangeInterpolator.interpolate(psi_c, psi_)
        #psi_c.vector()[:] += - Uy * X.vector()[:] + Ux * Y.vector()[:]

        #b = df.assemble(L)
        #null_space.orthogonalize(b)
        #solver.solve(w_.vector(), b)
        #solver.solve()
        #U_, Phi_ = w_.split(deepcopy=True)
        #U_.rename("U", "U")

        #df.solve(a == L, phi_, solver_parameters={"linear_solver": "gmres", "preconditioner": "hypre_amg"})
        #solver.solve(phi_.vector(), b)

        #xdmff.write(ux0, t)
        #xdmff.write(uy0, t)
        xdmff.write(u_, t)
        xdmff_fine.write(uw_, t)
        #xdmff.write(U_, t)
        xdmff.write(psi_, t)

        if rank == 0:
            pbar.update(1)
        
    #comm.Reduce(uxt_loc, uxt, op=MPI.SUM, root=0)
    #comm.Reduce(uyt_loc, uyt, op=MPI.SUM, root=0)
    #comm.Reduce(S1t_loc, S1t, op=MPI.SUM, root=0)
    #comm.Reduce(S2t_loc, S2t, op=MPI.SUM, root=0)

    if rank == 0:
        #tdata = np.vstack((t_, uxt, uyt, S1t, S2t))
        #np.savetxt(os.path.join(analysisfolder, "tdata.dat"), tdata)
        np.savetxt(os.path.join(analysisfolder, "UxUy.dat"), np.array(UxUy_))
        pass