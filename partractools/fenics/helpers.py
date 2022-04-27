from mpi4py import MPI
import dolfin as df

comm = MPI.COMM_WORLD

class DolfinParams():
    def __init__(self, input_file):
        self.dolfin_params = dict()
        with open(input_file, "r") as infile:
            for el in infile.read().split("\n"):
                if "=" in el:
                    key, val = el.split("=")
                    if val in ["true", "TRUE"]:
                        val = "True"
                    elif val in ["false", "FALSE"]:
                        val = "False"
                    try:
                        self.dolfin_params[key] = eval(val)
                    except:
                        self.dolfin_params[key] = val

    def __getitem__(self, key):
        if key in self.dolfin_params:
            return self.dolfin_params[key]
        else:
            return None

    def __str__(self):
        string = ""
        for key, val in self.dolfin_params.items():
            string += "{}: {}\n".format(key, val)
        return string


class Timestamps():
    def __init__(self, input_file):
        self.ts = []
        self.dirname = os.path.dirname(input_file)
        with open(input_file, "r") as infile:
            for el in infile.read().split("\n"):
                if " " in el:
                    t, filename_loc = el.split(" ")
                    t = float(t)
                    self.ts.append(
                        [t, os.path.join(self.dirname, filename_loc)])

    def __getitem__(self, i):
        return self.ts[i]

    def items(self):
        return self.ts


class PeriodicBC(df.SubDomain):
    def __init__(self, periodic, dim, x_min, x_max):
        self.periodic = periodic
        self.dim = dim
        self.x_min = x_min
        self.x_max = x_max
        super().__init__()

    def inside(self, x, on_boundary):
        return on_boundary and any([
            self.periodic[i] and x[i] < self.x_min[i] + df.DOLFIN_EPS_LARGE
            for i in range(self.dim)
        ]) and not any([
            self.periodic[i] and self.periodic[j] and
            ((x[i] < self.x_min[i] + df.DOLFIN_EPS_LARGE
              and x[j] > self.x_max[j] - df.DOLFIN_EPS_LARGE) or
             (x[i] > self.x_max[i] - df.DOLFIN_EPS_LARGE
              and x[j] < self.x_min[j] + df.DOLFIN_EPS_LARGE))
            for i, j in product(range(dim), range(dim))
        ])

    def map(self, x, y):
        any_periodicity = False
        for i in range(self.dim):
            if self.periodic[i] and x[i] > self.x_max[i] - df.DOLFIN_EPS_LARGE:
                y[i] = x[i] - (self.x_max[i] - self.x_min[i])
                any_periodicity = True
            else:
                y[i] = x[i]
        if not any_periodicity:
            for i in range(self.dim):
                y[i] = -1000.


def mpi_max(x):
    dim = x.shape[1]
    maxVal = np.zeros(dim)
    maxVal[:] = x.max(axis=0)
    comm.Barrier()
    comm.Allreduce(MPI.IN_PLACE, maxVal, op=MPI.MAX)
    comm.Barrier()
    return maxVal


def mpi_min(x):
    dim = x.shape[1]
    minVal = np.zeros(dim)
    minVal[:] = x.min(axis=0)
    comm.Barrier()
    comm.Allreduce(MPI.IN_PLACE, minVal, op=MPI.MIN)
    comm.Barrier()
    return minVal

class Btm(df.SubDomain):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max
        super().__init__()

    def inside(self, x, on_bnd):
        return on_bnd and x[1] < x_min[1] + df.DOLFIN_EPS_LARGE


def parse_element(string):
    if string[:-1] == "P":
        el = "Lagrange"
    return el, int(string[-1])