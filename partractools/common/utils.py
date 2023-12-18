import numpy as np
import os
import h5py
from xml.etree import cElementTree as ET
import mpi4py.MPI as MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_root = mpi_rank == 0

def mpi_print(*args):
    if mpi_rank == 0:
        print(*args)

def mpi_max(x):
    x_max_loc = np.array(x).max(axis=0)
    x_max = np.zeros_like(x_max_loc)
    mpi_comm.Allreduce(x_max_loc, x_max, op=MPI.MAX)
    return x_max

def mpi_min(x):
    x_min_loc = np.array(x).min(axis=0)
    x_min = np.zeros_like(x_min_loc)
    mpi_comm.Allreduce(x_min_loc, x_min, op=MPI.MIN)
    return x_min

def mpi_sum(data):
    data = mpi_comm.gather(data, root=0)
    if mpi_root:
        data = sum(data)
    else:
        data = 0    
    return data

class GenParams():
    def __init__(self, input_file=None, required=False):
        self.prm = dict()
        if input_file is not None:
            self.load(input_file, required=required)

    def load(self, input_file, required=False):
        if not os.path.exists(input_file) and required:
            mpi_print(f"No such parameters file: {input_file}")
            exit()
        if os.path.exists(input_file):
            with open(input_file, "r") as infile:
                for el in infile.read().split("\n"):
                    if "=" in el:
                        key, val = el.split("=")
                        if val in ["true", "TRUE"]:
                            val = "True"
                        elif val in ["false", "FALSE"]:
                            val = "False"
                        try:
                            self.prm[key] = eval(val)
                        except:
                            self.prm[key] = val

    def dump(self, output_file):
        with open(output_file, "w") as ofile:      
            ofile.write("\n".join([f"{key}={val}" for key, val in self.prm.items()]))

    def __getitem__(self, key):
        if key in self.prm:
            return self.prm[key]
        else:
            mpi_print("No such parameter: {}".format(key))
            exit()
            #return None

    def __setitem__(self, key, val):
        self.prm[key] = val

    def __str__(self):
        string = "\n".join(["{}: {}".format(key, val) for key, val in self.prm.items()])
        return string
    
    def __contains__(self, key):
        return key in self.prm

def parse_xdmf(xml_file, get_mesh_address=False):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    basedir = os.path.dirname(xml_file)

    dsets = []
    timestamps = []

    geometry_found = not get_mesh_address
    topology_found = not get_mesh_address

    for i, step in enumerate(root[0][0]):
        if step.tag == "Time":
            # Support for earlier dolfin formats
            timestamps = [float(time) for time in
                          step[0].text.strip().split(" ")]
        elif step.tag == "Grid":
            timestamp = None
            dset_address = None
            for prop in step:
                if prop.tag == "Time":
                    timestamp = float(prop.attrib["Value"])
                elif prop.tag == "Attribute":
                    dset_address = prop[0].text.split(":") # [1]
                    dset_address[0] = os.path.join(basedir, dset_address[0])
                elif not topology_found and prop.tag == "Topology":
                    topology_address = prop[0].text.split(":")
                    topology_address[0] = os.path.join(basedir, topology_address[0])
                    topology_found = True
                elif not geometry_found and prop.tag == "Geometry":
                    geometry_address = prop[0].text.split(":")
                    geometry_address[0] = os.path.join(basedir, geometry_address[0])
                    geometry_found = True
            if timestamp is None:
                timestamp = timestamps[i-1]
            dsets.append((timestamp, dset_address))
    if get_mesh_address and topology_found and geometry_found:
        return (dsets, topology_address, geometry_address)
    return dsets

class Params:
    def __init__(self, folder):
        self.params_dict = read_params(folder)
        self.ts = np.array(sorted(self.params_dict.keys()))

    def get(self, key, t):
        i = np.where(self.ts <= t)[0][-1]
        tkey = self.ts[i]
        return self.params_dict[tkey][key]

    def get_tmin(self):
        return self.ts[-1]
    
    def exists(self):
        return len(self.params_dict)
    
    def __str__(self) -> str:
        string = ""
        for key, val in self.params_dict.items():
            string += f"{key}: {val}\n"
        return string
    
    def __getitem__(self, key):
        return self.get(key, self.ts[0])

def find_params(folder):
    paramsfiles = dict()
    for filename in os.listdir(folder):
        if "params_from_t" in filename:
            t = float(filename[13:-4])
            paramsfiles[t] = os.path.join(folder, filename)
    return paramsfiles


def parse_params(paramsfiles):
    params = dict()
    for t, paramsfile in paramsfiles.items():
        params[t] = parse_paramsfile(paramsfile)
    return params


def parse_paramsfile(paramsfile):
    params = dict()
    with open(paramsfile) as pf:
        line = pf.readline()
        cnt = 1
        while line:
            item = line.strip().split("=")
            key = item[0]
            val = item[1]
            params[key] = val
            line = pf.readline()
            cnt += 1
    return params


def read_params(folder):
    paramsfiles = find_params(folder)
    return parse_params(paramsfiles)


def read_timestamps(infile):
    timestamps = []
    with open(infile, "r") as tf:
        for line in tf:
            line = line.strip()
            tstr, fname = line.split("\t")
            timestamps.append((float(tstr), fname))
    return timestamps


def get_timeseries(folder, t_min=-np.inf, t_max=np.inf):
    files = os.listdir(folder)

    posf = dict()
    for file in files:
        if file[:11] == "data_from_t" and file[-3:] == ".h5":
            t = float(file[11:-3])
            posft = os.path.join(folder, file)
            try:
                with h5py.File(posft, "r") as h5f:
                    for grp in h5f:
                        posf[float(grp)] = (posft, grp)
            except:
                pass

    ts = []
    for t in list(sorted(posf.keys())):
        if t >= t_min and t <= t_max:
            ts.append(t)

    return ts, posf

def get_folders(folder):
    folders = []
    paramsfiles = find_params(folder)

    if len(paramsfiles) == 0:
        subfolders = []
        for a in os.listdir(folder):
            if a.isdigit():
                subfolders.append(a)
        subfolders = sorted(subfolders)
        for subfolder in subfolders:
            fullpath = os.path.join(folder, subfolder)
            paramsfiles = find_params(fullpath)
            folders.append(fullpath)
    return folders

def get_first_one(d):
    for key, val in d.items():
        if len(val) == 1:
            return key
        #print(len(val))
    return key

def get_h5data_location(folder):
    files = os.listdir(folder)

    posf = dict()
    for file in files:
        if file[:11] == "data_from_t" and file[-3:] == ".h5":
            t = float(file[11:-3])
            posft = os.path.join(folder, file)
            try:
                with h5py.File(posft, "r") as h5f:
                    for cat in h5f:
                        posf[float(cat)] = (posft, cat)
            except:
                pass
    return posf

def compute_lines(pts, edges):
    """ edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5)] """
    v2e = dict()
    for iedge, edge in enumerate(edges):
        for iv in edge:
            if iv in v2e:
                v2e[iv].add(iedge)
            else:
                v2e[iv] = set([iedge])

    lines = []
    while len(v2e):
        line = []
        iv0 = get_first_one(v2e)
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
            line.append((iv_prev, iv, ie))
            if iv == iv0:
                closed = True
                break
            next_edges = set(v2e.pop(iv))
            next_edges.remove(ie)
        lines.append((line, closed))
    return lines