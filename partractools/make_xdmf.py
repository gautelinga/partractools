#!/usr/bin/env python3
import argparse
import os
import numpy as np
import h5py
from partractools.common.utils import Params, get_folders


header = """<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="{name}" GridType="Collection" CollectionType="Temporal">"""

grid_begin = """
      <Grid Name="mesh" GridType="Uniform">"""

mesh_face = """
        <Topology NumberOfElements="{num_faces}" TopologyType="Triangle" NodesPerElement="3">
          <DataItem Dimensions="{num_faces} 3" NumberType="UInt" Format="HDF">{filename}:{faces_loc}</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Dimensions="{num_nodes} 3" Format="HDF">{filename}:{nodes_loc}</DataItem>
        </Geometry>"""

mesh_face_quad = """
        <Topology NumberOfElements="{num_faces}" TopologyType="Quadrilateral" NodesPerElement="4">
          <DataItem Dimensions="{num_faces} 4" NumberType="UInt" Format="HDF">{filename}:{faces_loc}</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Dimensions="{num_nodes} 3" Format="HDF">{filename}:{nodes_loc}</DataItem>
        </Geometry>"""

mesh_edge = """
        <Topology NumberOfElements="{num_edges}" TopologyType="PolyLine" NodesPerElement="2">
          <DataItem Dimensions="{num_edges} 2" NumberType="UInt" Format="HDF">{filename}:{edges_loc}</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Dimensions="{num_nodes} 3" Format="HDF">{filename}:{nodes_loc}</DataItem>
        </Geometry>"""
mesh_vert = """
        <Topology TopologyType="Polyvertex" NodesPerElement="{num_nodes}">
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Dimensions="{num_nodes} 3" Format="HDF">{filename}:{nodes_loc}</DataItem>
        </Geometry>"""

timestamp = """
        <Time Value="{time}" />"""

attrib_scalar_node = """
        <Attribute Name="{field}" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="{num_nodes} 1" Format="HDF">{filename}:{field_loc}</DataItem>
        </Attribute>"""

attrib_vector_node = """
      <Attribute Name="{field}" AttributeType="Vector" Center="Node">
        <DataItem Dimensions="{num_nodes} 3" Format="HDF">{filename}:{field_loc}</DataItem>
      </Attribute>"""

attrib_tensor_node = """
      <Attribute Name="{field}" AttributeType="Tensor" Center="Node">
        <DataItem Dimensions="{num_nodes} 9" Format="HDF">{filename}:{field_loc}</DataItem>
      </Attribute>"""

attrib_scalar_face = """
        <Attribute Name="{field}" AttributeType="Scalar" Center="Cell">
          <DataItem Dimensions="{num_faces} 1" Format="HDF">{filename}:{field_loc}</DataItem>
        </Attribute>"""

attrib_scalar_edge = """
        <Attribute Name="{field}" AttributeType="Scalar" Center="Cell">
          <DataItem Dimensions="{num_edges} 1" Format="HDF">{filename}:{field_loc}</DataItem>
        </Attribute>"""

grid_end = """
      </Grid>"""

footer = """
    </Grid>
  </Domain>
</Xdmf>"""

possible_fields = [["u", "Vector", "Node"],
                  ["c", "Scalar", "Node"],
                  ["p", "Scalar", "Node"],
                  ["rho", "Scalar", "Node"],
                  ["H", "Scalar", "Node"],
                  ["w", "Scalar", "Node"],
                  ["S", "Scalar", "Node"],
                  ["t_loc", "Scalar", "Node"],
                  ["n", "Vector", "Node"],
                  ["dA", "Scalar", "Face"],
                  ["dA0", "Scalar", "Face"],
                  ["tau", "Scalar", "Face"],
                  ["dl", "Scalar", "Edge"],
                  ["dl0", "Scalar", "Edge"],
                  ["elong", "Scalar", "Edge"],
                  ["logelong", "Scalar", "Edge"],
                  ["doublings", "Scalar", "Edge"],
                  ["tau", "Scalar", "Edge"],
                  ["cell_type", "Scalar", "Node"]
                  ]

def main():
    parser = argparse.ArgumentParser(description="Make xdmf from sheet or strip")
    parser.add_argument("folder", type=str, help="Folder")
    parser.add_argument("-t_min", type=float, default=0.0, help="t_min")
    parser.add_argument("-t_max", type=float, default=np.inf, help="t_max")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    folders = get_folders(args.folder)
    nfld = len(folders)
    if nfld == 0:
      folders = [args.folder]
    for ifld, folder in enumerate(folders):
      files = os.listdir(folder)

      params = Params(folder)
      t0 = params.get_tmin()

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
          if t >= args.t_min and t <= args.t_max:
              ts.append(t)

      if args.verbose:
        print("Found data: {} with {} timestamps from t={} to t={}".format(folder, len(ts), ts[0], ts[-1]))

      text = header.format(name="Timeseries")
      for it, t in enumerate(ts):
          posft, grp = posf[t]
          fields = []
          with h5py.File(posft, "r") as h5f:
              nodes = np.array(h5f[grp + "/points"])
              has_faces = grp + "/faces" in h5f
              edges = []
              faces = []
              if has_faces:
                  faces = np.array(h5f[grp + "/faces"])
              has_edges = grp + "/edges" in h5f
              if has_edges:
                  edges = np.array(h5f[grp + "/edges"])
              for field in possible_fields:
                  if field[0] in h5f[grp]:
                      fields.append(field)

          posftrel = posft[len(args.folder):]

          text += grid_begin
          if has_faces:
              text += mesh_face.format(num_faces=len(faces), num_nodes=len(nodes),
                                      filename=posftrel, faces_loc=grp+"/faces",
                                      nodes_loc=grp+"/points")
          elif has_edges:
              text += mesh_edge.format(num_edges=len(edges), num_nodes=len(nodes),
                                      filename=posftrel, edges_loc=grp+"/edges",
                                      nodes_loc=grp+"/points")
          else:
              text += mesh_vert.format(num_nodes=len(nodes), filename=posftrel,
                                      nodes_loc=grp+"/points")
          text += timestamp.format(time=t)
          for field, vtype, vloc in fields:
              attrib = ""
              if vtype == "Vector" and vloc == "Node":
                  attrib += attrib_vector_node
              elif vtype == "Scalar" and vloc == "Node":
                  attrib += attrib_scalar_node
              elif vtype == "Tensor" and vloc == "Node":
                  attrib += attrib_tensor_node
              elif vtype == "Scalar" and vloc == "Face" and has_faces:
                  attrib += attrib_scalar_face
              elif vtype == "Scalar" and vloc == "Edge" and has_edges:
                  attrib += attrib_scalar_edge

              if attrib != "":
                  text += attrib.format(num_nodes=len(nodes),
                                        num_faces=len(faces),
                                        num_edges=len(edges),
                                        filename=posftrel,
                                        field_loc=grp+"/"+field,
                                        field=field)
          text += grid_end
      text += footer

      if nfld > 0:
        xdmffilename = "mesh_{}.xdmf".format(ifld)
      else:
        xdmffilename = "mesh.xdmf"
      with open(os.path.join(args.folder, xdmffilename), "w") as ofile:
          ofile.write(text)

if __name__ == "__main__":
    main()
