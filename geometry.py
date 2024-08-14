import typing
from pathlib import Path

import basix
import basix.ufl
from dolfinx import cpp as _cpp
from dolfinx import default_real_type
from dolfinx.cpp.graph import AdjacencyList_int32
from dolfinx.io.utils import distribute_entity_data
from dolfinx.mesh import CellType, Mesh, create_mesh, meshtags, meshtags_from_entities

from dolfinx.io import gmshio
from mpi4py import MPI
import gmsh

# Overwrite the gmshio.read_from_msh() function 
def read_from_msh(
    filename: typing.Union[str, Path],
    comm: MPI.Comm,
    rank: int = 0,
    gdim: int = 3,
    partitioner: typing.Optional[
        typing.Callable[[MPI.Comm, int, int, AdjacencyList_int32], AdjacencyList_int32]
    ] = None,
) -> tuple[Mesh, _cpp.mesh.MeshTags_int32, _cpp.mesh.MeshTags_int32]:
    try:
        import gmsh
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "No module named 'gmsh': dolfinx.io.gmshio.read_from_msh requires Gmsh.", name="gmsh"
        )

    if comm.rank == rank:
        gmsh.initialize(interruptible=False)
        gmsh.model.add("Mesh from file")
        gmsh.merge(str(filename))
        msh = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=gdim, partitioner=partitioner)
        gmsh.finalize()
        return msh
    else:
        return gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=gdim, partitioner=partitioner)


def create_mesh(path: str):
    gmsh.initialize(interruptible=False)
    gmsh.model.add("Glass 1D mesh")

    resolution_fine = 0.1
    resolution_mid = 1.0
    resolution_coarse = 3.0
    gmsh.model.occ.addPoint(0.0,0.0,0.0,resolution_fine,0)
    gmsh.model.occ.addPoint(5.0,0.0,0.0,resolution_mid,1)
    gmsh.model.occ.addPoint(25.0,0.0,0.0,resolution_coarse,2)
    gmsh.model.occ.addPoint(45.0,0.0,0.0,resolution_mid,3)
    gmsh.model.occ.addPoint(50.0,0.0,0.0,resolution_fine,4)

    gmsh.model.occ.addLine(0,1,0)
    gmsh.model.occ.addLine(1,2,1)
    gmsh.model.occ.addLine(2,3,2)
    gmsh.model.occ.addLine(3,4,3)

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(1, [0,1,2,3], 0)
    gmsh.model.setPhysicalName(1, 0, "cells")

    # Generate mesh
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(1)
    gmsh.write(path)