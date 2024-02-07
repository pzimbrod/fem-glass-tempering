import gmsh

def create_mesh(path: str):
    gmsh.initialize()
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