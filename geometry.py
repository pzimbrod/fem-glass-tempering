import gmsh

def create_mesh(path: str, dim: int):
    gmsh.initialize()
    gmsh.model.add(f"Glass {dim}D mesh")

    resolution_fine = 0.1
    resolution_mid = 1.0
    resolution_coarse = 3.0

    if dim == 1:
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
    
    elif dim == 2:
        # Create more points for a refined rectangle
        p1 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0, resolution_fine)
        p2 = gmsh.model.occ.addPoint(50.0, 0.0, 0.0, resolution_mid)
        p3 = gmsh.model.occ.addPoint(50.0, 10.0, 0.0, resolution_fine)
        p4 = gmsh.model.occ.addPoint(0.0, 10.0, 0.0, resolution_mid)

        # Create a more refined rectangle
        rect = gmsh.model.occ.addCurveLoop([
            gmsh.model.occ.addLine(p1, p2),
            gmsh.model.occ.addLine(p2, p3),
            gmsh.model.occ.addLine(p3, p4),
            gmsh.model.occ.addLine(p4, p1),

        ])

        gmsh.model.occ.addPlaneSurface([rect])

        gmsh.model.occ.synchronize()

        gmsh.model.addPhysicalGroup(2, [1], 0)
        gmsh.model.setPhysicalName(1, 0, "cells")

    # Generate mesh
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(dim=dim)
    gmsh.write(path)