import gmsh

maker = {"left": 1, "top": 2, "right": 1, "bottom": 2}
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
        # Add points with finer resolution 
        point_1 = gmsh.model.occ.add_point(0, 0, 0,resolution_mid,0)
        point_2 = gmsh.model.occ.add_point(50, 0, 0,resolution_mid,1)
        point_3 = gmsh.model.occ.add_point(50, 10, 0,resolution_mid,2)
        point_4 = gmsh.model.occ.add_point(0, 10, 0,resolution_mid,3)

        # Add lines between all points creating the rectangle
        left_line = gmsh.model.occ.add_line(point_1, point_2,0)
        top_line = gmsh.model.occ.add_line(point_2, point_3,1)
        right_line = gmsh.model.occ.add_line(point_3, point_4,2)
        bottom_line = gmsh.model.occ.add_line(point_4, point_1,3)

        # Create a line loop and plane surface for meshing
        lines_loop = gmsh.model.occ.add_curve_loop([left_line,top_line,right_line,bottom_line])
        gmsh.model.occ.add_plane_surface([lines_loop])

        gmsh.model.occ.synchronize()

        left_BC, top_BC, right_BC, bottom_BC = 0, 1, 2, 3
        gmsh.model.addPhysicalGroup(2, [left_line], left_BC)
        gmsh.model.setPhysicalName(2, left_BC, "left")
        gmsh.model.addPhysicalGroup(2, [top_line], top_BC)
        gmsh.model.setPhysicalName(2, top_BC, "top")
        gmsh.model.addPhysicalGroup(2, [right_line], right_BC)
        gmsh.model.setPhysicalName(2, right_BC, "right")
        gmsh.model.addPhysicalGroup(2, [bottom_line], bottom_BC)
        gmsh.model.setPhysicalName(2, bottom_BC, "bottom")


    # Generate mesh
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(dim=dim)
    gmsh.write(path)
