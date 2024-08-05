import gmsh

def create_mesh(path: str, dim: int):
    gmsh.initialize()
    gmsh.model.add(f"Glass {dim}D mesh")

    resolution_fine = 0.1
    resolution_mid = 1.0
    resolution_coarse = 3.0

    if dim == 1:
        left = gmsh.model.occ.addPoint(0.0,0.0,0.0,resolution_fine,0)
        gmsh.model.occ.addPoint(5.0,0.0,0.0,resolution_mid,1)
        gmsh.model.occ.addPoint(25.0,0.0,0.0,resolution_coarse,2)
        gmsh.model.occ.addPoint(45.0,0.0,0.0,resolution_mid,3)
        right = gmsh.model.occ.addPoint(50.0,0.0,0.0,resolution_fine,4)

        gmsh.model.occ.addLine(0,1,0)
        gmsh.model.occ.addLine(1,2,1)
        gmsh.model.occ.addLine(2,3,2)
        gmsh.model.occ.addLine(3,4,3)

        gmsh.model.occ.synchronize()

        gmsh.model.addPhysicalGroup(0, [left], 10)
        gmsh.model.addPhysicalGroup(0, [right], 12)

        gmsh.model.addPhysicalGroup(1, [0,1,2,3], 0)
        gmsh.model.setPhysicalName(1, 0, "cells")
    
    elif dim == 2:
        # Add points with finer resolution 
        point_1 = gmsh.model.occ.add_point(0, 0, 0,resolution_mid)
        point_2 = gmsh.model.occ.add_point(50.0, 0, 0,resolution_mid)
        point_3 = gmsh.model.occ.add_point(50.0, 10.0, 0,resolution_mid)
        point_4 = gmsh.model.occ.add_point(0, 10.0, 0,resolution_mid)

        # Add lines between all points creating the rectangle
        left_line = gmsh.model.occ.add_line(point_1, point_4)
        top_line = gmsh.model.occ.add_line(point_4, point_3)
        right_line = gmsh.model.occ.add_line(point_3, point_2)
        bottom_line = gmsh.model.occ.add_line(point_2, point_1)

        # Create a line loop and plane surface for meshing
        lines_loop = gmsh.model.occ.add_curve_loop([left_line,top_line,right_line,bottom_line])
        domain = gmsh.model.occ.add_plane_surface([lines_loop])

        gmsh.model.occ.synchronize()
        
        gmsh.model.addPhysicalGroup(2, [domain], 0)
    
        gmsh.model.addPhysicalGroup(1, [left_line], 10)
        gmsh.model.addPhysicalGroup(1, [top_line],  11)
        gmsh.model.addPhysicalGroup(1, [right_line], 12)
        gmsh.model.addPhysicalGroup(1, [bottom_line], 13)        

    elif dim == 3:
        # Create more points for a refined rectangle
        p1 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0, resolution_mid)
        p2 = gmsh.model.occ.addPoint(100.0, 0.0, 0.0, resolution_mid)
        p3 = gmsh.model.occ.addPoint(100.0, 10.0, 0.0, resolution_mid)
        p4 = gmsh.model.occ.addPoint(0.0, 10.0, 0.0, resolution_mid)
        p5 = gmsh.model.occ.addPoint(0.0, 10.0, 50.0, resolution_mid)
        p6 = gmsh.model.occ.addPoint(0.0, 0.0, 50.0, resolution_mid)
        p7 = gmsh.model.occ.addPoint(100.0, 0.0, 50.0, resolution_mid)
        p8 = gmsh.model.occ.addPoint(100.0, 10.0, 50.0, resolution_mid)
        # Create a more refined rectangle
        box = gmsh.model.occ.addCurveLoop([
            gmsh.model.occ.addLine(p1, p2),
            gmsh.model.occ.addLine(p2, p3),
            gmsh.model.occ.addLine(p3, p4),
            gmsh.model.occ.addLine(p4, p1),
            gmsh.model.occ.addLine(p1, p6),
            gmsh.model.occ.addLine(p6, p5),
            gmsh.model.occ.addLine(p5, p4),
            gmsh.model.occ.addLine(p6, p7),
            gmsh.model.occ.addLine(p7, p8),
            gmsh.model.occ.addLine(p5, p8),
            gmsh.model.occ.addLine(p7, p2),
            gmsh.model.occ.addLine(p8, p3),
        ])

        gmsh.model.occ.addPlaneSurface([box])

        gmsh.model.occ.synchronize()

        gmsh.model.addPhysicalGroup(3, [1], 300)
        gmsh.model.setPhysicalName(3, 300, "cells")

    # Generate mesh
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(dim=dim)
    gmsh.write(path)