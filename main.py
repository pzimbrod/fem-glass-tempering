import gmsh
from ThermalProblem import ThermalProblem
from ViscoElasticProblem import ViscoElasticModel
from math import ceil

# Time domain
t_start = 0.0
t_end = 50.0

#dt = (t_end - t_start) / n_steps
dt = 0.1
n_steps = ceil((t_end - t_start) / dt)
t = t_start

# Triangulation and Finite element
mesh_path = "mesh1d.msh"
def create_geometry(path: str):
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

create_geometry(path=mesh_path)
#mesh, cell_tags, facet_tags = gmshio.read_from_msh("glass_heat/untitled.msh", MPI.COMM_WORLD, 0, gdim=1)
#mesh, cell_tags, facet_tags = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, 0, gdim=1)

poly_degree = 1

model_params = {
    "f": 0.0,
    "epsilon": 0.93,
    "sigma": 5.670e-8,
    "T_0": 600.0,
    "alpha": 0.0005,
    "htc": 280.1,
    "rho": 2500.0,
    "cp": 1433.0,
    "k": 1.0,
}

visco_params = {
    "H": 627.8e3,
    "Tb": 869.0e0,
    "Rg": 8.314,
    "alpha_solid": 9.10e-6,
    "alpha_liquid": 25.10e-6,
    "Tf_init": 873.0,
}

thermal_prob = ThermalProblem(mesh_path=mesh_path,dt=dt,degree=poly_degree)
#mech_prob = ViscoElasticProblem(mesh=thermal_prob.mesh,dt=dt,degree=poly_degree,tensor_degree=poly_degree)
visco_model = ViscoElasticModel(parameters=visco_params,mesh=thermal_prob.mesh,dt=dt,degree=poly_degree,tensor_degree=poly_degree)

initial_temp = 873.0
thermal_prob.set_initial_condition(initial_temp)
thermal_prob.setup_weak_form(parameters=model_params)
thermal_prob.write_initial_output(output_name="diffusion")
thermal_prob.setup_solver()

visco_model.set_initial_condition_Tf(initial_temp)
visco_model.write_initial_output2(output_name="visco")

# Parameters
#alpha = Constant(mesh,ScalarType(0.0))
#alpha = Constant(mesh,ScalarType(0.005))
#alpha = Constant(mesh,ScalarType(1.0))
"""
n = FacetNormal(mesh)
heat_flux = form(dot(-alpha * grad(T_current), n)*ds)
Q = VectorFunctionSpace(mesh, ("DG", 1))
q = Function(Q)
flux_calculator = Expression(-alpha * grad(T_current), Q.element.interpolation_points())
fluxes = []
"""
#log.set_log_level(log.LogLevel.INFO)
postprocess = False

for i in range(n_steps):
    t += dt

    thermal_prob.solve(t=t)
    visco_model.solve_visco(t=t)
    visco_model.compute_Tf_current(T_current=thermal_prob.T_current,Tf_previous=visco_model.Tf_previous,dt=thermal_prob.dt)
    visco_model.compute_stress_tensor_next(T_next=thermal_prob.T_next,T_current=thermal_prob.T_current,T_previous=thermal_prob.T_previous,Tf_current=visco_model.Tf_current,Tf_previous=visco_model.Tf_previous,dt=thermal_prob.dt)
    """
    if postprocess:
        error = form((T_current)**2 * dx)
        E = np.sqrt(mesh.comm.allreduce(assemble_scalar(error), MPI.SUM))
        if mesh.comm.rank == 0:
            print(f"L2-Norm of the solution: {E}")
        
        q.interpolate(flux_calculator)
        #Q = np.sum(q.vector[:])
        Q = mesh.comm.allreduce(q.vector[:],MPI.SUM)
        if mesh.comm.rank == 0:
            print(f"Emitted heat: {Q}")
    """
thermal_prob.finalize()
visco_model.finalize2()

#with dt only, gives me good results, but still decreasing