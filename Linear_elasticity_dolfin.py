# Scaled variable
import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
from dolfinx.fem import (FunctionSpace, Function, Constant, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.mesh import locate_entities_boundary


L = 1
W = 0.2
mu = 1
rho = 1
delta = W / L
gamma = 0.4 * delta**2
beta = 1.25
lambda_ = beta
g = gamma


t = 0.0
dt = 0.1
n_steps = 50


# Define mesh
nx, ny = 50, 50
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([50, 10])],
                               [nx, ny], mesh.CellType.triangle)
V = fem.VectorFunctionSpace(domain, ("Lagrange", 1))


def left(x):
    return np.isclose(x[0], 0)
def right(x):
    return np.isclose(x[0], 50)
def top(x):
    return np.isclose(x[1], 10)
def bottom(x):
    return np.isclose(x[1], 0)

fdim = domain.topology.dim - 1

left_bc = locate_entities_boundary(domain, fdim, left)
right_bc = locate_entities_boundary(domain, fdim, right)
top_bc = locate_entities_boundary(domain, fdim, top)
bottom_bc = locate_entities_boundary(domain, fdim, bottom)

bc = [  fem.dirichletbc(ScalarType([0.,0.]),locate_dofs_topological(V, fdim, left_bc), V),
        fem.dirichletbc(ScalarType([0.,0.]),locate_dofs_topological(V, fdim, right_bc), V),
        fem.dirichletbc(ScalarType([0.,0.]),locate_dofs_topological(V, fdim, top_bc), V),
        fem.dirichletbc(ScalarType([0.,0.]),locate_dofs_topological(V, fdim, bottom_bc), V)
        ]

u_old = Function(V)

ds = ufl.Measure("ds", domain=domain)

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = fem.Constant(domain, default_scalar_type((0.5, 1)))
T = fem.Constant(domain, default_scalar_type((0, 0)))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

u = Function(V)

time = np.linspace(0, 1, n_steps+1)

for (i, t) in enumerate(np.diff(time)):

    # Update current time
    t += dt

    # Compute solution
    problem = LinearProblem(a, L, u=u, bcs=bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    problem.solve()

    # Update previous solution
    u_old.x.array[:] = u.x.array[:]
    
print("finish") 
outfile_u = io.XDMFFile(domain.comm,"linear_elasticity.xdmf", "w")
outfile_u.write_mesh(domain)
outfile_u.write_function(u, t)