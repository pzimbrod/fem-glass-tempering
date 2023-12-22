from dolfinx.mesh import create_interval, locate_entities_boundary
from mpi4py import MPI

from dolfinx.mesh import locate_entities_boundary
from dolfinx import fem, io, plot, nls, log
from dolfinx.io import gmshio
from dolfinx.fem import (FunctionSpace, Function, Constant, dirichletbc,
                         locate_dofs_geometrical, form, locate_dofs_topological,
                         assemble_scalar, VectorFunctionSpace, Expression)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, NonlinearProblem
from ufl import (TrialFunction, TestFunction, FiniteElement, grad, dot, inner,
                 lhs, rhs, Measure, SpatialCoordinate, FacetNormal, TensorElement, Identity)  # , ds, dx
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
import ufl
from math import factorial


class ViscoElasticProblem:
    def __init__(self, mesh, dt, degree=1, tensor_degree=1) -> None:
        self.mesh = mesh
        self.fe = FiniteElement("P", self.mesh.ufl_cell(), degree)
        self.tfe = TensorElement("P", self.mesh.ufl_cell(), tensor_degree)
        self.fs = FunctionSpace(mesh=self.mesh, element=self.fe)
        self.tfs = FunctionSpace(mesh=self.mesh, element=self.tfe)
        self.dt = dt

        # For nonlinear problems, there is no TrialFunction
        self.T_current = Function(self.fs)
        self.T_current.name = "Temperature"
        self.v = TestFunction(self.fs)
        self.T_previous = Function(self.fs)      # previous time step

    def set_initial_condition(self, init_value: float) -> None:
        x = SpatialCoordinate(self.mesh)

        def temp_init(x):
            values = np.full(x.shape[1], init_value, dtype=ScalarType)
            return values
        self.T_previous.interpolate(temp_init)
        self.T_current.interpolate(temp_init)

    def set_dirichlet_bc(self, bc_value: float) -> None:
        return

    def write_initial_output(self, output_name: str, t: float = 0.0) -> None:
        self.xdmf = io.XDMFFile(self.mesh.comm, f"{output_name}.xdmf", "w")
        self.xdmf.write_mesh(self.mesh)
        self.xdmf.write_function(self.T_current, t)

    def setup_weak_form(self, parameters: dict) -> None:
        return

    def setup_solver(self) -> None:
        return

    def solve(self, t) -> None:
        return

    def finalize(self) -> None:
        self.xdmf.close()


class ViscoElasticModel:
    def __init__(self, prob: ViscoElasticProblem, parameters: dict) -> None:
        self.mesh = prob.mesh
        dim = self.mesh.topology.dim
        # Identity tensor
        self.I = ufl.Identity(dim)
        # Intial fictive temperture [K]
        self.Tf_init = Constant(self.mesh, parameters["Tf_init"])
        # Activation energy [J/mol]
        self.H = Constant(self.mesh, parameters["H"])
        # Universal gas constant [J/(mol K)]
        self.Rg = Constant(self.mesh, parameters["Rg"])
        # Base temperature [K]
        self.Tb = Constant(self.mesh, parameters["Tb"])
        # Solid thermal expansion coefficient [K^-1]
        self.alpha_solid = Constant(self.mesh, parameters["alpha_solid"])
        # Liquid thermal expansion coefficient [K^-1]
        self.alpha_liquid = Constant(self.mesh, parameters["alpha_liquid"])
        # weighting coefficient for temperature and structural energies, c.f. Nielsen et al. eq. 8
        self.chi = 0.5
        self.dt = prob.dt
        self.fss= prob.fs
        self.tfss= prob.tfs
        self.Tf_current = Function(self.fss)
        self.Tf_previous = Function(self.fss)
        self.Tf_fss_current = Function(self.fss)
        self.Tf_fss_previous = Function(self.fss)
        
        self.m_n_tableau = np.array([
            5.523e-2,
            8.205e-2,
            1.215e-1,
            2.286e-1,
            2.860e-1,
            2.265e-1,
        ])
        self.lambda_m_n_tableau = np.array([
            5.965e-4,
            1.077e-2,
            1.362e-1,
            1.505e-1,
            6.747e+0,
            2.963e+1,
        ])
        self.g_n_tableau = np.array([
            1.585,
            2.354,
            3.486,
            6.558,
            8.205,
            6.498,
        ])
        self.lambda_g_n_tableau = np.array([
            6.658e-5,
            1.197e-3,
            1.514e-2,
            1.672e-1,
            7.497e-1,
            3.292e+0
        ])
        self.k_n_tableau = np.array([
            7.588e-1,
            7.650e-1,
            9.806e-1,
            7.301e+0,
            1.347e+1,
            1.900e+1,
            7.50e+0,
        ])
        self.lambda_k_n_tableau = np.array([
            5.009e-5,
            9.945e-4,
            2.022e-3,
            1.925e-2,
            1.199e-1,
            2.033e+0,
            1.000e+30,  # instead of Inf
        ])
        
        """
        Intial conditions for partial fictive tempertures, c.f. Nielsen et al., eq. 27
        """
        self.tf_fss_partial_previous = Expression((self.Tf_init), self.fss.element.interpolation_points())
        self.Tf_fss_previous.interpolate(self.tf_fss_partial_previous)
        
        self.tf_fss_partial_current = Expression((self.Tf_init), self.fss.element.interpolation_points())
        self.Tf_fss_current.interpolate(self.tf_fss_partial_current)
        
        # Intermediate functions
        # Fictive temperature
        self.Tf_partial_current = [self.Tf_fss_current for _ in range(0,self.m_n_tableau.size)]
        self.Tf_partial_previous = [self.Tf_fss_previous for _ in range(0,self.m_n_tableau.size)]
        # Deviatoric stress (tensor)
        self.s_partial_current = [Function(self.tfss) for _ in range(0,self.g_n_tableau.size)]
        self.s_partial_previous = [Function(self.tfss) for _ in range(0,self.g_n_tableau.size)]
        self.ds_partial = [Function(self.tfss) for _ in range(0,self.g_n_tableau.size)]
        self.deviatoric_part = [Function(self.tfss) for _ in range(0,self.g_n_tableau.size)]
        # Hydrostatic stress (scalar)
        self.sigma_partial_current = [Function(self.fss) for _ in range(0,self.m_n_tableau.size)]
        self.sigma_partial_previous = [Function(self.fss) for _ in range(0,self.m_n_tableau.size)]
        self.dsigma_partial = [Function(self.fss) for _ in range(0,self.m_n_tableau.size)]
        self.hydrostatic_part = [Function(self.fss) for _ in range(0,self.m_n_tableau.size)]
        # Total stress (tensor)
        self.stress_tensor_current = Function(self.tfss)
        #self.stress_tensor_previous = Function(self.tfss)
        
    def set_initial_condition_Tf(self, fict_temp_value: float) -> None:
        """
        Intial conditions for fictive tempertures, c.f. Nielsen et al., eq. 27
        """
        x = SpatialCoordinate(self.mesh)
        def fict_temp_init(x):
            values = np.full(x.shape[1], fict_temp_value, dtype = ScalarType) 
            return values
        self.Tf_current.interpolate(fict_temp_init)
        self.Tf_previous.interpolate(fict_temp_init)         
    
    def _phi_v(self, T_current):     #ufl.mathfunctions.Exp
        """
        The shift function, c.f. Nielsen et al., eq. 25
        """
        return ufl.exp(
            self.H / self.Rg * (
                1 / self.Tb -
                self.chi / T_current -
                (1 - self.chi) / self.Tf_previous
            )
        )
    
    def _Tf_partial_current(self,T_current,dt, phi_v):  #numpy.ndarray
        """
        Update current values for partial fictive temperature based on previous values.
        C.f. Nielsen et al., eq. 24
        """
        self.Tf_partial_current = (self.lambda_m_n_tableau * self.Tf_partial_previous + T_current * dt * phi_v) / \
                            (self.lambda_m_n_tableau + dt * phi_v)
        return self.Tf_partial_current

    def _Tf_current(self,T_current,dt):   #NoneType - the problem is here
        """
        Perform weighted summation of all partial fictive temperature values.
        C.f. Nielsen et al., eq. 26
        """
        # Reset for accumulation
        Tf_expression = Expression((np.dot(self._Tf_partial_current(T_current,dt,self._phi_v(T_current)),self.m_n_tableau)), self.fss.element.interpolation_points())
        return self.Tf_current.interpolate(Tf_expression)
    
    def compute_Tf_current(self,T_current,dt):  #NoneType
        return self._Tf_current(T_current,dt)  

    def _eps_th(self,T_current,T_previous):   #ufl.tensors.ComponentTensor(delta_eth)
        """
        Thermal strain tensor, c.f. Nielsen et al., eq. 9
        """
        return self.I * (
            self.alpha_solid * (T_current - T_previous)
            + (self.alpha_liquid - self.alpha_solid) * (self.Tf_current - self.Tf_previous)
            )
    
    def _strain_increment_tensor(self,T_current,T_previous):  #ufl.tensors.ComponentTensor(delta_eps)
        """
        The total strain tensor. In absence of mechanical loads, this is trivially given.
        C.f. Nielsen et al., eq. 28
        """
        return -self._eps_th(T_current,T_previous)

    def _eps_dev(self,T_current,T_previous):   #ufl.algebra.Sum(delta_eps_dev)
        """
        The Deviatoric strain increment tensor, c.f. Nielsen et al., eq. 29
        """
        eps = self._strain_increment_tensor(T_current,T_previous)
        return eps - ufl.tr(eps) * self.I
    
    def _phi(self,T_current):      #ufl.mathfunctions.Exp
        """
        The shift function, c.f. Nielsen et al., eq. 5
        """
        return ufl.exp(self.H / self.Rg * (1.0 / self.Tb - 1.0 / T_current))

    def _dxi(self,T_current,T_previous,dt):   #ufl.algebra.Product - the problem is here
        """
        The shifted time, c.f. Nielsen et al., eq. 19
        """
        return dt / 2.0 * (self._phi(T_current) - self._phi(T_previous)) #by assigning T_current, gives the mentioned intergration
    
    def _taylor_exponential(self,T_current,T_previous,which_lambda, dt):    #numpy.ndarray
        """
        The stability correction for dxi -> 0, replaces the exponential
        by a three parts taylor expansion, c.f. Nielsen et al., eq. 20
        """
        expr = 1.0
        dxi = self._dxi(T_current,T_previous,dt)
        if which_lambda == "g":
            lam = self.lambda_g_n_tableau
        elif which_lambda == "k":
            lam = self.lambda_k_n_tableau
        for k in range(0,3):
            expr -= 1.0 / factorial(k) * (- dxi / lam)**k
        return expr
    
    def _ds_partial(self,T_current,T_previous,dt):  #numpy.ndarray
        """
        The partial deviatoric stress increment at previous time, c.f. Nielsen et al., eq. 15a
        """
        self.ds_partial = 2.0 * self.g_n_tableau * self._eps_dev(T_current,T_previous)/self._dxi(T_current,T_previous,dt) * self.lambda_g_n_tableau \
                                    * self._taylor_exponential(T_current,T_previous,"g",dt)
        return self.ds_partial
     
    def _dsigma_partial(self,T_current,T_previous,dt):  #numpy.ndarray
        """
        The partial hydrostatic stress increment at previous time, c.f. Nielsen et al., eq. 15b
        """
        self.dsigma_partial = self.k_n_tableau * self._strain_increment_tensor(T_current,T_previous)/self._dxi(T_current,T_previous,dt) * self.lambda_k_n_tableau \
                                    * self._taylor_exponential(T_current,T_previous,"k",dt)
        return self.dsigma_partial

    def _s_partial_current(self,T_current,T_previous,dt):   #numpy.ndarray
        """
        The partial deviatoric stress increment at current time, c.f. Nielsen et al., eq. 16a
        """
        self.s_partial_current = self._ds_partial(T_current,T_previous,dt) * self._taylor_exponential(T_current,T_previous,"g",dt)
        return self.s_partial_current

    def _sigma_partial_current(self,T_current,T_previous,dt):  #numpy.ndarray
        """
        The partial hydrostatic stress increment at current time, c.f. Nielsen et al., eq. 16b
        """
        self.sigma_partial_current = self._dsigma_partial(T_current,T_previous,dt) * self._taylor_exponential(T_current,T_previous,"k",dt)
        return self.sigma_partial_current
    
    def _deviatoric_part(self,T_current,T_previous,dt):   #numpy.ndarray
        """
        Total deviatoric stresses part at current time, c.f. Nielsen et al., eq. 17 a 
        """
        self.deviatoric_part = self._s_partial_current(T_current,T_previous,dt) + self._ds_partial(T_current,T_previous,dt)
        return self.deviatoric_part
    
    def _hydrostatic_part(self,T_current,T_previous,dt):   #numpy.ndarray
        """
        Total hydrostatic stresses part at current time, c.f. Nielsen et al., eq. 17 b 
        """
        self.hydrostatic_part = self._dsigma_partial(T_current,T_previous,dt) + self._sigma_partial_current(T_current,T_previous,dt)
        return self.hydrostatic_part
    
    def compute_stress_tensor(self,T_current,T_previous,dt): #NoneType
        """
        The total stress tensor at current time, c.f. Nielsen et al., eq. 18 
        """
        stress_expression = Expression((sum(self._deviatoric_part(T_current,T_previous,dt)) + self.I * sum(self._hydrostatic_part(T_current,T_previous,dt))), self.tfss.element.interpolation_points())
        return self.stress_tensor_current.interpolate(stress_expression)

    def _M(self,scaled_time):
        """
        The response function M(xi), c.f. Nielsen et al., Eq. 23   
        args:
            scaled_time: The scaled time xi
        returns:
            expr: A UFL expression representing the response function
        """
        # Initialize value to loop on
        expr = 0.0
        for (m_n,lambda_m_n) in zip(self.m_n_tableau,self.lambda_m_n_tableau):
            expr += m_n * ufl.exp(- scaled_time / lambda_m_n)
        return expr
    
    def _G(self, t: float, g_n_tableau: np.ndarray,lambda_g_n_tableau: np.ndarray):
        """
        The shear relaxation modulus, c.f. Nielsen et al., eq. 11
        """
        g = 0.0
        for (g_n,lambda_g_n) in zip(g_n_tableau,lambda_g_n_tableau):
            g += g_n * ufl.exp( - t/lambda_g_n)
        return g

    def _K(self, t: float, k_n_tableau: np.ndarray,lambda_k_n_tableau: np.ndarray):
        """
        The bulk relaxation modulus, c.f. Nielsen et al., eq. 11
        """
        k = 0.0
        for (k_n,lambda_k_n) in zip(k_n_tableau,lambda_k_n_tableau):
            k += k_n * ufl.exp( - t/lambda_k_n)
        return k
        
    def write_initial_output2(self, output_name: str, t: float = 0.0) -> None:
        self.xdmf2 = io.XDMFFile(self.mesh.comm, f"{output_name}.xdmf", "w")
        self.xdmf2.write_mesh(self.mesh)
        self.xdmf2.write_function(self.Tf_current, t)
        #self.xdmf2.write_function(self.stress_tensor_current, t)
  
    def solve_visco(self,t):
        # assign values
        self.Tf_partial_previous[:][:] = self.Tf_partial_current[:][:]
        self.s_partial_previous[:][:] = self.s_partial_current[:][:]
        self.sigma_partial_previous[:][:] = self.sigma_partial_current[:][:]
        self.Tf_previous.x.array[:] = self.Tf_current.x.array[:]
        
        # Write solution to file
        self.xdmf2.write_function(self.Tf_current, t) 
        #self.xdmf2.write_function(self.stress_tensor_current, t)
        
    def finalize2(self) -> None:
        self.xdmf2.close()