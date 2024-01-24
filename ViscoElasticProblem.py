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


class ViscoElasticModel:
    def __init__(self, parameters: dict, mesh, dt, degree=1, tensor_degree=1) -> None:
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
        self.T_previous = Function(self.fs) # previous time step
        self.T_next = Function(self.fs)
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
        self.Tf_previous = Function(self.fs)
        self.Tf_current = Function(self.fs)        
        self.Tf_next = Function(self.fs)
        self.Tf_fss_previous = Function(self.fs)
        self.Tf_fss_current = Function(self.fs)
        self.Tf_fss_next = Function(self.fs)

        
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
            1.090e+1,
            
        ])
        self.lambda_k_n_tableau = np.array([
            5.009e-5,
            9.945e-4,
            2.022e-3,
            1.925e-2,
            1.199e-1,
            2.033e+0,
              # instead of Inf
        ])
        
        """
        Intial conditions for partial fictive tempertures, c.f. Nielsen et al., eq. 27
        """
        self.tf_fss_partial_previous = Expression((self.Tf_init), self.fs.element.interpolation_points())
        self.Tf_fss_previous.interpolate(self.tf_fss_partial_previous)
        
        self.tf_fss_partial_current = Expression((self.Tf_init), self.fs.element.interpolation_points())
        self.Tf_fss_current.interpolate(self.tf_fss_partial_current)
        
        self.tf_fss_partial_next = Expression((self.Tf_init), self.fs.element.interpolation_points())
        self.Tf_fss_next.interpolate(self.tf_fss_partial_next)
        
        # Intermediate functions
        # Fictive temperature
        self.Tf_partial_previous = [self.Tf_fss_previous for _ in range(0,self.m_n_tableau.size)]
        self.Tf_partial_current = [self.Tf_fss_previous for _ in range(0,self.m_n_tableau.size)]
        self.Tf_partial_next = [self.Tf_fss_previous for _ in range(0,self.m_n_tableau.size)]
        
        # Deviatoric stress (tensor)
        self.s_relax_previous = [Function(self.tfs) for _ in range(0,self.g_n_tableau.size)]
        self.s_relax_current = [Function(self.tfs) for _ in range(0,self.g_n_tableau.size)]
        self.s_relax_next = [Function(self.tfs) for _ in range(0,self.g_n_tableau.size)]
        self.ds_visco = [Function(self.tfs) for _ in range(0,self.g_n_tableau.size)]
        self.deviatoric_part_previous = [Function(self.tfs) for _ in range(0,self.g_n_tableau.size)]
        self.deviatoric_part_current = [Function(self.tfs) for _ in range(0,self.g_n_tableau.size)]
        self.deviatoric_part_next = [Function(self.tfs) for _ in range(0,self.g_n_tableau.size)]

        # Hydrostatic stress (scalar)
        self.sigma_relax_previous = [Function(self.fs) for _ in range(0,self.m_n_tableau.size)]
        self.sigma_relax_current = [Function(self.fs) for _ in range(0,self.m_n_tableau.size)]
        self.sigma_relax_next = [Function(self.fs) for _ in range(0,self.m_n_tableau.size)]
        self.dsigma_visco = [Function(self.fs) for _ in range(0,self.m_n_tableau.size)]
        self.hydrostatic_part_previous = [Function(self.fs) for _ in range(0,self.m_n_tableau.size)]
        self.hydrostatic_part_current = [Function(self.fs) for _ in range(0,self.m_n_tableau.size)]
        self.hydrostatic_part_next = [Function(self.fs) for _ in range(0,self.m_n_tableau.size)]

        # Total stress (tensor)
        self.stress_tensor_previous = Function(self.tfs)
        self.stress_tensor_current = Function(self.tfs)
        self.stress_tensor_next = Function(self.tfs)
     
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
    
    def _phi_v(self,T_current,Tf_previous):    
        """
        The shift function, c.f. Nielsen et al., eq. 25
        Returns:ufl.mathfunctions.Exp
        """
        return ufl.exp(
            self.H / self.Rg * (
                1 / self.Tb -
                self.chi / T_current -
                (1 - self.chi) / (Tf_previous)
            )
        )
    
    def _Tf_partial_current(self,T_current,dt,phi_v): 
        """
        Update current values for partial fictive temperature based on previous values.
        C.f. Nielsen et al., eq. 24
        Returns:numpy.ndarray
        """
        self.Tf_partial_current = (self.lambda_m_n_tableau * self.Tf_partial_previous + T_current * dt * phi_v) / \
                            (self.lambda_m_n_tableau + dt * phi_v)
        return self.Tf_partial_current

    def _Tf_current(self,T_current,Tf_previous,dt):  
        """
        Perform weighted summation of all partial fictive temperature values.
        C.f. Nielsen et al., eq. 26
        Returns:NoneType
        """
        # Reset for accumulation
        Tf_expression = Expression((np.dot(self._Tf_partial_current(T_current,dt,self._phi_v(T_current,Tf_previous)),self.m_n_tableau)), self.fs.element.interpolation_points())
        return self.Tf_current.interpolate(Tf_expression)
    
    def compute_Tf_current(self,T_current,Tf_previous,dt):
        """
        Returns:NoneType
        """
        return self._Tf_current(T_current,Tf_previous,dt)  

    def _eps_th(self,T_current,T_previous,Tf_current,Tf_previous):  
        """
        Thermal strain tensor, c.f. Nielsen et al., eq. 9
        Returns:ufl.tensors.ComponentTensor(delta_eth)
        """
        return self.I * (
            self.alpha_solid * (T_current - T_previous)
            + (self.alpha_liquid - self.alpha_solid) * (Tf_current - Tf_previous)
            )
    
    def _strain_increment_tensor(self,T_current,T_previous,Tf_current,Tf_previous):
        """
        The total strain tensor. In absence of mechanical loads, this is trivially given.
        C.f. Nielsen et al., eq. 28
        Returns:ufl.tensors.ComponentTensor(delta_eps)
        """
        return -self._eps_th(T_current,T_previous,Tf_current,Tf_previous)

    def _eps_dev(self,T_current,T_previous,Tf_current,Tf_previous): 
        """
        The Deviatoric strain increment tensor, c.f. Nielsen et al., eq. 29
        Returns:ufl.algebra.Sum(delta_eps_dev)
        """
        eps = self._strain_increment_tensor(T_current,T_previous,Tf_current,Tf_previous)
        return eps - ufl.tr(eps) * self.I
    
    def _phi(self,T_current):  
        """
        The shift function, c.f. Nielsen et al., eq. 5
        Returns:ufl.mathfunctions.Exp
        """
        return ufl.exp(self.H / self.Rg * (1.0 / self.Tb - 1.0 / T_current))

    def _dxi(self,T_next,T_current,dt): #the problem is here
        """
        The shifted time, c.f. Nielsen et al., eq. 19
        Returns:ufl.algebra.Product
        """
        return dt / 2.0 * (self._phi(T_next) - self._phi(T_current))
    
    def _taylor_exponential(self,T_next,T_current,which_lambda,dt):
        """
        The stability correction for dxi -> 0, replaces the exponential
        by a three parts taylor expansion, c.f. Nielsen et al., eq. 20
        Returns:numpy.ndarray
        """
        expr = 1.0
        dxi = self._dxi(T_next,T_current,dt)
        if which_lambda == "g":
            lam = self.lambda_g_n_tableau
        elif which_lambda == "k":
            lam = self.lambda_k_n_tableau
        for k in range(0,3):
            expr -= 1.0 / factorial(k) * (- dxi / lam)**k
        return expr
    
    def _exponential_term(self,T_next,T_current,which_lambda,dt):
        """
        The exponential term in 16 a,b
        Returns:numpy.ndarray
        """
        expr1 = 0.0
        dxi = self._dxi(T_next,T_current,dt)
        if which_lambda == "g":
            lam1 = self.lambda_g_n_tableau
        elif which_lambda == "k":
            lam1 = self.lambda_k_n_tableau
        for k in range(0,3):
            expr1 += 1.0 / factorial(k) * (- dxi / lam1)**k
        return expr1
    
    def _ds_visco(self,T_next,T_current,T_previous,Tf_current,Tf_previous,dt):
        """
        The partial deviatoric stress increment at previous time, c.f. Nielsen et al., eq. 15a
        Returns:numpy.ndarray
        """
        self.ds_visco = 2.0 * self.g_n_tableau * (self._eps_dev(T_current,T_previous,Tf_current,Tf_previous)) * self.lambda_g_n_tableau \
                                    * self._taylor_exponential(T_next,T_current,"g",dt)/(self._dxi(T_next,T_current,dt))
        return self.ds_visco
     
    def _dsigma_visco(self,T_next,T_current,T_previous,Tf_current,Tf_previous,dt):
        """
        The partial hydrostatic stress increment at previous time, c.f. Nielsen et al., eq. 15b
        Returns:numpy.ndarray
        """
        self.dsigma_visco = self.k_n_tableau * ufl.tr(self._strain_increment_tensor(T_current,T_previous,Tf_current,Tf_previous))* self.lambda_k_n_tableau \
                                    * self._taylor_exponential(T_next,T_current,"k",dt)/(self._dxi(T_next,T_current,dt))
        return self.dsigma_visco

    def _s_relax_next(self,T_next,T_current,T_previous,Tf_current,Tf_previous,dt): 
        """
        The partial deviatoric stress increment at current time, c.f. Nielsen et al., eq. 16a
        Returns:numpy.ndarray
        """
        self.s_relax_next = self._ds_visco(T_next,T_current,T_previous,Tf_current,Tf_previous,dt) * self._exponential_term(T_next,T_current,"g",dt)
        return self.s_relax_next

    def _sigma_relax_next(self,T_next,T_current,T_previous,Tf_current,Tf_previous,dt):
        """
        The partial hydrostatic stress increment at current time, c.f. Nielsen et al., eq. 16b
        Returns:numpy.ndarray
        """
        self.sigma_relax_next = self._dsigma_visco(T_next,T_current,T_previous,Tf_current,Tf_previous,dt) * self._exponential_term(T_next,T_current,"k",dt)
        return self.sigma_relax_next
    
    def _deviatoric_part_next(self,T_next,T_current,T_previous,Tf_current,Tf_previous,dt): 
        """
        Total deviatoric stresses part at current time, c.f. Nielsen et al., eq. 17 a 
        Returns:numpy.ndarray
        """
        self.deviatoric_part_next = self._ds_visco(T_next,T_current,T_previous,Tf_current,Tf_previous,dt) + self._s_relax_next(T_next,T_current,T_previous,Tf_current,Tf_previous,dt)
        return self.deviatoric_part_next
    
    def _hydrostatic_part_next(self,T_next,T_current,T_previous,Tf_current,Tf_previous,dt):
        """
        Total hydrostatic stresses part at current time, c.f. Nielsen et al., eq. 17 b 
        Returns:numpy.ndarray
        """
        self.hydrostatic_part_next = self._dsigma_visco(T_next,T_current,T_previous,Tf_current,Tf_previous,dt) + self._sigma_relax_next(T_next,T_current,T_previous,Tf_current,Tf_previous,dt)
        return self.hydrostatic_part_next
    
    def compute_stress_tensor_next(self,T_next,T_current,T_previous,Tf_current,Tf_previous,dt):
        """
        The total stress tensor at current time, c.f. Nielsen et al., eq. 18 
        Returns:NoneType
        """
        stress_expression = Expression((sum(self._deviatoric_part_next(T_next,T_current,T_previous,Tf_current,Tf_previous,dt)) + self.I * sum(self._hydrostatic_part_next(T_next,T_current,T_previous,Tf_current,Tf_previous,dt))), self.tfs.element.interpolation_points())
        return self.stress_tensor_next.interpolate(stress_expression)

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
            k += k_n * ufl.exp(-t/lambda_k_n)
        return k
        
    def write_initial_output2(self, output_name: str, t: float = 0.0) -> None:
        self.xdmf2 = io.XDMFFile(self.mesh.comm, f"{output_name}.xdmf", "w")
        self.xdmf2.write_mesh(self.mesh)
        #self.xdmf2.write_function(self.Tf_next, t)
        self.xdmf2.write_function(self.stress_tensor_next, t)
  
    def solve_visco(self,t):
        # assign values
        self.Tf_partial_previous[:][:] = self.Tf_partial_current[:][:]
        self.Tf_previous.x.array[:] = self.Tf_current.x.array[:]
        #self.Tf_partial_current = self.Tf_partial_next.copy()
        #self.s_relax_previous[:][:]  = self.s_relax_current[:][:] 
        self.s_relax_current[:][:] = self.s_relax_next[:][:] 
        #self.sigma_relax_previous[:][:] = self.sigma_relax_current[:][:] 
        self.sigma_relax_current[:][:]  = self.sigma_relax_next[:][:] 
        self.Tf_current.x.array[:] = self.Tf_next.x.array[:]
        #self.stress_tensor_previous.x.array[:] = self.stress_tensor_current.x.array[:]
        self.stress_tensor_current.x.array[:] = self.stress_tensor_next.x.array[:]
        #self.hydrostatic_part_previous[:][:]  = self.hydrostatic_part_current[:][:] 
        self.hydrostatic_part_current[:][:]  = self.hydrostatic_part_next[:][:] 
        #self.deviatoric_part_previous[:][:]  = self.deviatoric_part_current[:][:] 
        self.deviatoric_part_current[:][:]  = self.deviatoric_part_next[:][:] 
        self.T_previous.x.array[:] = self.T_current.x.array[:]
        self.T_current.x.array[:] = self.T_next.x.array[:]
        # Write solution to file
        #self.xdmf2.write_function(self.Tf_current, t) 
        self.xdmf2.write_function(self.stress_tensor_next, t)
        
    def finalize2(self) -> None:
        self.xdmf2.close()
        

#revise all functions
#compute untill equiblrium, time loops