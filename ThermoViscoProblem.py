from dolfinx.mesh import create_interval, locate_entities_boundary
from mpi4py import MPI
from dolfinx.mesh import locate_entities_boundary
from dolfinx import fem, io, plot, nls, log
from dolfinx.nls import petsc
from dolfinx.io import gmshio
from dolfinx.fem import (FunctionSpace, Function, Constant, dirichletbc, 
                        locate_dofs_geometrical, form, locate_dofs_topological, Expression,
                        assemble_scalar, VectorFunctionSpace, Expression)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, NonlinearProblem
from ufl import (TrialFunction, TestFunction, FiniteElement, TensorElement,
                 VectorElement,grad, dot, inner, Identity, exp, tr,
                 lhs, rhs, Measure, SpatialCoordinate, FacetNormal)#, ds, dx
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import numpy as np
import ufl
from math import ceil
from time import time

class ThermoViscoProblem:
    def __init__(self,mesh_path: str,time: tuple, dt: float,
                 config: dict, model_parameters: dict,
                 jit_options: (dict|None) = None ) -> None:
        self.mesh, self.cell_tags, self.facet_tags = gmshio.read_from_msh(
            mesh_path, MPI.COMM_WORLD, 0, gdim=1)
        
        self.dim = self.mesh.topology.dim
        self.dt = dt
        # The time domain
        self.time = time
        # The current timestep
        self.t = self.time[0]
        self.n_steps = ceil((self.time[1] - self.time[0]) / self.dt)

        self.__init_material_model_constants()
        self.__init_model_parameters(model_parameters=model_parameters)
        self.__init_function_spaces(config=config)
        self.__init_functions()

        self.jit_options = jit_options

        return



    def __init_function_spaces(self,config: dict) -> None:
        """
        Create all necessary finite element data structures
        
        Args:
            - `config`: dictionary holding the types and degrees
                of each finite element
        """
        # Temperature
        self.fe_T = FiniteElement(config["T"]["element"],
                                  self.mesh.ufl_cell(),
                                  config["T"]["degree"])
        self.fs_T = FunctionSpace(mesh=self.mesh,element=self.fe_T)

        # Stress / strain
        self.fe_sigma = TensorElement(config["sigma"]["element"],
                                      self.mesh.ufl_cell(),
                                      config["sigma"]["degree"])
        self.fs_sigma = FunctionSpace(mesh=self.mesh, element=self.fe_sigma)
        
        return
    

    def __init_functions(self) -> None:
        """
        Create all necessary FEniCS functions to model the viscoelastic
        problem, c.f. Nielsen et al., Fig. 5
        """
        # Temperature
        # Heat equation with radiation BC is nonlinear, thus
        # there is no TrialFunction
        # BUG: For C++ forms to compile directly, function names must
        # conform to c++ standards, i.e. no whitespace
        self.T_current = Function(self.fs_T, name="Temperature")
        # For output
        self.T_previous = Function(self.fs_T) # previous time step
        self.T_next = Function(self.fs_T) 
        # For building the weak form
        self.v = TestFunction(self.fs_T)

        

        # Partial fictive temperatures
        # Looping through a list causes problems, likely during
        # JIT and AD
        self.Tf_partial_previous = np.array([Function(self.fs_T) for i in range(0,self.m_n_tableau.size)],
                                            dtype=object)
        self.Tf_partial_current = np.array([Function(self.fs_T,name=f"{i}-th_partial_fictive_temperature") 
                                            for i in range(0,self.m_n_tableau.size)],
                                           dtype=object)

        # Fictive temperature
        self.Tf_previous = Function(self.fs_T)
        self.Tf_current = Function(self.fs_T, name="Fictive_Temperature") 

        # Shift function
        self.phi = Function(self.fs_T, name="Shift_function")
        self.phi_expr = Expression(
            ufl.exp(
            self.H / self.Rg * (
                1.0 / self.Tb
                - self.chi / self.T_current
                - (1.0 - self.chi) / self.Tf_previous
            )),
            self.fs_T.element.interpolation_points()
        )       

        # Thermal strain
        self.thermal_strain = Function(self.fs_sigma, name="Thermal_Strain")
        self.thermal_expr = Expression(
            self.I * (self.alpha_solid * (self.T_current - self.T_previous)
                      + (self.alpha_liquid - self.alpha_solid) * (self.Tf_current - self.Tf_previous)),
            self.fs_sigma.element.interpolation_points()
        )
    
        self.total_strain = Function(self.fs_sigma, name="Total_strain")
        self.total_expr = Expression(
            - self.thermal_strain,
            self.fs_sigma.element.interpolation_points()
        )

        self.deviatoric_strain = Function(self.fs_sigma, name="Deviatoric_strain")
        self.deviatoric_expr = Expression(
            # TODO: Find out if 1/3 applies for all dims
            self.total_strain - 1/self.dim * self.I * tr(self.total_strain),
            self.fs_sigma.element.interpolation_points()
        )
    
        return


    def __init_material_model_constants(self) -> None:
        """
        Define the material model
        """
        # weighting coefficient for temperature and structural energies, c.f. Nielsen et al. eq. 8
        self.chi = 0.5

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
        return
    

    def __init_model_parameters(self,model_parameters: dict) -> None:
        """
        Create all parameters of the viscoelastic model
        as FEniCS constants
        
        Args:
            - `model_parameters`: dict holding the parameter values
        """
        # Identity tensor
        self.I = Identity(self.dim)
        # Intial (fictive) temperture [K]
        self.T_init = Constant(self.mesh, model_parameters["T_0"])
        # Ambient temperature [K]
        self.T_ambient = Constant(self.mesh, model_parameters["T_ambient"])
        # Activation energy [J/mol]
        self.H = Constant(self.mesh, model_parameters["H"])
        # Universal gas constant [J/(mol K)]
        self.Rg = Constant(self.mesh, model_parameters["Rg"])
        # Base temperature [K]
        self.Tb = Constant(self.mesh, model_parameters["Tb"])
        # Solid thermal expansion coefficient [K^-1]
        self.alpha_solid = Constant(self.mesh, model_parameters["alpha_solid"])
        # Liquid thermal expansion coefficient [K^-1]
        self.alpha_liquid = Constant(self.mesh, model_parameters["alpha_liquid"])

        # Right hand side
        self.f = Constant(self.mesh,ScalarType(model_parameters["f"]))
        self.epsilon = Constant(self.mesh,ScalarType(model_parameters["epsilon"])) # view factor
        self.sigma = Constant(self.mesh,ScalarType(model_parameters["sigma"])) # Stefan Boltzmann constant - W/m^2K^4
        self.alpha = Constant(self.mesh,ScalarType(model_parameters["alpha"]))
        self.htc = Constant(self.mesh,ScalarType(model_parameters["htc"]))     # heat convective coefficent - W/(m^2*K) 
        self.rho = Constant(self.mesh, ScalarType(model_parameters["rho"]))    # density kg/m^3
        self.cp = Constant(self.mesh,ScalarType(model_parameters["cp"]))       # specific heat - J/(kg*K)
        self.k = Constant(self.mesh,ScalarType(model_parameters["k"]))         # thermal conductivity - W/(m*K) 

        return
    

    def setup(self, dirichlet_bc: bool = False,
              outfile_name: str = "visco") -> None:
        self._set_initial_condition(temp_value=self.T_init)
        if dirichlet_bc:
            self._set_dirichlet_bc(bc_value=self.T_ambient)
        self._write_initial_output(outfile_name=outfile_name,t=self.t)
        self._setup_weak_form()
        self._setup_solver()


    def _set_initial_condition(self, temp_value: float) -> None:
       self.__set_IC_T(temp_value) 
       self.__set_IC_Tf()
       self.__set_IC_Tf_partial()
    

    def __set_IC_T(self, temp_value: float) -> None:
        x = SpatialCoordinate(self.mesh)
        def temp_init(x):
            values = np.full(x.shape[1], temp_value, dtype = ScalarType) 
            return values
        self.T_previous.interpolate(temp_init)
        self.T_current.interpolate(temp_init)

        return
    

    def __set_IC_Tf(self) -> None:
        """
        Set the initial condition for fictive temperature.
        For t0, Tf = T (c.f. Nielsen et al., eq. 27)
        """
        self.Tf_previous.x.array[:] = self.T_previous.x.array[:]
        self.Tf_current.x.array[:] = self.T_current.x.array[:]

        return
    

    def __set_IC_Tf_partial(self) -> None:
        """
        Set the initial condition for the partial fictive temperature
        values.
        For t0, Tf(n) = T (c.f. Nielsen et al., eq. 27)
        """
        for (previous,current) in zip(self.Tf_partial_previous,self.Tf_partial_current):
            current.x.array[:] = self.T_current.x.array[:]
            previous.x.array[:] = self.T_previous.x.array[:]

        return
    

    def _set_dirichlet_bc(self, bc_value: float) -> None:
        fdim = self.mesh.topology.dim - 1
        boundary_facets = locate_entities_boundary(
            self.mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
        self.bc = fem.dirichletbc(PETSc.ScalarType(bc_value), 
                                  fem.locate_dofs_topological(self.fs, fdim, boundary_facets), self.fs)


    def _write_initial_output(self, outfile_name: str, t: float = 0.0) -> None:
        self.outfile = io.VTKFile(self.mesh.comm, f"output/{outfile_name}.pvd", "w")

        self.outfile.write_mesh(self.mesh)
        # Temperature
        self.outfile.write_function(self.T_current, t)
        # Shift function
        self.outfile.write_function(self.phi,t)
        # Fictive temperature
        self.outfile.write_function(self.Tf_current, t)
        self.outfile.write_function([*self.Tf_partial_current], t)

        

    def _setup_weak_form(self) -> None:
        ds = Measure("exterior_facet",domain=self.mesh)
        dx = Measure("dx",domain=self.mesh)
        
        self.F = (
            # Mass Matrix
            (self.T_current - self.T_previous) * self.v * dx
            + self.dt * (
            # Laplacian
            + dot(grad(self.T_current),grad(self.v)) * dx
            # Right hand side
            - self.f * self.v * dx
            # Radiation
            + 0.001 * (self.sigma * self.epsilon) * (self.T_current**4 - self.T_ambient**4) * self.v * ds
            # Convection
            + 0.001 * self.htc * (self.T_current - self.T_ambient) * self.v * ds
            )
        )
    

    def _setup_solver(self) -> None:
        self.prob = fem.petsc.NonlinearProblem(F=self.F,u=self.T_current,
                                               jit_options=self.jit_options)

        self.solver = petsc.NewtonSolver(self.mesh.comm, self.prob)
        self.solver.convergence_criterion = "incremental"
        self.solver.rtol = 1e-12
        self.solver.report = True

        self.ksp = self.solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = self.ksp.getOptionsPrefix()
        # Linear system produced by heat equation is SPD, thus we can use CG
        opts[f"{option_prefix}ksp_type"] = "cg"
        opts[f"{option_prefix}pc_type"] = "gamg"
        opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        self.ksp.setFromOptions()
    
        
    def _update_values(self,current: Function,previous: Function) -> None:
        # Update ghost values across processes, relevant for MPI computations
        current.x.scatter_forward()
        # assign values
        previous.x.array[:] = current.x.array[:]
        return
    

    def _write_output(self) -> None:
        self.outfile.write_function(
            [
                self.T_current,
                self.phi,
                *self.Tf_partial_current,
                self.Tf_current
            ],
            t=self.t
        )
    

    def solve_timestep(self,t) -> None:
        print(f"t={self.t}")
        self._solve_T()
        self._solve_Tf()
        self._solve_strains()
        self._write_output()
    

    def _solve_T(self) -> None:
        """
        Solve the heat equation for each time step.
        Update values and write current values to file.
        """
        n, converged = self.solver.solve(self.T_current)
        self._update_values(current=self.T_current,
                            previous=self.T_previous)
        return
    
    def _solve_Tf(self) -> None:
        """
        Calculate the current fictive temperature,
        c.f. Nielsen et al., Fig. 5.:

        Steps:
        - compute shift function `phi`
        - compute current partial fictive temperatures
        - compute current fictive temperature
        """
        self.__update_shift_function()
        self.__update_partial_fictive_temperature()
        self.__update_fictive_temperature()
        
        return 
    
    def _solve_strains(self) -> None:
        """
        Calculate the current strains,
        c.f. Nielsen et al., Fig. 5.:

        Steps:
        - compute thermal strain
        - compute total strain (normally from thermal and mechanical strain)
        - compute deviatoric strain
        """ 
        self.__update_thermal_strain()
        self.__update_total_strain()
        self.__update_deviatoric_strain()

        return

    def __update_shift_function(self) -> None:
        self.phi.interpolate(self.phi_expr)

        return

    
    def __update_partial_fictive_temperature(self) -> None:
        """
        Update the partial fictive temperature for the current timestep.
        C.f. Nielsen et al., Eq. 24
        """        
        for i in range(0,self.Tf_partial_current.size):
            self.Tf_partial_current[i].x.array[:] = (
                self.lambda_m_n_tableau[i] * self.Tf_partial_previous[i].vector
                + self.T_current.vector * self.dt * self.phi.vector
            ) / (
                self.lambda_m_n_tableau[i] + self.dt * self.phi.vector
            )
            self._update_values(current=self.Tf_partial_current[i],
                                previous=self.Tf_partial_previous[i])

        return


    def __update_fictive_temperature(self) -> None:
        """
        Update the fictive temperature for the current timestep.
        C.f. Nielsen et al., Eq. 26
        """
        self.Tf_current.x.array[:] = 0.0
        for i in range(0,self.m_n_tableau.size):
            self.Tf_current.x.array[:] += self.Tf_partial_current[i].vector * self.m_n_tableau[i]
        self._update_values(current=self.Tf_current,
                            previous=self.Tf_previous)

        return
    
    
    def __update_thermal_strain(self) -> None:
        """
        Update the thermal strain for the current timestep.
        c.f. Nielsen et al., Eq. 9
        """
        self.thermal_strain.interpolate(self.thermal_expr)

        return
    

    def __update_total_strain(self) -> None:
        """
        Update the total strain for the current timestep.
        c.f. Nielsen et al., Eq. 28
        """
        self.total_strain.interpolate(self.total_expr)

        return
    

    def __update_deviatoric_strain(self) -> None:
        """
        Update the total strain for the current timestep.
        c.f. Nielsen et al., Eq. 28
        """
        self.deviatoric_strain.interpolate(self.deviatoric_expr)

        return


    def solve(self) -> None:
        print("Starting solve")
        t_start = time()
        for _ in range(self.n_steps):
            self.t += self.dt
            self.solve_timestep(t=self.t)
        t_end = time()
        print(f"Solve finished in {t_end - t_start} seconds.")

        self._finalize()
        return


    def _finalize(self) -> None:
        self.outfile.close()