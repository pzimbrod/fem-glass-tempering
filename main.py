from geometry import create_mesh
from ThermoViscoProblem import ThermoViscoProblem
import matplotlib.pyplot as plt
import numpy as np
from AnalyticalSoln import AnalyticalSoln




# Enable full compiler optimizations for generated
# C++ code
jit_options = {
            "cffi_extra_compile_args": ["-O3", "-march=native"]
        }

# Time domain (whole time domain or for each zone)
t_start = 0.0
t_end = 20.0
time = (t_start, t_end)

dt = 0.1
t = t_start

# Problem dimensions (1D, 2D, or 3D)
problem_dim = 3

# Name of the glass zone 
Zone_name = "all"

mesh_path = f"mesh{problem_dim}d.msh"

create_new_mesh = True

if create_new_mesh:
    create_mesh(path=mesh_path,dim=problem_dim, name=Zone_name, t_start=t_start, t_end=t_end)

fe_config = {
    "T":        {"element": "CG", "degree": 1},
    "sigma":    {"element": "CG", "degree": 1},
    "U":        {"element": "CG", "degree": 1}
}

model_params = {
    # Volumetric heat dissipation
    "f": 0.0,
    # Radiative heat emissivity
    "epsilon": 0.93,
    # Boltzmann constant
    "sigma": 5.670e-8,
    # Ambient temperature
    "T_ambient": 293.15,
    # Initial temperature
    "T_0": 923.15,
    "alpha": 40.0,    #ideal 2
    # Convective heat transfer coefficient (Controlling cooling rate)
    "htc": 280.1,
    # Material density
    "rho": 2500.0,
    # Specific heat capacity
    "cp": 1433.0,
    # Heat conduction coefficient
    "k": 1.0,
    "Hv": 457.05e3,
    "H": 627.8e3,
    "Tb": 869.0,
    "Rg": 8.314,
    "alpha_solid": 9.10e-6,
    "alpha_liquid": 25.10e-6,
    "Tf_init": 923.1,
    "lambda_": 1.25,
    "mu": 1.0,
    "Young's_modulus": 72.0e9,
    "Possion_ratio": 0.22,
}

analytical_constants = {
        "a":    0.2957,
        "c":    1.676e3,
        "Tb":   779.9,
        "E0":   70e9,
        "b":    6.937,
        "H":    22.380e3,
        "k":    -1.231e8,
        "lambda_":   0.7012,
        }
model = ThermoViscoProblem(mesh_path=mesh_path,problem_dim=problem_dim,
                           config=fe_config,time=time,dt=dt,model_parameters=model_params, analy_parameters=analytical_constants,
                           jit_options=jit_options)

model.setup(dirichlet_bc_mech=False)
model.solve()

'''t_ = np.linspace(start=0.0, stop=150.0, num=1500)

#Variables of analytical equations in arrays over time loop

T_ = [AnalyticalSoln .T(t_i, constants=analytical_constants) for t_i in t_]
phi_ = [AnalyticalSoln.phi(t_i, constants=analytical_constants) for t_i in t_]
E_ = [AnalyticalSoln.E(t_i, constants=analytical_constants) for t_i in t_]
xi_ = [AnalyticalSoln.xi(t_i, constants=analytical_constants) for t_i in t_]
epsilon_ = [AnalyticalSoln.epsilon(t_i, constants=analytical_constants) for t_i in t_]
sigma_ = [AnalyticalSoln.stress(t_i, constants=analytical_constants) for t_i in t_]
sigma_analytical_ = [AnalyticalSoln.sigma_analytical(t_i, constants=analytical_constants) for t_i in t_]

fig, axs = plt.subplots(2, 3)

# Temperatures
plt.subplot(2, 3, 1)
plt.plot(t_, T_, label='Analytical results', color='r')
plt.plot(t_, model.avg_T, label='Simulated results', color='b')
plt.title('Plot of Temperature vs Time')
plt.xlabel('Time (t)')
plt.ylabel('Tempertures (K)')
plt.legend()
plt.grid(True)

# Shift functions
plt.subplot(2, 3, 2)
plt.plot(t_, phi_, label='Analytical results', color='r')
plt.plot(t_, model.avg_phi, label='Simulated results', color='b')
plt.title('Plot of shift function vs Time')
plt.xlabel('Time (t)')
plt.ylabel('Shift function')
plt.legend()
plt.grid(True)

#Scaled times
plt.subplot(2, 3, 3)
plt.plot(t_, xi_, label='Analytical results', color='r')
plt.plot(t_, model.avg_xi, label='Simulated results', color='b')
plt.title('Plot of scaled times vs Time')
plt.xlabel('Time (t)')
plt.ylabel('Scaled time')
plt.legend()
plt.grid(True)

#Strains
plt.subplot(2, 3, 4)
plt.plot(t_, epsilon_, label='Analytical results', color='r')
plt.plot(t_, model.avg_t_epsilon, label='Simulated results', color='b')
plt.title('Plot of total strains vs Time')
plt.xlabel('Time (t)')
plt.ylabel('Total strain')
plt.legend()
plt.grid(True)

#Stresses
plt.subplot(2, 3, 5)
plt.plot(t_, sigma_analytical_, label='Analytical results', color='r')
plt.plot(t_, model.avg_t_sigma, label='Simulated results', color='b')
plt.title('Plot of stresses vs Time')
plt.xlabel('Time (t)')
plt.ylabel('Stress (MPa)')
plt.legend()
plt.grid(True)

#Thermal Strains
plt.subplot(2, 3, 6)
#plt.plot(t_, epsilon_, label='Analytical results', color='r')
plt.plot(t_, model.avg_thermal_epsilon, label='Simulated results', color='b')
plt.title('Plot of thermal strains vs Time')
plt.xlabel('Time (t)')
plt.ylabel('Thermal strain')
plt.legend()
plt.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()
'''
