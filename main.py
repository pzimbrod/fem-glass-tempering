from geometry import create_mesh
from ThermoViscoProblem import ThermoViscoProblem

# Enable full compiler optimizations for generated
# C++ code
jit_options = {
            "cffi_extra_compile_args": ["-O3", "-march=native"]
        }

# Time domain
t_start = 0.0
t_end = 50.0
time = (0.0, 50.0)

dt = 0.1
t = t_start

mesh_path = "mesh1d.msh"
create_new_mesh = False

if create_new_mesh:
    create_mesh(path=mesh_path)

fe_config = {
    "T":        {"element": "CG", "degree": 1},
    "sigma":    {"element": "CG", "degree": 1},
}

model_params = {
    # Volumetric heat dissipation
    "f": 0.0,
    # Radiative heat emissivity
    "epsilon": 0.93,
    # Boltzmann constant
    "sigma": 5.670e-8,
    # Ambient temperature
    "T_ambient": 600.0,
    # Initial temperature
    "T_0": 800.0,
    # Convective heat transfer coefficient
    "alpha": 0.0005,
    "htc": 280.1,
    # Material density
    "rho": 2500.0,
    # Specific heat capacity
    "cp": 1433.0,
    # Heat conduction coefficient
    "k": 1.0,
    "H": 627.8e3,
    "Tb": 869.0e0,
    "Rg": 8.314,
    "alpha_solid": 9.10e-6,
    "alpha_liquid": 25.10e-6,
    "Tf_init": 873.0,
}

model = ThermoViscoProblem(mesh_path=mesh_path,config=fe_config,
                           time=time,dt=dt,model_parameters=model_params,
                           jit_options=jit_options)

model.setup(dirichlet_bc=False,outfile_name="visco",outfile_name1="stresses")
model.solve()
