from geometry import create_mesh
from ThermoViscoProblem import ThermoViscoProblem
from OutgoingDto import OutgoingDto
import logging
import traceback
import numpy as np

# Logging
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
logger.propagate = False
formatter = logging.Formatter(
    "{asctime} - {levelname} - {filename} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
file_handler.setLevel("INFO")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Enable full compiler optimizations for generated
# C++ code
jit_options = {"cffi_extra_compile_args": ["-O3", "-march=native"]}

# Time domain
t_start = 0.0
t_end = 602.1  # 50.0
# time = (0.0, 50.0)
time = (t_start, t_end)

# Time discretization
dt = 0.1

# Simulation starting time
t = t_start

# Create Mesh for Simulation
mesh_path = "mesh1d.msh"
create_new_mesh = False

try:
    if create_new_mesh:
        create_mesh(path=mesh_path)
        logger.info("Mesh created")

    # Create VTX Files for visualization in Paraview
    create_vtx_files = False

    fe_config = {
        "T": {"element": "DG", "degree": 1},
        "sigma": {"element": "CG", "degree": 1},
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
        "alpha": 1.0,
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

    model = ThermoViscoProblem(
        mesh_path=mesh_path,
        config=fe_config,
        time=time,
        dt=dt,
        model_parameters=model_params,
        jit_options=jit_options,
    )

    model.setup(dirichlet_bc=False, create_vtx_files=create_vtx_files)
    dto = model.solve()
    # dto = OutgoingDto()
    logger.info(
        f"Simulation executed. - Execution Time: {np.round(model.execution_time,6)}s"
    )

    logger.info(f"Number of elements in OutgoingDto: {dto.num_elements()}")

    result = dto.to_json()
    logger.info("Code executed")


except Exception as e:
    logger.error(e, exc_info=True)
    result = OutgoingDto().to_json()
