import jax
import openferro as of
from openferro.interaction import *
from openferro.simulation import *
from openferro.engine import *
from openferro.ewald import get_dipole_dipole_ewald
from matplotlib import pyplot as plt
import json
from time import time as get_time
jax.config.update("jax_enable_x64", True)


##########################################################################################
## Define the lattice, order parameters, and the Hamiltonian
##########################################################################################

L = 12
N = L**3
hydropres =  -4.8e4
config = json.load(open('BaTiO3.json'))
latt_vecs = jnp.eye(3) * config['lattice']['a1']
latt = of.BravaisLattice3D(L, L, L, latt_vecs[0], latt_vecs[1], latt_vecs[2])
bto = of.System(latt, pbc=True)

## define fields
dipole_field = bto.add_field(name="dipole", ftype="Rn", dim=3, value=0.1, mass = 1)
# lstrain_field = bto.add_field(name="lstrain", ftype="local_strain", value=0.0, mass = 40)
gstrain_field = bto.add_field(name="gstrain", ftype="global_strain", value=jnp.array([0.01,0.01,0.01,0,0,0]), mass = 200 *  L**3)

## define Hamiltonian
bto.add_self_interaction('self_onsite', 
                         field_name="dipole", 
                         energy_engine=self_energy_onsite_isotropic, 
                         parameters=[config["onsite"]["k2"],config["onsite"]["alpha"], config["onsite"]["gamma"], config["onsite"]["offset"]], 
                         enable_jit=True)
bto.add_self_interaction('short_range_1', 
                         field_name="dipole", 
                         energy_engine=short_range_1stnn_isotropic, 
                         parameters=[config["short_range"]["j1"], config["short_range"]["j2"], config["short_range"]["offset"]], 
                         enable_jit=True)
bto.add_self_interaction('short_range_2', 
                         field_name="dipole", 
                         energy_engine=short_range_2ednn_isotropic, 
                         parameters=[config["short_range"]["j3"], config["short_range"]["j4"], config["short_range"]["j5"], config["short_range"]["offset"]], 
                         enable_jit=True)
bto.add_self_interaction('short_range_3', 
                         field_name="dipole", 
                         energy_engine=short_range_3rdnn_isotropic, 
                         parameters=[config["short_range"]["j6"], config["short_range"]["j7"], config["short_range"]["offset"]], 
                         enable_jit=True)
bto.add_self_interaction('dipole_dipole', 
                         field_name="dipole", 
                         energy_engine=get_dipole_dipole_ewald(latt), 
                         parameters=[config["born"]["Z_star"],  config["born"]["epsilon_inf"]], 
                         enable_jit=True)

bto.add_self_interaction('homo_elastic', 
                         field_name="gstrain", 
                         energy_engine=homo_elastic_energy, 
                         parameters= [config["elastic"]["B11"], config["elastic"]["B12"], config["elastic"]["B44"], float(N)],
                         enable_jit=True)

bto.add_mutual_interaction('homo_strain_dipole', 
                         field_name1="gstrain", 
                         field_name2="dipole", 
                         energy_engine=homo_strain_dipole_interaction, 
                         parameters= [config["elastic_dipole"]["B1xx"], config["elastic_dipole"]["B1yy"], config["elastic_dipole"]["B4yz"], config["elastic_dipole"]["offset"]],
                         enable_jit=True)

# bto.add_mutual_interaction('elastic', field_name1="lstrain", field_name2="gstrain", energy_engine=elastic_energy, parameters=config["elastic"], enable_jit=True)
# bto.add_mutual_interaction('inhomo_strain_dipole', field_name1="lstrain", field_name2="dipole", energy_engine=inhomo_strain_dipole_interaction, parameters=config["elastic_dipole"], enable_jit=True)

 
##########################################################################################
## NPT cooling simulation
##########################################################################################

log_freq = 100
total_time = 1000
dt = 0.002
relax_steps = int(50/dt)
total_steps = int(total_time / dt)
niters = total_steps // log_freq
field_history = []
gs_history = []


temperature = 300
simulation = SimulationNPTLangevin(bto, dt=dt, temperature=temperature, pressure=-4.8e4, tau=0.1, tauP = 1)
simulation.init_velocity(mode='gaussian')

t0 = get_time()
jax.block_until_ready(simulation.step(1, profile=False))
t1 = get_time()
print(f"initialization takes: {t1-t0} seconds")

t0 = get_time()
# simulation.step(500, profile=False)
jax.block_until_ready(simulation.step(500, profile=False))
t1 = get_time()
print(f"500 steps takes: {t1-t0} seconds")
 
 