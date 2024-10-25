import jax
import openferro as of
from openferro.interaction import *
from openferro.simulation import *
from openferro.engine import *
import json
from openferro.parallelism import DeviceMesh
from time import time as timer

##########################################################################################
## Define the lattice 
##########################################################################################
L = 250
N = L**3
hydropres =  -4.8e4
config = json.load(open('BaTiO3.json'))
latt_vecs = jnp.eye(3) * config['lattice']['a1']
latt = of.SimpleCubic3D(L, L, L, latt_vecs[0], latt_vecs[1], latt_vecs[2])
bto = of.System(latt)

##########################################################################################
## Define the fields
##########################################################################################
dipole_field = bto.add_field(name="dipole", ftype="Rn", dim=3, value=0.1, mass = 1.0)
# lstrain_field = bto.add_field(name="lstrain", ftype="LocalStrain3D", value=0.0, mass = 40)
gstrain  = bto.add_global_strain(value=jnp.array([0.01,0.01,0.01,0,0,0]), mass = 10.0 *  L**3)
 
##########################################################################################
## Define the Hamiltonian
##########################################################################################
bto.add_self_interaction('self_onsite', 
                         field_name="dipole", 
                         energy_engine=self_energy_onsite_isotropic, 
                         parameters=[config["onsite"]["k2"],config["onsite"]["alpha"], config["onsite"]["gamma"], config["onsite"]["offset"]])
bto.add_self_interaction('short_range_1', 
                         field_name="dipole", 
                         energy_engine=short_range_1stnn_isotropic, 
                         parameters=[config["short_range"]["j1"], config["short_range"]["j2"], config["short_range"]["offset"]])
bto.add_self_interaction('short_range_2', 
                         field_name="dipole", 
                         energy_engine=short_range_2ednn_isotropic, 
                         parameters=[config["short_range"]["j3"], config["short_range"]["j4"], config["short_range"]["j5"], config["short_range"]["offset"]])
bto.add_self_interaction('short_range_3', 
                         field_name="dipole", 
                         energy_engine=short_range_3rdnn_isotropic, 
                         parameters=[config["short_range"]["j6"], config["short_range"]["j7"], config["short_range"]["offset"]],)

bto.add_dipole_dipole_interaction('dipole_ewald', field_name="dipole", prefactor = config["born"]["Z_star"]**2 / config["born"]["epsilon_inf"] )

bto.add_self_interaction('homo_elastic', 
                         field_name="gstrain", 
                         energy_engine=homo_elastic_energy, 
                         parameters= [config["elastic"]["B11"], config["elastic"]["B12"], config["elastic"]["B44"], float(N)])

bto.add_mutual_interaction('homo_strain_dipole', 
                         field_name1="gstrain", 
                         field_name2="dipole", 
                         energy_engine=homo_strain_dipole_interaction, 
                         parameters= [config["elastic_dipole"]["B1xx"], config["elastic_dipole"]["B1yy"], config["elastic_dipole"]["B4yz"], config["elastic_dipole"]["offset"]])

# bto.add_mutual_interaction('elastic', field_name1="lstrain", field_name2="gstrain", energy_engine=elastic_energy, parameters=config["elastic"])
# bto.add_mutual_interaction('inhomo_strain_dipole', field_name1="lstrain", field_name2="dipole", energy_engine=inhomo_strain_dipole_interaction, parameters=config["elastic_dipole"])

 
##########################################################################################
## NPT simulation
##########################################################################################

dt = 0.002
temperature = 300
simulation = SimulationNPTLangevin(bto, dt=dt, temperature=temperature, pressure=-4.8e4, tau=0.1, tauP = 1)
simulation.init_velocity(mode='gaussian')

t0 = timer()
jax.block_until_ready(simulation.step(1, profile=True))
t1 = timer()
print(f"initialization takes: {t1-t0} seconds")

# t0 = timer()
# jax.block_until_ready(simulation.step(500, profile=False))
# t1 = timer()
# print(f"500 steps takes: {t1-t0} seconds")
 
simulation.step(5, profile=True)
 