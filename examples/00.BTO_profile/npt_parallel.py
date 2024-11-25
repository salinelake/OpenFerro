import jax
import openferro as of
from openferro.interaction import *
from openferro.simulation import *
from openferro.engine.elastic import *
from openferro.engine.ferroelectric import *
import json
from openferro.parallelism import DeviceMesh
from time import time as get_time


##########################################################################################
## Define the lattice 
##########################################################################################
l1 = 1024
l2 = 1024
l3 = 64   ## 192 will be billion atoms.
hydropres =  -4.8e4
config = json.load(open('BaTiO3.json'))
latt_vecs = jnp.eye(3) * config['lattice']['a1']
latt = of.SimpleCubic3D(l1, l2, l3, latt_vecs[0], latt_vecs[1], latt_vecs[2])
bto = of.System(latt)

##########################################################################################
## Define the fields
##########################################################################################
dipole_field = bto.add_field(ID="dipole", ftype="Rn", dim=3, value=0.1, mass = 1.0)
lstrain_field = bto.add_field(ID="lstrain", ftype="LocalStrain3D", value=0.0, mass = 40)
gstrain  = bto.add_global_strain(value=jnp.array([0.01,0.01,0.01,0,0,0]), mass = 10.0 *  latt.nsites)

##########################################################################################
## move fields to multi-GPUs
##########################################################################################
gpu_mesh = DeviceMesh(num_rows=2, num_cols=2)
# gpu_mesh = DeviceMesh(jax.devices()[:2], num_rows=2, num_cols=1)
bto.move_fields_to_multi_devs(gpu_mesh)

##########################################################################################
## Define the Hamiltonian
##########################################################################################
bto.add_dipole_onsite_interaction('self_onsite', field_ID="dipole", K2=config["onsite"]["k2"], alpha=config["onsite"]["alpha"], gamma=config["onsite"]["gamma"])
bto.add_dipole_interaction_1st_shell('short_range_1', field_ID="dipole", j1=config["short_range"]["j1"], j2=config["short_range"]["j2"])
bto.add_dipole_interaction_2nd_shell('short_range_2', field_ID="dipole", j3=config["short_range"]["j3"], j4=config["short_range"]["j4"], j5=config["short_range"]["j5"])
bto.add_dipole_interaction_3rd_shell('short_range_3', field_ID="dipole", j6=config["short_range"]["j6"], j7=config["short_range"]["j7"])
bto.add_dipole_dipole_interaction('dipole_ewald', field_ID="dipole", prefactor = config["born"]["Z_star"]**2 / config["born"]["epsilon_inf"] )
bto.add_homo_elastic_interaction('homo_elastic', field_ID="gstrain", B11=config["elastic"]["B11"], B12=config["elastic"]["B12"], B44=config["elastic"]["B44"])
bto.add_homo_strain_dipole_interaction('homo_strain_dipole', field_1_ID="gstrain", field_2_ID="dipole", B1xx=config["elastic_dipole"]["B1xx"], B1yy=config["elastic_dipole"]["B1yy"], B4yz=config["elastic_dipole"]["B4yz"])
bto.add_inhomo_elastic_interaction('inhomo_elastic', field_ID="lstrain", B11=config["elastic"]["B11"], B12=config["elastic"]["B12"], B44=config["elastic"]["B44"])
bto.add_inhomo_strain_dipole_interaction('inhomo_strain_dipole', field_1_ID="lstrain", field_2_ID="dipole", B1xx=config["elastic_dipole"]["B1xx"], B1yy=config["elastic_dipole"]["B1yy"], B4yz=config["elastic_dipole"]["B4yz"])

##########################################################################################
## NPT simulation
##########################################################################################

dt = 0.002
temperature = 300
dipole_field.set_integrator('isothermal', dt=dt, temp=temperature, tau=0.1)
lstrain_field.set_integrator('isothermal', dt=dt, temp=temperature, tau=1)
gstrain.set_integrator('isothermal', dt=dt, temp=temperature, tau=1)
simulation = SimulationNPTLangevin(bto, pressure=-4.8e4 )
simulation.init_velocity(mode='gaussian', temp=temperature)

t0 = get_time()
jax.block_until_ready(simulation.run(1, profile=True))
t1 = get_time()
print(f"initialization takes: {t1-t0} seconds")


t0 = get_time()
jax.block_until_ready(simulation.run(500, profile=False))
t1 = get_time()
print(f"500 steps takes: {t1-t0} seconds")
 

# simulation.run(5, profile=True)
 