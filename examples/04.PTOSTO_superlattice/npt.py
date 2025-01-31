"""
This example simulates domain motion in a PTO/STO superlattice under an external electric field. 
The system is large. Expecting a bunch of "Constant folding an instruction is taking > **s: " warnings from JAX. Do not worry. 
"""

import json
import logging
import os
import jax.numpy as jnp
import openferro as of
from openferro.interaction import *
from openferro.simulation import *
from openferro.engine.elastic import *
from openferro.engine.ferroelectric import *
from openferro.engine.ferroelectric_superlatt import *
from openferro.units import Constants
from openferro.parallelism import DeviceMesh

##########################################################################################
## IO
##########################################################################################
os.makedirs('output', exist_ok=True)
logging.basicConfig(level=logging.INFO, filename='simulation.log')
pto_config = json.load(open('PbTiO3.json'))
sto_config = json.load(open('SrTiO3.json'))

##########################################################################################
## Define the lattice 
##########################################################################################
m=48
n=16
l1 = 256
l2 = 256
l3 = m+n
N = l1*l2*l3
hydropressure = 0
latt_vecs = jnp.eye(3) * pto_config['lattice']['a1']
latt = of.SimpleCubic3D(l1, l2, l3, latt_vecs[0], latt_vecs[1], latt_vecs[2])
system = of.System(latt)
Z_pto = pto_config["born"]["Z_star"] 
Z_sto = sto_config["born"]["Z_star"] 

##########################################################################################
## Define the fields
##########################################################################################
dipole_field = system.add_field(ID="dipole", ftype="Rn", dim=3, value=[0,0,-3], mass = 200 * Constants.amu)
# Here the "dipole_field" represents the actual local dipole moment in unit of eA, instead of the soft mode in unit of length (Angstrom)!
# This way we can directly use the built-in dipole-dipole interaction engine. 
# Next, we set the initial local dipole moments in STO to be zero. 
d0 = np.array(dipole_field.get_values().tolist())
d0[:,:,m-1:] *= 0
dipole_field.set_values(jnp.array(d0))

# Define the global strain field
gstrain  = system.add_global_strain(value=jnp.array([-0.012, -0.012, 0.051, 0, 0, 0]), mass = 200 * N * Constants.amu)
# Skip the local strain field since it does not play significant role. 
# lstrain = system.add_field(ID="lstrain", ftype="Rn", dim=6, value=jnp.array([0,0,0,0,0,0]), mass = 200 * Constants.amu)


##########################################################################################
## move fields to multi-GPUs. Comment all if you have only one GPU or running on CPUs. 
##########################################################################################
# Below is for 4 GPUs.
gpu_mesh = DeviceMesh(num_rows=2, num_cols=2)
# Below is for 2 GPUs.
# gpu_mesh = DeviceMesh(jax.devices()[:2], num_rows=2, num_cols=1)
# switch to GPUs
system.move_fields_to_multi_devs(gpu_mesh)

##########################################################################################
## Define the Hamiltonian
## Because the field with ID='dipole' is actually the local dipole moment in unit of eA.
## This is not the convention of the parameters we load, where local dipole is represented by the soft mode in unit of length (Angstrom).
## Therefore, in the following, we need to scale the parameters by the Born effective charge.
##########################################################################################

# On-site energy. 
# "get_self_energy_onsite_on_AmBnLattice" is energy engine implemented in openferro.engine.ferroelectric_superlatt
system.add_self_interaction(ID='self_onsite', field_ID='dipole', 
                            energy_engine=get_self_energy_onsite_on_AmBnLattice(latt, m, n), 
                            parameters=[
                                pto_config["onsite"]["k2"] / Z_pto**2, 
                                pto_config["onsite"]["alpha"] / Z_pto**4, 
                                pto_config["onsite"]["gamma"] / Z_pto**4,
                                sto_config["onsite"]["k2"] / Z_sto**2, 
                                sto_config["onsite"]["alpha"] / Z_sto**4, 
                                sto_config["onsite"]["gamma"] / Z_sto**4
                                ])
# First-shell, short-range dipole-dipole interaction. 
system.add_self_interaction(ID='short_range_1', field_ID='dipole', 
                            energy_engine=get_short_range_1stnn_on_AmBnLattice(latt, m, n), 
                            parameters=[
                                pto_config["short_range"]["j1"] / Z_pto**2,
                                pto_config["short_range"]["j2"] / Z_pto**2,
                                sto_config["short_range"]["j1"] / Z_sto**2,
                                sto_config["short_range"]["j2"] / Z_sto**2
                                ])
# Second-shell, short-range dipole-dipole interaction. 
system.add_self_interaction(ID='short_range_2', field_ID='dipole', 
                            energy_engine=get_short_range_2ednn_on_AmBnLattice(latt, m, n), 
                            parameters=[
                                pto_config["short_range"]["j3"] / Z_pto**2,
                                pto_config["short_range"]["j4"] / Z_pto**2,
                                pto_config["short_range"]["j5"] / Z_pto**2,
                                sto_config["short_range"]["j3"] / Z_sto**2,
                                sto_config["short_range"]["j4"] / Z_sto**2,
                                sto_config["short_range"]["j5"] / Z_sto**2
                                ])
# Third-shell, short-range dipole-dipole interaction. 
system.add_self_interaction(ID='short_range_3', field_ID='dipole', 
                            energy_engine=get_short_range_3rdnn_on_AmBnLattice(latt, m, n), 
                            parameters=[
                                pto_config["short_range"]["j6"] / Z_pto**2,
                                pto_config["short_range"]["j7"] / Z_pto**2,
                                sto_config["short_range"]["j6"] / Z_sto**2,
                                sto_config["short_range"]["j7"] / Z_sto**2,
                                ])
# Homogeneous elastic energy. 
system.add_homo_elastic_interaction(ID='homo_elastic', field_ID='gstrain',
                                    B11=pto_config["elastic"]["B11"], 
                                    B12=pto_config["elastic"]["B12"], 
                                    B44=pto_config["elastic"]["B44"],
                                    )
# Homogeneous strain-dipole interaction. Here we mute the interaction between STO and the global strain. 
system.add_mutual_interaction(ID='homo_strain_dipole', field_1_ID='gstrain', field_2_ID='dipole',
                              energy_engine=get_homo_strain_dipole_interaction_on_AmBnLattice(latt, m, n), 
                              parameters=[
                                pto_config["elastic_dipole"]["B1xx"] / Z_pto**2, 
                                pto_config["elastic_dipole"]["B1yy"] / Z_pto**2, 
                                pto_config["elastic_dipole"]["B4yz"] / Z_pto**2,
                                0,0,0
                                # sto_config["elastic_dipole"]["B1xx"] / Z_sto**2, 
                                # sto_config["elastic_dipole"]["B1yy"] / Z_sto**2, 
                                # sto_config["elastic_dipole"]["B4yz"] / Z_sto**2,
                                ])
# Long-range dipole-dipole interaction. 
system.add_dipole_dipole_interaction(ID='dipole_ewald', field_ID="dipole", prefactor = 1/pto_config["born"]["epsilon_inf"] )
# External electric field. Let's start with zero field. 
dipole_field_interaction = system.add_dipole_efield_interaction(ID='dipole_efield', field_ID="dipole", E=[0,0,0] )

# Inhomogeneous elastic energy.  Does not seem to be significant.
# system.add_inhomo_elastic_interaction(ID='inhomo_elastic', field_ID='lstrain', B11=pto_config["elastic"]["B11"], B12=pto_config["elastic"]["B12"], B44=pto_config["elastic"]["B44"])
# Inhomogeneous strain-dipole interaction. Does not seem to be significant.
# system.add_mutual_interaction(ID='inhomo_strain_dipole', field_1_ID='lstrain', field_2_ID='dipole', 
#                               energy_engine=get_inhomo_strain_dipole_interaction_on_AmBnLattice(latt, m, n), 
#                               parameters=[
#                                 pto_config["elastic_dipole"]["B1xx"] / Z_pto**2, 
#                                 pto_config["elastic_dipole"]["B1yy"] / Z_pto**2, 
#                                 pto_config["elastic_dipole"]["B4yz"] / Z_pto**2,
#                                 sto_config["elastic_dipole"]["B1xx"] / Z_sto**2, 
#                                 sto_config["elastic_dipole"]["B1yy"] / Z_sto**2, 
#                                 sto_config["elastic_dipole"]["B4yz"] / Z_sto**2,
#                                 ])

##########################################################################################
## Simulation setup
##########################################################################################
dt = 0.002 * Constants.ps # times Constants.ps just for better readability.  It's one. 
log_freq = 5000  # logging every 10ps
dump_freq = 50000 # dump the field every 100ps
relax_time = 500 * Constants.ps
drive_time = 4000 * Constants.ps
relax_steps = int(relax_time / dt)
drive_steps = int(drive_time / dt)
temperature = 300
dipole_field.set_integrator('isothermal', dt=dt, temp=temperature, tau=0.1)
gstrain.set_integrator('isothermal', dt=dt, temp=temperature, tau=1, freeze_x=True, freeze_y=True, freeze_z=False)
# lstrain.set_integrator('isothermal', dt=dt, temp=temperature, tau=1)
simulation = SimulationNPTLangevin(system, pressure=hydropressure)

##########################################################################################
## First, relax with zero external field
##########################################################################################
simulation.init_velocity(mode='gaussian', temp=temperature)
simulation.add_thermo_reporter(file='output/relax.log', log_interval=log_freq, 
    global_strain=True, volume=True, potential_energy=True, kinetic_energy=True, temperature=True)
simulation.add_field_reporter(file_prefix='output/relax_field', field_ID="dipole", log_interval=dump_freq, 
    field_average=False, dump_field=True)
simulation.run(relax_steps)

##########################################################################################
## Then, apply external electric field, simulate long-term dynamics
##########################################################################################
external_field = 0.01 * Constants.V_Angstrom # 0.01V/A=1000kV/cm
dipole_field_interaction.set_parameters(jnp.array([0,0, external_field]))
simulation.remove_all_reporters()
simulation.add_thermo_reporter(file='output/drive.log', log_interval=log_freq, 
    global_strain=True, volume=True, potential_energy=True, kinetic_energy=True, temperature=True)
simulation.add_field_reporter(file_prefix='output/drive_field', field_ID="dipole", log_interval=dump_freq, 
    field_average=False, dump_field=True)
simulation.run(drive_steps)
