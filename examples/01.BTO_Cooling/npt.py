import jax
import openferro as of
from openferro.interaction import *
from openferro.simulation import *
from openferro.engine.elastic import *
from openferro.engine.ferroelectric import *
from openferro.units import Constants
import json
import logging
import os
os.makedirs('output', exist_ok=True)
logging.basicConfig(level=logging.INFO, filename='simulation.log')
##########################################################################################
## Define the lattice 
##########################################################################################
L = 20
hydropres =  -4.8e4
config = json.load(open('BaTiO3.json'))
latt_vecs = jnp.eye(3) * config['lattice']['a1']
latt = of.SimpleCubic3D(L, L, L, latt_vecs[0], latt_vecs[1], latt_vecs[2])
bto = of.System(latt)

##########################################################################################
## Define the fields
##########################################################################################
dipole_field = bto.add_field(ID="dipole", ftype="Rn", dim=3, value=0.0, mass = 200 * Constants.amu)
lstrain_field = bto.add_field(ID="lstrain", ftype="LocalStrain3D", value=0.0, mass = 200 * Constants.amu)
gstrain  = bto.add_global_strain(value=jnp.array([0.01,0.01,0.01,0,0,0]), mass = 200 * Constants.amu * L**3)

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
## Structure relaxation
##########################################################################################
logging.info('Structure relaxation')
dipole_field.set_integrator('optimization', dt=0.0001)
gstrain.set_integrator('optimization', dt=0.0001)
lstrain_field.set_integrator('optimization', dt=0.0001)
minimizer = MDMinimize(bto, max_iter=1000, tol=1e-5)
minimizer.add_thermo_reporter(file='output/optimization.log', log_interval=10, global_strain=True, volume=True, potential_energy=True, kinetic_energy=False, temperature=False)
minimizer.run(variable_cell=True, pressure=hydropres)

##########################################################################################
## NPT cooling simulation
##########################################################################################
log_freq = 100
total_time = 100
dt = 0.002
relax_steps = int(10/dt)
sampling_steps = int(total_time / dt)

temp_list = np.array([400, 350, 320, 310, 300, 290, 280, 270, 260, 250, 240, 230, 220, 210, 200, 190, 180, 170, 160, 150, 140]).astype(int)
simulation = SimulationNPTLangevin(bto, pressure=hydropres)
simulation.init_velocity(mode='gaussian', temp=temp_list[0])

for temperature in temp_list:
    dipole_field.set_integrator('isothermal', dt=dt, temp=temperature, tau=0.1)
    gstrain.set_integrator('isothermal', dt=dt, temp=temperature, tau=1)
    lstrain_field.set_integrator('isothermal', dt=dt, temp=temperature, tau=1)
    ## equilibration
    logging.info('T={}K, NPT Equlibration'.format(temperature))
    simulation.remove_all_reporters()
    simulation.run(relax_steps)
    ## sampling
    logging.info('T={}K, NPT Sampling'.format(temperature))
    simulation.add_thermo_reporter(file='output/thermo_{}K.log'.format(temperature), log_interval=log_freq, 
        global_strain=True, excess_stress=True, volume=True, potential_energy=True, kinetic_energy=True, temperature=True)
    simulation.add_field_reporter(file_prefix='output/field_{}K'.format(temperature), field_ID="dipole", log_interval=log_freq, 
        field_average=True, dump_field=False)
    simulation.run(sampling_steps)