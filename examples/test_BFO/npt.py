import json
import logging
import os
import numpy as np
import openferro as of
from openferro.interaction import *
from openferro.simulation import *
import openferro.engine.multiferroic as mf
from openferro.units import Constants

os.makedirs('output', exist_ok=True)
logging.basicConfig(level=logging.INFO, filename='simulation.log')
##########################################################################################
## Define the lattice 
##########################################################################################
L = 12
config = json.load(open('BFO_internal.json'))
latt_vecs = jnp.eye(3) * config['lattice']['a1']
latt = of.SimpleCubic3D(L, L, L, latt_vecs[0], latt_vecs[1], latt_vecs[2])
BFO = of.System(latt)

##########################################################################################
## Define the fields
##########################################################################################
dipole_field = BFO.add_field(ID="dipole", ftype="Rn", dim=3, value=jnp.array([0.1,0.1,0.1]), mass = 200 * Constants.amu)
lstrain_field = BFO.add_field(ID="lstrain", ftype="LocalStrain3D", value=0.0, mass = 200 * Constants.amu)
gstrain = BFO.add_global_strain(value=jnp.array([0.01,0.01,0.01,0,0,0]), mass = 200 * Constants.amu * L**3)
## AFD initialization
afd_field = BFO.add_field(ID="AFD", ftype="Rn", dim=3, value=jnp.array([0.0,0.0,0.0]), mass = 200 * Constants.amu)
afd_init = jnp.ones((L,L,L,3)) * 0.001
i, j, k = np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing="ij")
sign_array = jnp.array((-1) ** (i + j + k))
afd_field.set_values(afd_init * sign_array[...,None])
# spin initialization
spin_field = BFO.add_field(ID="spin", ftype="SO3", value=jnp.array([0,0,1]))
spin_field.set_magnitude(4)  # set the magnitude of the spin field to 4 uB
spin_init = jnp.ones((L,L,L,3))
spin_field.set_values(spin_init * sign_array[...,None])
spin_field.normalize()

##########################################################################################
## Define the Hamiltonian: the electric dipolar part
##########################################################################################
BFO.add_dipole_onsite_interaction('dp_onsite', field_ID="dipole", K2=config["dipole_onsite"]["k2"], alpha=config["dipole_onsite"]["alpha"], gamma=config["dipole_onsite"]["gamma"])
BFO.add_dipole_interaction_1st_shell('dp_sr_1nn', field_ID="dipole", j1=config["dipole_short_range"]["j1"], j2=config["dipole_short_range"]["j2"])
BFO.add_dipole_interaction_2nd_shell('dp_sr_2nn', field_ID="dipole", j3=config["dipole_short_range"]["j3"], j4=config["dipole_short_range"]["j4"], j5=config["dipole_short_range"]["j5"])
BFO.add_dipole_interaction_3rd_shell('dp_sr_3nn', field_ID="dipole", j6=config["dipole_short_range"]["j6"], j7=config["dipole_short_range"]["j7"])
BFO.add_dipole_dipole_interaction('dp_ewald', field_ID="dipole", prefactor = config["born"]["Z_star"]**2 / config["born"]["epsilon_inf"] )
BFO.add_homo_elastic_interaction('homo_elastic', field_ID="gstrain", B11=config["elastic"]["B11"], B12=config["elastic"]["B12"], B44=config["elastic"]["B44"])
BFO.add_homo_strain_dipole_interaction('homo_strain_dipole', field_1_ID="gstrain", field_2_ID="dipole", B1xx=config["elastic_dipole"]["B1xx"], B1yy=config["elastic_dipole"]["B1yy"], B4yz=config["elastic_dipole"]["B4yz"])
BFO.add_inhomo_elastic_interaction('inhomo_elastic', field_ID="lstrain", B11=config["elastic"]["B11"], B12=config["elastic"]["B12"], B44=config["elastic"]["B44"])
BFO.add_inhomo_strain_dipole_interaction('inhomo_strain_dipole', field_1_ID="lstrain", field_2_ID="dipole", B1xx=config["elastic_dipole"]["B1xx"], B1yy=config["elastic_dipole"]["B1yy"], B4yz=config["elastic_dipole"]["B4yz"])

##########################################################################################
## Define the Hamiltonian: the AFD part
##########################################################################################
BFO.add_dipole_onsite_interaction('afd_onsite', field_ID="AFD", K2=config["AFD_onsite"]["k2"], alpha=config["AFD_onsite"]["alpha"], gamma=config["AFD_onsite"]["gamma"])
BFO.add_dipole_interaction_1st_shell('afd_sr_1nn_quad', field_ID="AFD", j1= config["AFD_short_range"]["k1"], j2= config["AFD_short_range"]["k2"])
BFO.add_self_interaction(ID='afd_sr_1nn_quartic', field_ID='AFD', energy_engine=mf.short_range_1stnn_uniaxial_quartic, parameters=[config["AFD_short_range"]["k_prime"]])
BFO.add_homo_strain_dipole_interaction('homo_strain_afd', field_1_ID="gstrain", field_2_ID="AFD", B1xx=config["elastic_AFD"]["B1xx"], B1yy=config["elastic_AFD"]["B1yy"], B4yz=config["elastic_AFD"]["B4yz"])
BFO.add_inhomo_strain_dipole_interaction('inhomo_strain_afd', field_1_ID="lstrain", field_2_ID="AFD", B1xx=config["elastic_AFD"]["B1xx"], B1yy=config["elastic_AFD"]["B1yy"], B4yz=config["elastic_AFD"]["B4yz"])
BFO.add_mutual_interaction(ID='afd_dipole_trilinear', field_1_ID='dipole', field_2_ID='AFD', energy_engine=mf.short_range_1stnn_trilinear_two_sublattices, parameters=[config["dipole_AFD_trilinear"]["D"]])
BFO.add_mutual_interaction(ID='afd_dipole_biquad', field_1_ID='dipole', field_2_ID='AFD', energy_engine=mf.short_range_1stnn_biquadratic_iiii_two_sublattices, parameters=[config["dipole_AFD_biquadratic"]["Exxxx"], config["dipole_AFD_biquadratic"]["Exxyy"], config["dipole_AFD_biquadratic"]["Exyxy"]])

##########################################################################################
## Define the Hamiltonian: the magnetic part
##########################################################################################
## the built-in dipole-dipole interaction is for electric dipole.  
## multiply by epsilon_0*mu_0 to get the magnetic dipole-dipole interaction.
BFO.add_dipole_dipole_interaction('spin_ewald', field_ID="spin", prefactor = Constants.epsilon0 * Constants.mu0  )
BFO.add_dipole_interaction_1st_shell('spin_sr_1nn', field_ID="spin", j1=config["spin_short_range"]["j1"], j2=config["spin_short_range"]["j2"])
BFO.add_dipole_interaction_2nd_shell('spin_sr_2nn', field_ID="spin", j3=config["spin_short_range"]["j3"], j4=config["spin_short_range"]["j4"], j5=config["spin_short_range"]["j5"])
BFO.add_dipole_interaction_3rd_shell('spin_sr_3nn', field_ID="spin", j6=config["spin_short_range"]["j6"], j7=config["spin_short_range"]["j7"])
BFO.add_mutual_interaction('dipole_spin_sr', field_1_ID='dipole', field_2_ID='spin', energy_engine=mf.short_range_biquadratic_ijii_two_sublattices, 
    parameters=[config["spin_dipole"]["E1xxxx"], config["spin_dipole"]["E1yyxx"], config["spin_dipole"]["E1xyxy"], 
                config["spin_dipole"]["E2xxxx"], config["spin_dipole"]["E2yyxx"], config["spin_dipole"]["E2xyxy"], 
                config["spin_dipole"]["E3xxxx"], config["spin_dipole"]["E3yyxx"], config["spin_dipole"]["E3xyxy"]])
BFO.add_mutual_interaction('afd_spin', field_1_ID='AFD', field_2_ID='spin', energy_engine=mf.short_range_biquadratic_ijii, 
    parameters=[config["spin_AFD"]["F1xxxx"], config["spin_AFD"]["F1yyxx"], config["spin_AFD"]["F1xyxy"], 
                config["spin_AFD"]["F2xxxx"], config["spin_AFD"]["F2yyxx"], config["spin_AFD"]["F2xyxy"], 
                config["spin_AFD"]["F3xxxx"], config["spin_AFD"]["F3yyxx"], config["spin_AFD"]["F3xyxy"]])
BFO.add_mutual_interaction('homo_strain_spin', field_1_ID='gstrain', field_2_ID='spin', energy_engine=mf.homo_strain_spin_interaction, 
    parameters=[config["spin_elastic"]["G1xxxx"], config["spin_elastic"]["G1yyxx"], config["spin_elastic"]["G1xyxy"], 
                config["spin_elastic"]["G2xxxx"], config["spin_elastic"]["G2yyxx"], config["spin_elastic"]["G2xyxy"], 
                config["spin_elastic"]["G3xxxx"], config["spin_elastic"]["G3yyxx"], config["spin_elastic"]["G3xyxy"]])
BFO.add_mutual_interaction('inhomo_strain_spin', field_1_ID='lstrain', field_2_ID='spin', energy_engine=mf.get_inhomo_strain_spin_interaction(), 
    parameters=[config["spin_elastic"]["G1xxxx"], config["spin_elastic"]["G1yyxx"], config["spin_elastic"]["G1xyxy"], 
                config["spin_elastic"]["G2xxxx"], config["spin_elastic"]["G2yyxx"], config["spin_elastic"]["G2xyxy"], 
                config["spin_elastic"]["G3xxxx"], config["spin_elastic"]["G3yyxx"], config["spin_elastic"]["G3xyxy"]])
BFO.add_mutual_interaction('DM_spin_AFD', field_1_ID='AFD', field_2_ID='spin', energy_engine=mf.DM_AFD_1stnn,  parameters=[config["spin_DM"]["L1"]])

##########################################################################################
## Structure relaxation
##########################################################################################
dt_optimization = 0.00001
logging.info('Structure relaxation')
dipole_field.set_integrator('optimization', dt=dt_optimization)
gstrain.set_integrator('optimization', dt=dt_optimization)
afd_field.set_integrator('optimization', dt=dt_optimization)
lstrain_field.set_integrator('optimization', dt=dt_optimization)
spin_field.set_integrator('optimization', dt=dt_optimization, alpha=1)
minimizer = MDMinimize(BFO, max_iter=1001, tol=1e-5)
minimizer.add_thermo_reporter(file='output/optimization.log', log_interval=100, global_strain=True, volume=True, potential_energy=True, kinetic_energy=False, temperature=False)
minimizer.add_field_reporter(file_prefix='output/AFD_relax', field_ID="AFD", log_interval=500, field_average=True, dump_field=True)
minimizer.add_field_reporter(file_prefix='output/dipole_relax', field_ID="dipole", log_interval=500, field_average=True, dump_field=True)
minimizer.add_field_reporter(file_prefix='output/spin_relax', field_ID="spin", log_interval=500, field_average=True, dump_field=True)
minimizer.run(variable_cell=True, pressure=0)
print('AFD mode:', jnp.mean(afd_field.get_values()*sign_array[...,None], axis=(0,1,2)))
print('Dipole mode:', jnp.mean(dipole_field.get_values(), axis=(0,1,2)))
print('Spin mode:', jnp.mean(spin_field.get_values()*sign_array[...,None], axis=(0,1,2)))

##########################################################################################
## NPT Heating simulation
##########################################################################################
log_freq = 100
dump_freq = 10000
equlibration_time = 10 * Constants.ps
sampling_time = 10 * Constants.ps
dt = 0.0004 * Constants.ps   ## dt>=0.001ps will leads to inconvergence of SIB integrator.  ~0.036s/step.  50000steps=1800s=30min
equlibration_steps = int(equlibration_time / dt)
sampling_steps = int(sampling_time / dt)

temp_list = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]).astype(int)
simulation = SimulationNPTLangevin(BFO, pressure=0)
simulation.init_velocity(mode='gaussian', temp=temp_list[0])

for temperature in temp_list:
    dipole_field.set_integrator('isothermal', dt=dt, temp=temperature, tau=0.1)
    gstrain.set_integrator('isothermal', dt=dt, temp=temperature, tau=1)
    lstrain_field.set_integrator('isothermal', dt=dt, temp=temperature, tau=1)
    afd_field.set_integrator('isothermal', dt=dt, temp=temperature, tau=1)
    spin_field.set_integrator('isothermal', dt=dt, temp=temperature, alpha=1)
    ## equilibration
    logging.info('T={}K, NPT Equlibration'.format(temperature))
    simulation.remove_all_reporters()
    simulation.run(equlibration_steps)
    # sampling
    logging.info('T={}K, NPT Sampling'.format(temperature))
    simulation.add_thermo_reporter(file='output/thermo_{}K.log'.format(temperature), log_interval=log_freq, 
        global_strain=True, excess_stress=True, volume=True, potential_energy=True, kinetic_energy=True, temperature=True)
    simulation.add_field_reporter(file_prefix='output/field_{}K'.format(temperature), field_ID="dipole", log_interval=log_freq, 
        field_average=True, dump_field=False)
    simulation.add_field_reporter(file_prefix='output/AFD_{}K'.format(temperature), field_ID="AFD", log_interval=dump_freq, 
        field_average=False, dump_field=True)
    simulation.add_field_reporter(file_prefix='output/spin_{}K'.format(temperature), field_ID="spin", log_interval=dump_freq, 
        field_average=False, dump_field=True)
    simulation.run(sampling_steps)
    AFD = jnp.mean(afd_field.get_values()*sign_array[...,None], axis=(0,1,2))
    dipole = jnp.mean(dipole_field.get_values(), axis=(0,1,2))
    spin = jnp.mean(spin_field.get_values()*sign_array[...,None], axis=(0,1,2))
    logging.info('T={}K, AFD mode: {}, dipole mode: {}, spin mode: {}'.format(temperature, AFD, dipole, spin))