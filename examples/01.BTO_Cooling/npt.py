import jax
import openferro as of
from openferro.interaction import *
from openferro.simulation import *
from openferro.engine.elastic import *
from openferro.engine.ferroelectric import *
from matplotlib import pyplot as plt
import json
from time import time as timer

##########################################################################################
## Define the lattice 
##########################################################################################
L = 12
hydropres =  -4.8e4
config = json.load(open('BaTiO3.json'))
latt_vecs = jnp.eye(3) * config['lattice']['a1']
latt = of.SimpleCubic3D(L, L, L, latt_vecs[0], latt_vecs[1], latt_vecs[2])
bto = of.System(latt)
dipole2polar = config['born']['Z_star'] / latt.unit_volume * 16.0217646  # eA -> C/m^2

##########################################################################################
## Define the fields
##########################################################################################
dipole_field = bto.add_field(name="dipole", ftype="Rn", dim=3, value=0.1, mass = 1.0)
lstrain_field = bto.add_field(name="lstrain", ftype="LocalStrain3D", value=0.0, mass = 40)
gstrain  = bto.add_global_strain(value=jnp.array([0.01,0.01,0.01,0,0,0]), mass = 200.0 *  L**3)

##########################################################################################
## Define the Hamiltonian
##########################################################################################
bto.add_dipole_onsite_interaction('self_onsite', field_name="dipole", K2=config["onsite"]["k2"], alpha=config["onsite"]["alpha"], gamma=config["onsite"]["gamma"])
bto.add_dipole_interaction_1st_shell('short_range_1', field_name="dipole", j1=config["short_range"]["j1"], j2=config["short_range"]["j2"])
bto.add_dipole_interaction_2nd_shell('short_range_2', field_name="dipole", j3=config["short_range"]["j3"], j4=config["short_range"]["j4"], j5=config["short_range"]["j5"])
bto.add_dipole_interaction_3rd_shell('short_range_3', field_name="dipole", j6=config["short_range"]["j6"], j7=config["short_range"]["j7"])
bto.add_dipole_dipole_interaction('dipole_ewald', field_name="dipole", prefactor = config["born"]["Z_star"]**2 / config["born"]["epsilon_inf"] )
bto.add_homo_elastic_interaction('homo_elastic', field_name="gstrain", B11=config["elastic"]["B11"], B12=config["elastic"]["B12"], B44=config["elastic"]["B44"])
bto.add_homo_strain_dipole_interaction('homo_strain_dipole', field_name1="gstrain", field_name2="dipole", B1xx=config["elastic_dipole"]["B1xx"], B1yy=config["elastic_dipole"]["B1yy"], B4yz=config["elastic_dipole"]["B4yz"])
bto.add_inhomo_elastic_interaction('inhomo_elastic', field_name="lstrain", B11=config["elastic"]["B11"], B12=config["elastic"]["B12"], B44=config["elastic"]["B44"])
bto.add_inhomo_strain_dipole_interaction('inhomo_strain_dipole', field_name1="lstrain", field_name2="dipole", B1xx=config["elastic_dipole"]["B1xx"], B1yy=config["elastic_dipole"]["B1yy"], B4yz=config["elastic_dipole"]["B4yz"])

##########################################################################################
## Structure relaxation
##########################################################################################
dipole_field.set_integrator('optimization', dt=0.01)
gstrain.set_integrator('optimization', dt=0.01)
lstrain_field.set_integrator('optimization', dt=0.01)
minimizer = MDMinimize(bto, max_iter=1000, tol=1e-5)
minimizer.run(variable_cell=True, pressure=hydropres)
 
##########################################################################################
## NPT cooling simulation
##########################################################################################
log_freq = 100
total_time = 1000
dt = 0.002
relax_steps = int(50/dt)
total_steps = int(total_time / dt)
niters = total_steps // log_freq


temp_list = np.array([140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320, 350, 400]).astype(int)
temp_list = np.flip(temp_list)
dipole_field.set_integrator('isothermal', dt=dt, temp=temp_list[0], tau=0.1)
gstrain.set_integrator('isothermal', dt=dt, temp=temp_list[0], tau=1)
lstrain_field.set_integrator('isothermal', dt=dt, temp=temp_list[0], tau=1)
simulation = SimulationNPTLangevin(bto, pressure=-4.8e4)
simulation.init_velocity(mode='gaussian', temp=temp_list[0])
simulation.run(relax_steps)

for temperature in temp_list:
    print('T={}K'.format(temperature))
    dipole_field.set_integrator('isothermal', dt=dt, temp=temperature, tau=0.1)
    gstrain.set_integrator('isothermal', dt=dt, temp=temperature, tau=1)
    lstrain_field.set_integrator('isothermal', dt=dt, temp=temperature, tau=1)
    simulation = SimulationNPTLangevin(bto, pressure=-4.8e4 )
    average_field = []
    global_strain = []
    for ii in range(niters):
        simulation.run(log_freq)
        average_field.append(bto.get_field_by_name('dipole').get_values().mean((0,1,2)))
        global_strain.append(bto.get_field_by_name("gstrain").get_values().flatten())
        excess_pres = bto.calc_excess_stress()
        if ii % 10 == 0:
            print('=================T={}K, iter={}======================='.format(temperature, ii))
            print('temperature:', bto.calc_temp_by_name('dipole'))
            print('Polarization : {}C/m^2'.format(average_field[-1] * dipole2polar ))
            print('global strain: {} '.format(global_strain[-1]))
            print('inhomogeneous strain std:',  lstrain_field.get_values().std((0,1,2)))
            print('excessive pressure:{}'.format(excess_pres))
    average_field = jnp.array(average_field)
    global_strain = jnp.array(global_strain)
    jnp.save('output/field_history_T{}.npy'.format(temperature) , average_field)
    jnp.save('output/strain_history_T{}.npy'.format(temperature) , global_strain)
