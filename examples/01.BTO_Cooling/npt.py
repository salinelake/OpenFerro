import jax
import openferro as of
from openferro.interaction import *
from openferro.simulation import *
from openferro.engine import *
from matplotlib import pyplot as plt
import json
from time import time as timer


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
dipole_field = bto.add_field(name="dipole", ftype="Rn", dim=3, value=0.1, mass = 1.0)
# lstrain_field = bto.add_field(name="lstrain", ftype="LocalStrain3D", value=0.0, mass = 40)
gstrain  = bto.add_global_strain(value=jnp.array([0.01,0.01,0.01,0,0,0]), mass = 200.0 *  L**3)

##########################################################################################
## define Hamiltonian
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
## structure relaxation
##########################################################################################
minimizer = MDMinimize(bto, max_iter=1000, tol=1e-5, dt=0.01)
minimizer.minimize(variable_cell=True, pressure=hydropres)
equilibrium_field = bto.get_field_by_name("dipole").get_values().copy()
avg_field = equilibrium_field.mean((0,1,2))
dipole2polar = config['born']['Z_star'] / latt.unit_volume * 16.0217646  # eA -> C/m^2
polarization = avg_field * dipole2polar  # C/m^2
strain4 = - config['elastic_dipole']['B4yz'] / config['elastic']['B44'] * avg_field[1] * avg_field[2]
strain5 = - config['elastic_dipole']['B4yz'] / config['elastic']['B44'] * avg_field[0] * avg_field[2]
strain6 = - config['elastic_dipole']['B4yz'] / config['elastic']['B44'] * avg_field[0] * avg_field[1]
strain1 = - 0.5 * (config['elastic_dipole']['B1xx']+2*config['elastic_dipole']['B1yy']) * avg_field[0]**2
strain1 -= hydropres * Constants.bar * latt.unit_volume
strain1 /= (config['elastic']['B11']+2*config['elastic']['B12'])

print('max force after optimization',jnp.abs(bto.get_field_by_name("dipole").get_force()).max())
print('numerical   global strain:', bto.get_field_by_name("gstrain").get_values())
print('theoretical global strain=({:.4f},**,**,{:.4f},{:.4f},{:.4f})'.format(strain1, strain4, strain5, strain6))
print('spontaneous polarization = {} C/m^2'.format(polarization))
print('|polarization|={}C/m^2'.format((polarization**2).sum()**0.5))

##########################################################################################
## NPT cooling simulation
##########################################################################################

log_freq = 100
total_time = 500
dt = 0.002
relax_steps = int(50/dt)
total_steps = int(total_time / dt)
niters = total_steps // log_freq
field_history = []
gs_history = []


temp_list = np.array([140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320, 350, 400]).astype(int)
temp_list = np.flip(temp_list)
simulation = SimulationNPTLangevin(bto, dt=dt, temperature=temp_list[0], pressure=-4.8e4, tau=0.1, tauP = 1)
simulation.init_velocity(mode='gaussian')
t1 = timer()
simulation.step(relax_steps)
t2 = timer()
print('relaxation (50ps) time:', t2-t1)

for temperature in temp_list:
    print('T={}K'.format(temperature))
    simulation = SimulationNPTLangevin(bto, dt=dt, temperature=temperature, pressure=-4.8e4, tau=0.1, tauP = 1)
    average_field = []
    global_strain = []
    for ii in range(niters):
        simulation.step(log_freq)
        average_field.append(bto.get_field_by_name('dipole').get_values().mean((0,1,2)))
        global_strain.append(bto.get_field_by_name("gstrain").get_values().flatten())
        excess_pres = bto.calc_excess_stress()
        if ii % 10 == 0:
            print('=================T={}K, iter={}======================='.format(temperature, ii))
            print('temperature:', bto.calc_temp_by_name('dipole'))
            print('Polarization : {}C/m^2'.format(average_field[-1] * dipole2polar ))
            print('global strain: {} '.format(global_strain[-1]))
            print('excessive pressure:{}'.format(excess_pres))
    average_field = jnp.array(average_field)
    global_strain = jnp.array(global_strain)

    ## plot and save the history of average field and global strain
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].plot(average_field[:,0])
    ax[0].plot(average_field[:,1])
    ax[0].plot(average_field[:,2])
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('Field ')
    ax[1].plot(global_strain[:,0])
    ax[1].plot(global_strain[:,1])
    ax[1].plot(global_strain[:,2])
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('Strain')
    plt.savefig('output/history_T{}.png'.format(temperature))
    jnp.save('output/field_history_T{}.npy'.format(temperature) , average_field)
    jnp.save('output/strain_history_T{}.npy'.format(temperature) , global_strain)
