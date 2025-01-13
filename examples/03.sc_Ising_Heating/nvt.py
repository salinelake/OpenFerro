import jax
import openferro as of
from openferro.interaction import *
from openferro.simulation import *
from openferro.engine import *
from openferro.units import Constants
from openferro.engine.magnetic import *
import logging
import os

os.makedirs("./output", exist_ok=True)
logging.basicConfig(level=logging.INFO, filename="simulation.log")
##########################################################################################
## Define the lattice 
##########################################################################################
L = 20
N = L**3
latt = of.SimpleCubic3D(L, L, L)
Fe_bcc = of.System(latt)

##########################################################################################
## Define the fields
##########################################################################################
Ms = 1.5   #in muB
spin_field = Fe_bcc.add_field(ID="spin", ftype="SO3", value=jnp.array([0,0,1]))
spin_field.set_magnitude(Ms)

##########################################################################################
## Define the Hamiltonian
##########################################################################################
J1 = 6.72e-21 # Joule/link, without double counting ij and ji, from [https://vampire.york.ac.uk/tutorials/simulation/curie-temperature/]. Theoretical Tc is around 700K. 
J1  = J1 / 2 * Constants.Joule / Ms**2  # convert to our convention and in unit of (eV / muB^2)
Fe_bcc.add_isotropic_exchange_interaction_1st_shell(ID="exchange_1st_shell", field_ID="spin", coupling=J1)

##########################################################################################
## NVT simulation
##########################################################################################
dt = 0.0002 # in ps
temp_list = np.linspace(50, 900, 18)
simulation = SimulationNVTLangevin(Fe_bcc )
for temp in temp_list:
    spin_field.set_integrator('isothermal', dt=dt, temp=temp, alpha=1)
    ## equlibration
    logging.info("Temp={}K, NVT Equilibration".format(temp))
    simulation.remove_all_reporters()
    simulation.run(5000)
    ## sampling
    simulation.add_thermo_reporter(file='./output/thermo_{}K.log'.format(temp), log_interval=100, global_strain=False, volume=True, potential_energy=True, kinetic_energy=True, temperature=True)
    simulation.add_field_reporter(file_prefix="./output/spin_{}K".format(temp), field_ID="spin", log_interval=100, field_average=True, dump_field=False)
    logging.info("Temp={}K, NVT Sampling".format(temp))
    simulation.run(20000)
