import jax
import openferro as of
from openferro.interaction import *
from openferro.simulation import *
from openferro.engine import *
from openferro.parallelism import DeviceMesh
from openferro.units import Constants
from openferro.engine.magnetic import *
from time import time as timer
from matplotlib import pyplot as plt
import logging
##########################################################################################
## Define the lattice 
##########################################################################################
L = 20
N = L**3
latt = of.SimpleCubic3D(L, L, L)
Fe_bcc = of.System(latt)
logging.basicConfig(level=logging.INFO, filename="simulation.log")
##########################################################################################
## Define the fields
##########################################################################################
Ms = 1.5   #in muB
spin_field = Fe_bcc.add_field(name="spin", ftype="SO3", value=jnp.array([0,0,1]))
spin_field.set_magnitude(Ms)

##########################################################################################
## Define the Hamiltonian
##########################################################################################
J1 = 6.72e-21 # Joule/link, without double counting ij and ji, from [https://vampire.york.ac.uk/tutorials/simulation/curie-temperature/]. Theoretical Tc is around 700K. 
J1  = J1 / 2 * Constants.Joule / Ms**2  # convert to our convention and in unit of (eV / muB^2)
Fe_bcc.add_isotropic_exchange_interaction_1st_shell(name="exchange_1st_shell", field_name="spin", coupling=J1)

##########################################################################################
## NVT simulation
##########################################################################################

dt = 0.0002 # in ps
temp_list = np.linspace(50, 900, 18)

for temp in temp_list:
    spin_field.set_integrator('isothermal', dt=dt, temp=temp, alpha=1)
    simulation = SimulationNVTLangevin(Fe_bcc )
    logging.info(f"Relaxing at {temp}K")
    simulation.run(2000, profile=False)
    
    logging.info("Sampling...")
    M_avg = []
    for i in range(100):
        M_avg.append(spin_field.get_values().mean(axis=[0,1,2]))
        simulation.run(100, profile=False)
        
    M_avg = jnp.array(M_avg) # shape=(100, 3)
    logging.info("Temp={}K, M_avg/Ms = {}".format(temp, jnp.linalg.norm(M_avg, axis=-1).mean()/Ms))
    jnp.save(f"data/Mavg_{temp}.npy", M_avg)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(np.arange(M_avg.shape[0])*dt, jnp.linalg.norm(M_avg, axis=-1) /Ms)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("M_avg/Ms")
    ax.set_title(f"{temp}K")
    plt.tight_layout()
    plt.savefig(f"data/Mavg_{temp}.png")
    plt.close()
