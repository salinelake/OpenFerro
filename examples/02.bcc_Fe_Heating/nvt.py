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

logging.basicConfig(level=logging.INFO, filename="simulation.log")
##########################################################################################
## Define the lattice 
##########################################################################################
L = 20
N = L**3
latt = of.BodyCenteredCubic3D(L, L, L)
Fe_bcc = of.System(latt)

##########################################################################################
## Define the fields
##########################################################################################
Ms = 2.23   #in muB
spin_field = Fe_bcc.add_field(ID="spin", ftype="SO3", value=jnp.array([0,0,1]))
spin_field.set_magnitude(Ms)

##########################################################################################
## Define the Hamiltonian
##########################################################################################
## getting these parameters from PRL 95, 087207 (2005)
couplings = jnp.array([1.33767484769984, 0.75703576545650, -0.05975437643846, -0.08819834160658])  # in mRy, SPR-KKR convention
J1, J2, J3, J4 = couplings * Constants.mRy / Ms**2  # convert to our convention (eV / muB^2)
Fe_bcc.add_isotropic_exchange_interaction_1st_shell(ID="exchange_1st_shell", field_ID="spin", coupling=J1)
Fe_bcc.add_isotropic_exchange_interaction_2nd_shell(ID="exchange_2nd_shell", field_ID="spin", coupling=J2)
Fe_bcc.add_isotropic_exchange_interaction_3rd_shell(ID="exchange_3rd_shell", field_ID="spin", coupling=J3)
Fe_bcc.add_isotropic_exchange_interaction_4th_shell(ID="exchange_4th_shell", field_ID="spin", coupling=J4)
# Fe_bcc.add_cubic_anisotropy_interaction(ID="cubic_anisotropy", field_ID="spin", K1=K1, K2=K2)

##########################################################################################
## NPT simulation
##########################################################################################
dt = 0.0002
temp_list = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500]
simulation = SimulationNVTLangevin(Fe_bcc )

for temp in temp_list:
    logging.info("Temp={}K, NVT Simulation".format(temp))
    spin_field.set_integrator('isothermal', dt=dt, temp=temp, alpha=0.5)
    spin_traj = []
    M_avg = []
    pot_energy = []
    for i in range(110):
        spin_traj.append(spin_field.get_values()[0, 0, 0, :])
        pot_energy.append(Fe_bcc.calc_total_potential_energy())
        M_avg.append(spin_field.get_values().mean(axis=[0,1,2]))
        simulation.run(100, profile=False)
        
    spin_traj = jnp.array(spin_traj) # shape=(100, 3)
    M_avg = jnp.array(M_avg) # shape=(100, 3)
    theta = jnp.arccos( spin_traj[:, 2]  / Ms )
    pot_energy = jnp.array(pot_energy)
    logging.info("Temp={}K, M_avg = {} uB".format(temp, jnp.linalg.norm(M_avg, axis=-1)[10:].mean()))
    jnp.save(f"Mavg_{temp}.npy", M_avg)
    # fig, ax = plt.subplots(1,5, figsize=(15, 3))
    # ax[0].plot(spin_traj[:, 0], spin_traj[:, 1]  )
    # ax[0].set_aspect("equal")
    # ax[0].set_xlim(-Ms, Ms)
    # ax[0].set_ylim(-Ms, Ms)
    # ax[1].plot(np.arange(len(spin_traj))*dt, jnp.linalg.norm(spin_traj, axis=-1)/Ms )
    # ax[1].set_xlabel("time (ps)")
    # ax[1].set_ylabel("|M| / Ms")
    # ax[1].axhline(1, c="k", ls="--")
    # ax[2].plot( theta /jnp.pi * 180, marker="o", ms=1.5 )
    # ax[2].set_xlabel("time (ps)")
    # ax[2].set_ylabel("theta (deg)")
    # ax[3].plot(np.arange(len(spin_traj))*dt, M_avg[:, 0]/Ms, marker="o", ms=1.5, label="Mx" )
    # ax[3].plot(np.arange(len(spin_traj))*dt, M_avg[:, 1]/Ms, marker="o", ms=1.5, label="My" )
    # ax[3].plot(np.arange(len(spin_traj))*dt, M_avg[:, 2]/Ms, marker="o", ms=1.5, label="Mz" )
    # ax[3].plot(np.arange(len(spin_traj))*dt, jnp.linalg.norm(M_avg, axis=-1)/Ms, marker="o", ms=1.5, label="|M|" )
    # ax[3].set_xlabel("time (ps)")
    # ax[3].set_ylabel("Avg M / Ms")
    # ax[3].legend()
    # ax[4].plot(np.arange(len(spin_traj))*dt, pot_energy, marker="o", ms=1.5)
    # ax[4].set_xlabel("time (ps)")
    # ax[4].set_ylabel("potential energy (eV)")
    # plt.tight_layout()
    # plt.savefig("spin_traj.png", dpi=200)
 