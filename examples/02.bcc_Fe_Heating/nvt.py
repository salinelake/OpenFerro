import jax
import openferro as of
from openferro.interaction import *
from openferro.simulation import *
from openferro.engine import *
from openferro.units import Constants
from openferro.engine.magnetic import *
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
temp_list = [10,  100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500]
simulation = SimulationNVTLangevin(Fe_bcc)
for temp in temp_list:
    simulation.remove_all_reporters()
    simulation.add_thermo_reporter(file='thermo_{}K.log'.format(temp), log_interval=100, global_strain=False, volume=True, potential_energy=True, kinetic_energy=True, temperature=True)
    simulation.add_field_reporter(file_prefix="spin_{}K".format(temp), field_ID="spin", log_interval=100, field_average=True, dump_field=False)
    logging.info("Temp={}K, NVT Simulation".format(temp))
    spin_field.set_integrator('isothermal', dt=dt, temp=temp, alpha=0.5)
    simulation.run(10000)
 