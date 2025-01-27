import json
from openferro.units import AtomicUnits_to_InternalUnits as au

config = json.load(open('BFO_atomic_units.json'))
config['dipole_onsite']['k2'] *= au.energy / au.length**2
config['dipole_onsite']['alpha'] *= au.energy / au.length**4
config['dipole_onsite']['gamma'] *= au.energy / au.length**4

for key in config['dipole_short_range']:
    config['dipole_short_range'][key] *= au.energy / au.length**2

for key in config['elastic']:
    config['elastic'][key] *= au.energy  

for key in config['elastic_dipole']:
    config['elastic_dipole'][key] *= au.energy / au.length**2

config['AFD_onsite']['k2'] *= au.energy  
config['AFD_onsite']['alpha'] *= au.energy  
config['AFD_onsite']['gamma'] *= au.energy 

config['AFD_short_range']['k1'] *= au.energy  
config['AFD_short_range']['k2'] *= au.energy  
config['AFD_short_range']['k_prime'] *= au.energy  

for key in config['elastic_AFD']:
    config['elastic_AFD'][key] *= au.energy  

config['dipole_AFD_trilinear']['D'] *= au.energy / au.length

for key in config['dipole_AFD_biquadratic']:
    config['dipole_AFD_biquadratic'][key] *= au.energy / au.length**2

for key in config['spin_short_range']:
    config['spin_short_range'][key] *= au.energy**2  ## TODO: check if the unit of magnetic moment is muB or 2*muB

for key in config['spin_dipole']:
    config['spin_dipole'][key] *= au.energy / au.length**2

for key in config['spin_AFD']:
    config['spin_AFD'][key] *= au.energy

for key in config['spin_elastic']:
    config['spin_elastic'][key] *= au.energy

for key in config['spin_DM']:
    config['spin_DM'][key] *= au.energy

## save the config
with open('BFO_internal.json', 'w') as f:
    json.dump(config, f, indent=4)
