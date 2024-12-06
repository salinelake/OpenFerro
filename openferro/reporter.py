"""
Reporter of the simulation.
"""
# This file is part of OpenFerro.
import os
import time
import jax.numpy as jnp
class Thermo_Reporter:
    def __init__(self, file='thermo.log', log_interval=100, global_strain=True, volume=True, potential_energy=False, kinetic_energy=False, temperature=False):
        self.file = file
        self.counter = -1
        self.log_interval = log_interval
        self.report_global_strain = global_strain
        self.report_volume = volume
        self.report_potential_energy = potential_energy
        self.report_kinetic_energy = kinetic_energy
        self.report_temperature = temperature
        self.item_list = []

    def initialize(self, system):
        ## make the directory if not exists
        if os.path.exists(os.path.dirname(self.file)) is False:
            os.makedirs(os.path.dirname(self.file))
        so3_fields = system.get_all_SO3_fields()
        other_fields = system.get_all_non_SO3_fields()
        all_fields = so3_fields + other_fields
        self.item_list = ['Step']
        if self.report_global_strain:
            self.item_list.append('Strain-1')
            self.item_list.append('Strain-2')
            self.item_list.append('Strain-3')
            self.item_list.append('Strain-4')
            self.item_list.append('Strain-5')
            self.item_list.append('Strain-6')
        if self.report_volume:
            self.item_list.append('Volume')
        if self.report_potential_energy:
            for field in all_fields:
                self.item_list.append('Potential-{}'.format(field.ID))
        if self.report_kinetic_energy:
            for field in other_fields:
                self.item_list.append('Kinetic-{}'.format(field.ID))
        if self.report_temperature:
            for field in other_fields:
                self.item_list.append('Temperature-{}'.format(field.ID))
        ## write the header: system lattice constants, lattice type, lattice size, etc.
        with open(self.file, 'a') as f:
            ## write the start time
            f.write(f"# Log file for OpenFerro Simulation started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
            f.write(f"# Lattice Type: {system.lattice.name}, Lattice Size: {system.lattice.size}\n")
            f.write(f"# Lattice Constants: {system.lattice.a}, {system.lattice.b}, {system.lattice.c}\n")
            f.write(f"# Current Time Step: {self.counter+1}\n")
            ## write the report items
            f.write(f"# {', '.join(self.item_list)}\n")

    def step(self, system):
        self.counter += 1
        if self.counter % self.log_interval == 0:
            values = [self.counter]
            if self.report_global_strain:
                gs = system.get_field_by_ID('gstrain').get_values().flatten().tolist()
                values.extend(gs)
            if self.report_volume:
                gs = system.get_field_by_ID('gstrain').get_values().flatten().tolist()
                vol_ref = system.lattice.ref_volume
                vol = (1+gs[0] + gs[1] + gs[2]) * vol_ref
                values.append(vol)
            if self.report_potential_energy or self.report_kinetic_energy or self.report_temperature:
                so3_fields = system.get_all_SO3_fields()
                other_fields = system.get_all_non_SO3_fields()
                all_fields = so3_fields + other_fields
            if self.report_potential_energy:
                for field in all_fields:
                    values.append(field.get_potential_energy().tolist())
            if self.report_kinetic_energy:
                for field in other_fields:
                    values.append(field.get_kinetic_energy().tolist())
            if self.report_temperature:
                for field in other_fields:
                    values.append(field.get_temperature().tolist())
            with open(self.file, 'a') as f:
                f.write(", ".join(map(str, values)) + "\n")

class Field_Reporter:
    def __init__(self, field_id, log_interval=100, field_average=True, dump_field=False):
        self.field_id = field_id
        self.log_interval = log_interval
        self.report_field_average = field_average
        self.dump_field = dump_field
        self.item_list = []
        self.counter = -1
    
    def initialize(self, system):
        field = system.get_field_by_ID(self.field_id)  ## check if the field exists
        dim = field
        if self.report_field_average:
            file_name = '{}_average.log'.format(self.field_id)
            self.item_list = ['Step', 'Average']
            ## make the directory if not exists
            if os.path.exists(os.path.dirname(file_name)) is False:
                os.makedirs(os.path.dirname(file_name))
            with open(file_name, 'a') as f:
                ## write the start time
                f.write(f"# Log file for the average of field [{self.field_id}] started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
                f.write(f"# Lattice Type: {system.lattice.name}, Lattice Size: {system.lattice.size}\n")
                f.write(f"# Lattice Constants: {system.lattice.a}, {system.lattice.b}, {system.lattice.c}\n")
                f.write(f"# Current Time Step: {self.counter+1}\n")
                ## write the report items
                f.write(f"# {', '.join(self.item_list)}\n")
    
    def step(self, system):
        self.counter += 1
        if self.counter % self.dump_interval == 0:
            field = system.get_field_by_ID(self.field_id)
            values = field.get_values()
            file_name = '{}_{}.npy'.format(self.field_id, self.counter)
            jnp.save(file_name, values)
            
