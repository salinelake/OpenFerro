"""
Reporter of the simulation.
"""
# This file is part of OpenFerro.
import os
import time
import jax.numpy as jnp
class Thermo_Reporter:
    def __init__(self, file='thermo.log', log_interval=100, global_strain=False, volume=True, potential_energy=False, kinetic_energy=False, temperature=False):
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
        all_fields = system.get_all_fields()
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
            self.item_list.append('Total_Potential_Energy')
        if self.report_kinetic_energy:
            for field in all_fields:
                self.item_list.append('Kinetic-{}'.format(field.ID))
        if self.report_temperature:
            for field in all_fields:
                self.item_list.append('Temperature-{}'.format(field.ID))
        ## write the header: system lattice constants, lattice type, lattice size, etc.
        with open(self.file, 'a') as f:
            ## write the start time
            f.write(f"# Log file for OpenFerro Simulation started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
            f.write(f"# Lattice Type: {system.lattice.name}, Lattice Size: {system.lattice.size}\n")
            f.write(f"# Lattice Vectors: {system.lattice.a1}, {system.lattice.a2}, {system.lattice.a3}\n")
            f.write(f"# Current Time Step: {self.counter+1}\n")
            ## write the report items
            f.write("# ")
            f.write(", ".join(self.item_list))
            f.write("\n")

    def step(self, system):
        self.counter += 1
        if self.counter % self.log_interval == 0:
            values = [self.counter]
            if self.report_global_strain:
                gs = system.get_field_by_ID('gstrain').get_values().flatten().tolist()
                values.extend(gs)
            if self.report_volume:
                try:
                    gs = system.get_field_by_ID('gstrain').get_values().flatten().tolist()
                except:
                    gs = [0.0, 0.0, 0.0]
                vol_ref = system.lattice.ref_volume
                vol = (1+gs[0] + gs[1] + gs[2]) * vol_ref
                values.append(vol)
            all_fields = system.get_all_fields()
            if self.report_potential_energy:
                values.append(system.calc_total_potential_energy())
            if self.report_kinetic_energy:
                for field in all_fields:
                    values.append(field.get_kinetic_energy())
            if self.report_temperature:
                for field in all_fields:
                    values.append(field.get_temperature())
            with open(self.file, 'a') as f:
                f.write(", ".join(map(str, values)))
                f.write("\n")

class Field_Reporter:
    def __init__(self, file_prefix, field_ID, log_interval=100, field_average=True, dump_field=False):
        self.file_prefix = file_prefix
        self.file_avg_name = '{}_avg.log'.format(self.file_prefix)
        self.field_ID = field_ID
        self.log_interval = log_interval
        self.report_field_average = field_average
        self.dump_field = dump_field
        self.item_list = []
        self.counter = -1
    
    def initialize(self, system):
        field = system.get_field_by_ID(self.field_ID)  ## check if the field exists
        dim = field
        if self.report_field_average:
            self.item_list = ['Step', 'Average(Vector)']
            with open(self.file_avg_name, 'a') as f:
                ## write the start time
                f.write(f"# Log file for the average of field [{self.field_ID}] started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
                f.write(f"# Lattice Type: {system.lattice.name}, Lattice Size: {system.lattice.size}\n")
                f.write(f"# Lattice Vectors: {system.lattice.a1}, {system.lattice.a2}, {system.lattice.a3}\n")
                f.write(f"# Current Time Step: {self.counter+1}\n")
                ## write the report items
                f.write("# ")
                f.write(", ".join(self.item_list))
                f.write("\n")
    
    def step(self, system):
        self.counter += 1
        if self.counter % self.log_interval == 0:
            field = system.get_field_by_ID(self.field_ID)
            values = field.get_values()
            dim = len(values.shape)
            if self.report_field_average:
                over_axes = tuple(range(dim-1))
                with open(self.file_avg_name, 'a') as f:
                    f.write("{}, ".format(self.counter))
                    f.write(", ".join(map(str, values.mean(over_axes))))
                    f.write("\n")
            if self.dump_field:
                file_name = '{}_dump_{}.npy'.format(self.file_prefix, self.counter)
                jnp.save(file_name, values)
            
