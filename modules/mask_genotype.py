#!/usr/bin/env python
# coding: utf-8


from intervaltree import Interval, IntervalTree
import numpy as np


class MissingGenotypeMask:
    """ Make a missing genotype mask using an interval tree. """
    def __init__(self,
                 individuals, # List of individual names/ids
                 sequence_length = 1_000_000, # Default: 1 Mbp
                 proportion_missing = 0.01,
                 num_regions_missing = 1_000,
                 contig_id = None):
        self.individuals = individuals
        self.sequence_length = sequence_length
        self.proportion_missing = float(proportion_missing)
        self.num_regions_missing = num_regions_missing
        
        assert len(self.individuals) > 0,            "len(individuals) must be at least 0."
        assert self.sequence_length > 0,            "sequence_length must be at least 0."
        assert self.proportion_missing >= 0            and self.proportion_missing <= 1,            "proportion_missing must be between 0 and 1."
        assert self.num_regions_missing > 0,            "num_regions_missing must be at least 0."
        
        self.contig_id = contig_id
        self.total_missing_length = round(self.sequence_length * self.proportion_missing)
        self.region_size = round(self.total_missing_length / num_regions_missing)
        
        assert self.total_missing_length >= 1,            "total_missing_length must be at least 1."
        assert self.region_size >= 1,            "region_size must be at least 1."
        
        self.mask = {individual: self.create_random_mask()
                     for individual in self.individuals}
        
    def create_random_mask(self):
        mask = IntervalTree()
        missing_positions = np.random.randint(0,
                                              self.sequence_length - 1,
                                              self.num_regions_missing)
        for i, start in enumerate(missing_positions):
            # Half-open interval [start, end)
            end = min(start + self.region_size,
                      self.sequence_length - 1)
            mask[start:end] = True
        return(mask)
        
    def get_missing_positions(self, individual):
        try:
            for i, interval in enumerate(sorted(self.mask[individual])):
                print(f"{i} {interval}")
        except KeyError:
            print(f"Individual {individual} cannot be found.")
            
    def query_position(self, individual, position):
        try:
            if len(self.mask[individual][position]) > 0:
                return(True)
            else:
                return(False)
        except KeyError:
            print(f"Individual {individual} cannot be found.")
