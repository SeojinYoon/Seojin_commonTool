# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 13:21:00 2022

@author: Seojin
"""
from enum import Enum

class Visualizing(Enum):
    scatter = 1 << 0
    plot = 1 << 1
    mean = 1 << 2
    error = 1 << 3
    bar = 1 << 4
    swarm = 1 << 5
    violin = 1 << 6
    
    sc_pl = scatter | plot
    sc_pl_mean = scatter | plot | mean
    mean_error = mean | error
    
    bar_swarm = bar | swarm
    
    all = scatter | plot | mean | error
    
class File_validation(Enum):
    exist = 1 << 0
    only = 1 << 1
    
    exist_only = exist | only

class File_comparison(Enum):
    file_name = 1 << 0
    file_type = 1 << 1
    file_size = 1 << 2
    file_checksum = 1 << 3
    
    all = file_name | file_type | file_size | file_checksum
    
    @staticmethod
    def name(number):
        if number == File_comparison.file_name.value:
            return "name"
        elif number == File_comparison.file_type.value:
            return "type"
        elif number == File_comparison.file_size.value:
            return "size"
        elif number == File_comparison.file_checksum.value:
            return "checksum"
        
    @staticmethod
    def number(name):
        if name == File_comparison.name(File_comparison.file_name.value):
            return File_comparison.file_name.value
        elif name == File_comparison.name(File_comparison.file_type.value):
            return File_comparison.file_type.value
        elif name == File_comparison.name(File_comparison.file_size.value):
            return File_comparison.file_size.value
        elif name == File_comparison.name(File_comparison.file_checksum.value):
            return File_comparison.file_checksum.value
        
    @staticmethod
    def names(numbers):
        return [File_comparison.name(number) for number in numbers]
    
    @staticmethod
    def numbers(names):
        return [File_comparison.number(name) for name in names]

class ConnectionType(Enum):
    evoke = 0 # State change
    ret = 1 # return 

    