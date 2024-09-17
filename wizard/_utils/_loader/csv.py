"""
_utils/_loader/csv.py
====================================

.. module:: csv
   :platform: Unix
   :synopsis: Provides csv reader and writer.

Module Overview
---------------

This module includes reader and writer for .csv-files.

Functions
---------

.. autofunction:: _read_csv

"""
import csv

from ..._core import DataCube
from ._helper import to_cube

def _read_csv(filepath: str) -> DataCube:
    """
    Read csv file.

    :param path: path to csv file
    :return:
    """
    wave_data = []
    
    with open(filepath, mode='r') as file:
        reader = csv.reader(file)
        
        # Skip header row
        headers = next(reader)  
        
        for row in reader:
            x = int(row[0])
            y = int(row[1])
            
            # All columns after x, y are waves
            waves = [int(wave_value) for wave_value in row[2:]]  
            
            wave_data.append(waves)
            
    cube = to_cube(wave_data, len_x=x, len_y=y)
    
    return DataCube(cube, wavelengths=headers[2:])