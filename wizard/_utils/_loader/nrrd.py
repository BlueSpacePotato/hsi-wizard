"""
_utils/_loader/nrrd.py
========================

.. module:: nrrd
   :platform: Unix
   :synopsis: Provides functions to read and write NRRD files.

Module Overview
---------------

This module includes functions for reading and writing .nrrd files, which are commonly used 
to store multi-dimensional data, particularly in the field of medical imaging.

Functions
---------

.. autofunction:: _read_nrrd
.. autofunction:: _write_nrrd

"""

import os
import nrrd
import numpy as np
from ..._core import DataCube

def _read_nrrd(path: str) -> DataCube:
    """
    Read a NRRD file and convert it into a DataCube.

    :param path: The file path to the NRRD file.
    :type path: str
    :return: A DataCube containing the data from the NRRD file.
    :rtype: DataCube

    :raises FileNotFoundError: If the specified file does not exist.
    :raises ValueError: If the NRRD file does not contain the expected metadata.

    :Example:

    >>> dc = _read_nrrd('path/to/file.nrrd')
    >>> print(dc.shape)  # Output: shape of the data cube
    """
    file = nrrd.read(filename=path)

    wavelengths = list(map(int, file[1]['wavelengths'].strip('[]').split()))

    notation = file[1]['notation'] if file[1]['notation'] != 'None' else None
    record = file[1]['record'] == 'True'

    return DataCube(cube=file[0], wavelengths=wavelengths, name=file[1]['name'], notation=notation, record=record)


def _write_nrrd(dc: DataCube, path: str) -> None:
    """
    Write a DataCube to a NRRD file.

    :param dc: The DataCube to write to the NRRD file.
    :type dc: DataCube
    :param path: The file path where the NRRD file will be saved.
    :type path: str
    :return: None

    :raises ValueError: If the DataCube is empty or does not contain valid data.

    :Example:

    >>> dc = DataCube(cube=np.random.rand(10, 20, 30), wavelengths=[400, 500, 600], name='Sample Data')
    >>> _write_nrrd(dc, 'path/to/output.nrrd')
    """
    if dc.cube.size == 0:
        raise ValueError("The DataCube is empty. Cannot write to file.")

    header = {
        'wavelengths': dc.wavelengths,
        'name': dc.name,
        'notation': dc.notation,
        'record': dc.record
    }

    nrrd.write(file=path, data=dc.cube, header=header)