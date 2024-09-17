"""
_utils/_loader/_helper.py
====================================

.. module:: _helper
   :platform: Unix
   :synopsis: Provides utility functions for file operations and data transformations.

Module Overview
---------------

This module includes helper functions for file operations such as transforming data arrays,
retrieving files by extension, and converting paths to absolute form.

Functions
---------

.. autofunction:: to_cube
.. autofunction:: get_files_by_extension
.. autofunction:: make_path_absolute

"""

import os
import glob
import numpy as np


def to_cube(data, len_x, len_y) -> np.array:
    """
    Transform a 2D numpy array to a data cube-like array.

    The data is stored in Fortran-like index ordering, so the function uses Fortran order 
    instead of C order.

    :param data: 1D array of data in Fortran order.
    :type data: np.array
    :param len_x: The length of the data cube along the x-axis (pixel size x).
    :type len_x: int
    :param len_y: The length of the data cube along the y-axis (pixel size y).
    :type len_y: int
    :return: Transformed data cube.
    :rtype: np.array
    """
    return data.reshape(-1, len_x, len_y, order='F')


def get_files_by_extension(path: str, extension: str) -> list:
    """
    Return a sorted list of filenames with a given extension from a directory.

    :param path: Directory path to search for files.
    :type path: str
    :param extension: File extension to filter by (e.g., `.csv`).
    :type extension: str
    :return: List of filenames with the specified extension.
    :rtype: list
    """
    # Check if extension exists
    if not extension:
        return []

    # Check if path is valid
    if not os.path.isdir(path):
        return []

    # Check if extension doesn't start with `.`
    if not extension.startswith('.'):
        extension = '.' + extension

    return sorted(glob.glob(os.path.join(path, '*' + extension.lower())))


def make_path_absolute(path: str) -> str:
    """
    Check if the path is absolute. If not, convert it to an absolute path.

    :param path: Path to the file or directory.
    :type path: str
    :return: Absolute path to the file or directory.
    :rtype: str
    :raises ValueError: If the input path is not a string or is invalid.
    """
    if isinstance(path, str) and path:
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return path.lower()
    else:
        raise ValueError("Input path must be a string.")
