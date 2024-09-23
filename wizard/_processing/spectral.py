"""
_processing/spectral.py
========================

.. module:: spectral
   :platform: Unix
   :synopsis: Provides methods for processing data cubes, including spike removal and intensity calculations.

Module Overview
---------------

This module includes functions for processing data cubes. It calculates the modified z-score of a data cube to
identify intensity differences, removes spikes from the data by replacing them with mean values of neighboring data,
and processes data cubes in parallel to enhance performance.

Functions
---------

.. autofunction:: calculate_modified_z_score

"""

import numpy as np

def calculate_modified_z_score(cube):
    """
    Calculate the modified z-score of a data cube by computing the difference in intensity
    along the first axis.
    
    :param cube: The input data cube of shape (Z, Y, X), where Z is the number of slices,
                 Y is the height, and X is the width.
    :type cube: numpy.ndarray
    :return: The modified z-score, which represents the difference in intensity along the first axis.
    :rtype: numpy.ndarray

    :Example:
    
    >>> cube = np.random.rand(10, 20, 30)  # Example data cube of shape (10, 20, 30)
    >>> modified_z_score = calculate_modified_z_score(cube)
    >>> print(modified_z_score.shape)  # Output: (9, 20, 30)
    """
    delta_intensity = np.diff(cube, axis=0)
    return delta_intensity
