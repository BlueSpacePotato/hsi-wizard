"""
_processing/spectral.py
============================

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
.. autofunction:: process_slice
.. autofunction:: remove_spikes

"""

import numpy as np
from joblib import Parallel, delayed


def calculate_modified_z_score(cube):
    """
    Calculate the modified z-score of a data cube by computing the difference in intensity
    along the first axis.
    
    :param cube: The input data cube of shape (Z, Y, X), where Z is the number of slices,
                 Y is the height, and X is the width.
    :type cube: numpy.ndarray
    :return: The modified z-score, which is the difference in intensity along the first axis.
    :rtype: numpy.ndarray

    :Example:
    
    >>> cube = np.random.rand(10, 20, 30)  # Example data cube of shape (10, 20, 30)
    >>> modified_z_score = calculate_modified_z_score(cube)
    >>> print(modified_z_score.shape)  # Output: (20, 30)
    """
    delta_intensity = np.diff(cube, axis=0)
    return delta_intensity


def process_slice(spec_out_flat, spikes_flat, idx, window):
    """
    Process a single slice of the data cube to remove spikes by replacing them with the mean
    of the neighboring values within a given window.
    
    :param spec_out_flat: Flattened output spectrum.
    :type spec_out_flat: numpy.ndarray
    :param spikes_flat: Flattened array indicating the presence of spikes.
    :type spikes_flat: numpy.ndarray
    :param idx: Index of the current slice to process.
    :type idx: int
    :param window: Size of the window used to calculate the mean of neighboring values.
    :type window: int
    :return: A tuple containing the index of the processed slice and the modified slice.
    :rtype: tuple

    """
    w_h = int(window / 2)
    spike = spikes_flat[idx]
    tmp = np.copy(spec_out_flat[idx])

    for spk_idx in np.where(spike)[0]:
        window_min = max(0, spk_idx - w_h)
        window_max = min(len(tmp), spk_idx + w_h + 1)

        if window_min == spk
