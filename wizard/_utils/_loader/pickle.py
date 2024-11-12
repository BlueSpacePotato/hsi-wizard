"""
_utils/_loader/pickle.py
==========================

.. module:: pickle
   :platform: Unix
   :synopsis: Provides functions to read and write pickle files.

Module Overview
---------------

This module includes functions for reading pickle files that store serialized Python objects,
specifically NumPy arrays. It converts the loaded data into a DataCube format for further processing.

Functions
---------

.. autofunction:: _read_pickle

"""

import numpy as np
from ..._core import DataCube


def _read_pickle(path: str) -> DataCube:
    """
    Load a pickled NumPy array and convert it into a DataCube.

    :param path: The file path to the pickle file.
    :type path: str
    :return: A DataCube containing the loaded NumPy array.
    :rtype: DataCube

    :raises FileNotFoundError: If the specified file does not exist.
    :raises ValueError: If the loaded data is not in the expected format.

    :Example:

    >>> dc = _read_pickle('path/to/file.pkl')
    >>> print(dc.shape)  # Output: shape of the DataCube
    """
    data = np.load(path, allow_pickle=True)

    if not isinstance(data, np.ndarray):
        raise ValueError("Loaded data is not a NumPy array.")

    # Create and return the DataCube
    return DataCube(data)
