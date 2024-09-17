"""
_utils/_loader/pickle.py
====================================

.. module:: pickle
   :platform: Unix
   :synopsis: Provides pickle reader and writer.

Module Overview
---------------

This module includes reader and writer for pickle-files.


Functions
---------

.. autofunction:: _read_pickle


"""

import numpy as np

from ..._core import DataCube

def _read_pickle(path) -> DataCube:
    """
    Load numpy data.

    :param path:
    :return:
    """
    data = np.load(path)
    
    # put data in DataCube and return
    return DataCube(data)