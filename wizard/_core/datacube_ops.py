"""
_core/datacube_ops.py
===========

.. module:: datacube_ops
    :platform: Unix
    :synopsis: DataCube Operations.

Module Overview
---------------

This module contains operation function for processing datacubes.

Functions
---------

.. autofunction:: merge_cubes
.. autofunction:: merge_waves

"""

import numpy as np
from . import DataCube


def merge_cubes(cube1: DataCube, cube2: DataCube) -> DataCube:
    """
    Merge to datacubes to a new one.

    :param cube1:
    :param cube2:
    :return:
    """
    c1 = cube1.cube
    c2 = cube2.cube
    if c1.shape[:2] == c2.shape[:2]:
        c3 = np.concatenate(c1, c2)
    else:
        c3 = None
        raise NotImplementedError('Sorry - '
                                  'This function is not implemented yet.'
                                  'At the moment you just can merge cubes'
                                  ' with the same size x,y.')
    return DataCube(c3)


def merge_waves(wave1: list, wave2: list) -> list:
    """
    Merge two wave lists.

    todo: better merge algorithms

    :param wave1: first list with waves
    :param wave2: second list with waves
    :return: merged waves
    :rtype: list
    """
    def common_members(a: list, b: list) -> set:
        """
        Check for comon members between two lists.
    
        :param a: list a
        :param b: list to compare to a
        :return: return a set of common members
        :rtype: set
        """
    # check for coman memebers in sets
    if set(wave1) & set(wave2):
        raise NotImplementedError('Sorry - your wavelengths are overlapping,'
                                  ' we working on a solution')

    return wave1 + wave2