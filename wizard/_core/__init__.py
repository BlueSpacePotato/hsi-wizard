"""
_core/__init__.py
===========

.. module:: __init__
   :platform: Unix
   :synopsis: init method for _core package

Module Overview
---------------

This module inits the core funciton of the `hsi-wizard`.

"""


from .datacube import DataCube
from .datacube_ops import merge_cubes, merge_waves

__all__ = [
    'DataCube',
    'merge_cubes',
    'merge_waves'
]
