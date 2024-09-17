"""
__init__.py
===========

.. module:: __init_
   :platform: Unix
   :synopsis: init method for hsi-wizard package.

Module Overview
---------------

This module inits the `hsi-wizard`.

"""


# Import necessary submodules and classes/functions from them
from ._core.datacube import DataCube
from ._exploration.plotter import plotter

from ._utils._loader import read


# Define what should be accessible when using 'from wizard import *'
__all__ = [
    'DataCube',
    'read',
    'plotter'
]

# Example of setting package metadata
__version__ = "0.1.2"
__author__ = 'flx'
