"""
__init__.py
============

.. module:: __init__
   :platform: Unix
   :synopsis: Initialization module for the hsi-wizard package.

Module Overview
---------------

This module initializes the `hsi-wizard` package, making key components
accessible for users.

Importing
---------

This module imports essential submodules and classes/functions, including:

- `DataCube` from the `_core.datacube` module
- `plotter` from the `_exploration.plotter` module
- `read` from the `_utils._loader` module

:no-index:
"""

# Import necessary submodules and classes/functions from them
from ._core.datacube import DataCube
from ._exploration.plotter import plotter
from ._exploration.surface import plot_surface
from ._exploration.faces import plot_datacube_faces
from .io import read
from ._processing.cluster import isodata, smooth_kmeans

__all__ = ["DataCube", "read", "plotter"]


# Meta Data
__version__ = "1.0.0"
__author__ = 'flx'
