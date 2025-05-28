.. _datacube_ops:

DataCube Operations
===================

.. module:: wizard._core.datacube_ops
    :platform: Unix
    :synopsis: DataCube Operations.

Module Overview
---------------

This module contains functions for processing datacubes. The methods are dynamically added to the DataCube class in its __init__ method. Therefore, they can be used as standalone functions or as methods of the DataCube class.

Functions
---------

.. _remove_spikes:
.. autofunction:: remove_spikes

.. _remove_background:
.. autofunction:: remove_background

.. _resize:
.. autofunction:: resize

.. _baseline_als:
.. autofunction:: baseline_als

.. _merge_cubes:
.. autofunction:: merge_cubes

.. _inverse:
.. autofunction:: inverse

.. _register_layers:
.. autofunction:: register_layers

.. _remove_vingetting:
.. autofunction:: remove_vingetting
