.. _cluster:

cluster
=======

This module initializes the cluster functions of the hsi-wizard package.

Module Overview
---------------

.. currentmodule:: wizard._processing.cluster

Functions
---------

.. autofunction:: compute_pairwise_distances
.. autofunction:: isodata

.. note::
  The Isodata code was inspired by `pyRadar <https://github.com/PyRadar/pyradar/>` from PyRadar.


  .. warning::
   Agglomerative clustering with spatial connectivity is conceptually elegant, but it doesn’t scale well to large 2D grids, and processing very large datasets can lead to high computing time.
