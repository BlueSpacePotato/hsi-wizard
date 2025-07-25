.. _pca:

PCA
===

The :meth:`pca()` function performs spectral dimensionality reduction on a :class:`DataCube` by computing its principal components.
It decomposes the spectral signatures into orthogonal axes ordered by variance, then projects each pixel’s spectrum onto the first N components.

Example
-------

The following example demonstrates PCA on a synthetically generated :class:DataCube, extracting the first three principal components.

.. literalinclude:: ../../../../examples/02_process/05_pca.py
   :language: python
   :linenos:

Output:

   .. code-block:: text

      Reduced cube shape: (10, 200, 200)