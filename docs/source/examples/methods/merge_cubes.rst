.. _merge_cubes:

Merge Cubes
===========

This example shows how to merge two synthetic :class:`DataCubes`, using the :func:`dc.merge_cubes()` method.

Example
-------

Below is a script that:

* Generates a synthetic data useing :func:`generate_pattern_stack`.
* Builds two :class:`DataCube` instances (`dc_a` and `dc_b`) with different spatial resolutions and wavelength ranges.
* Displays a side-by-side comparison of a single wavelength slice before resizing.
* Resizes `dc_b` to match the spatial dimensions of `dc_a` and displays the aligned slices.
* Merges `dc_b` into `dc_a`, producing a combined cube with both wavelength sets.
* Uses the :func:`plotter` to visualize the merged cube interactively.

.. literalinclude:: ../../../../examples/01_dc_methods/02_merge_cubes.py
   :language: python
   :linenos:

Figure Outputs
--------------

**Figure 1: Initial Comparison**

The first figure presents `dc_a` and `dc_b` at the same wavelength index. You can see that `dc_b` (right) has half the resolution of `dc_a` (left), causing a blockier appearance.

.. figure:: ../../../../resources/imgs/examples/merge_1.png
   :align: center
   :alt: Original dc_a vs. low-res dc_b slice

Display cube infos:

.. code-block:: text

  Name: dc_a                                          Name: dc_b
  Shape: (20, 600, 400)                               Shape: (20, 300, 200)
  Wavelengths:                                        Wavelengths:
          Len: 20                                             Len: 20
          From: 0                                             From: 60
          To: 20                                              To: 100

**Figure 2: After Resizing**

The second figure shows both cubes after calling `dc_b.resize(...)`. Now `dc_b` matches `dc_a` in spatial dimensions, and their patterns align perfectly.

.. figure:: ../../../../resources/imgs/examples/merge_2.png
   :align: center
   :alt: dc_a vs. resized dc_b slice

`dc_b` print output after resize:

.. code-block:: text

  Name: dc_b
  Shape: (20, 600, 400)
  Wavelengths:
         Len: 20
         From: 60
         To: 100

**Figure 3: Merged Cube Visualization**

After merging, `dc_a` contains two spectral ranges: the original wavelengths and those from `dc_b`.

.. figure::  ../../../../resources/imgs/examples/merge_3.png
   :align: center
   :alt: Interactive plot of merged DataCube slices

`dc_a` print output after merge:

.. code-block:: text

  Name: dc_a
  Shape: (40, 600, 400)
  Wavelengths:
         Len: 40
         From: 0
         To: 100