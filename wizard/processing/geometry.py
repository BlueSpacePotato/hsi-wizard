"""
wizard.processing.geometry
==========================

.. module:: geometry
:platform: Unix
:synopsis: Spatial combination and geometric utilities for DataCubes.

Module Overview
---------------

This module provides functions for geometric operations on
:class:`~wizard.core.DataCube` objects, such as merging multiple
datacubes into a single cube.

Operations may optionally include spatial alignment before combining
data. All functions modify the provided :class:`DataCube` **in-place**
and return the same object to allow method chaining.

Functions
---------
.. autofunction:: merge_cubes
"""


from skimage.transform import warp

from .._utils.helper import feature_registration, RegistrationError
from ..core import DataCube
import numpy as np
import random


def merge_cubes(dc1: DataCube, dc2: DataCube, register: bool = False) -> DataCube:
    """
    Merge two DataCubes into a single DataCube, with optional registration.

    If both datacubes are already registered and the `register` flag is True,
    the function will sample up to 10 random spectral layers from dc2, attempt to
    register each to the first layer of dc1, choose the transform with the lowest
    mean-squared-error alignment, then apply that best transform to all layers of dc2
    before merging.

    Parameters
    ----------
    dc1 : DataCube
        The first DataCube (used as reference).
    dc2 : DataCube
        The second DataCube to be merged into the first.
    register : bool, optional
        If True (default), registration will be attempted if both cubes are marked as registered.

    Returns
    -------
    DataCube
        A new DataCube containing merged spatial and spectral data.

    Raises
    ------
    NotImplementedError
        If the cubes have mismatched spatial dimensions and cannot be merged,
        or if wavelengths overlap without being purely indices.

    Examples
    --------
    >>> import wizard
    >>> dc_a = wizard.read('example.fsm')
    >>> dc_b = wizard.read('another_file.csv')
    >>> dc_a.merge_cubes(dc_b)
    """
    c1 = dc1.cube
    c2 = dc2.cube
    wave1 = dc1.wavelengths
    wave2 = dc2.wavelengths

    # Optional registration step with sampling
    if register and getattr(dc1, 'registered', False) and getattr(dc2, 'registered', False):
        print("Both datacubes registered. Sampling layers for alignment...")
        ref_img = c1[0]
        num_layers = c2.shape[0]
        sample_indices = random.sample(range(num_layers), min(10, num_layers))

        best_score = np.inf
        best_transform = None
        # Try to register sampled layers and pick best
        for idx in sample_indices:
            try:
                aligned_slice, transform = feature_registration(ref_img, c2[idx])
                # Compute alignment quality (mean squared error)
                mse = np.mean((ref_img - aligned_slice)**2)
                if mse < best_score:
                    best_score = mse
                    best_transform = transform
            except RegistrationError as e:
                print(f"Registration of sampled layer {idx} failed: {e}")

        if best_transform is not None:
            # Apply best transform to all layers of dc2
            for i in range(num_layers):
                try:
                    c2[i] = warp(c2[i], inverse_map=best_transform.inverse, preserve_range=True)
                except Exception as e:
                    print(f"Failed to apply best transform to layer {i}: {e}")
        else:
            print("No successful sampled registration. Skipping registration.")

    # Spatial size check
    if c1.shape[1:] == c2.shape[1:]:
        c3 = np.concatenate([c1, c2], axis=0)
    else:
        raise NotImplementedError(
            'Sorry - this function can only merge cubes with the same spatial dimensions.'
        )

    # Handle wavelength merge
    if set(wave1) & set(wave2):
        # If wavelengths are index-based, just concatenate indices
        if set(wave1) <= set(range(c1.shape[0])) and set(wave2) <= set(range(c2.shape[0])):
            wave3 = list(range(c1.shape[0] + c2.shape[0]))
        else:
            raise NotImplementedError(
                'Sorry - your wavelengths overlap and are not purely index-based.'
            )
    else:
        wave3 = np.concatenate((wave1, wave2))

    # Create new merged DataCube (modify dc1 in-place)
    dc1.set_cube(c3)
    dc1.set_wavelengths(wave3)

    return dc1
