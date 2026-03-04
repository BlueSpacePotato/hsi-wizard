"""
wizard.processing.denoise
========================

.. module:: #denoise
:platform: Unix
:synopsis: Denoising and artifact removal utilities for DataCubes.

Module Overview
---------------

This module provides functions to reduce noise and remove common artifacts in
:class:`~wizard.core.DataCube` objects, including spike removal in spectral
signals and spatial smoothing of individual spectral bands.

All functions modify the provided :class:`DataCube` **in-place** and return the
same object to allow method chaining.

Functions
---------
.. autofunction:: remove_spikes
.. autofunction:: uniform_filter_dc
"""

from ..core import DataCube
from .._utils.helper import _process_slice
from ..processing.spectral import calculate_modified_z_score


import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import uniform_filter


def remove_spikes(dc: DataCube, threshold: int = 6500, window: int = 5) -> DataCube:
    """
    Remove cosmic spikes from each pixel's spectral data.

    This function computes the modified z-score for each pixel's spectral vector,
    identifies spikes where the score exceeds the threshold, and replaces spike
    values with the local mean within a sliding window along the spectrum.

    Parameters
    ----------
    dc : DataCube
        The input DataCube with shape (v, x, y), where v is the number of spectral bands.
    threshold : int, optional
        Threshold for spike detection via modified z-score, defaults to 6500.
    window : int, optional
        Window size (in spectral channels) for mean replacement of spikes, defaults to 5.

    Returns
    -------
    DataCube
        A new DataCube instance with spikes removed per-pixel.

    Raises
    ------
    ValueError
        If `window` is not in the range [1, number of spectral bands].

    Notes
    -----
    - The original DataCube is not modified in place; es wird eine Kopie zurückgegeben.
    - Die Modifizierte z-Score-Berechnung erwartet Input mit Form (n_samples, n_features).
    - Parallelisierung beschleunigt die Einzelpixel-Bearbeitung.

    Examples
    --------
    >>> import wizard
    >>> dc = wizard.read("example.fsm")
    >>> dc.remove_spikes(threshold=6500, window=5)
    """
    v, x, y = dc.cube.shape
    if not (1 <= window <= v):
        raise ValueError(f"window must be between 1 and {v}, got {window}")

    # reshape to (n_pixels, v)
    n_pixels = x * y
    flat_cube = dc.cube.reshape(v, n_pixels).T  # shape: (n_pixels, v)

    # Berechne pro-pixel modifizierten z-score
    z_scores = calculate_modified_z_score(flat_cube)  # (n_pixels, v)
    spikes = np.abs(z_scores) > threshold
    flat_out = flat_cube.copy()

    # Parallel auf jedes Pixel anwenden
    results = Parallel(n_jobs=-1)(
        delayed(_process_slice)(flat_out, spikes, idx, window)
        for idx in range(n_pixels)
    )
    for idx, spec in results:
        flat_out[idx] = spec

    # zurück in (v, x, y) formen
    clean_cube = flat_out.T.reshape(v, x, y)

    # Kopie des DataCube mit dem bereinigten Cube

    dc.set_cube(clean_cube)
    return dc


def uniform_filter_dc(dc, size=3):
    """
    Smooth each spectral band of a DataCube using a uniform spatial filter.

    Applies a uniform filter of the given window size to every slice (band) in the
    DataCube’s cube, reducing spatial noise by averaging within a local neighborhood.
    The operation modifies the DataCube in place.

    Parameters
    ----------
    dc : DataCube
        The DataCube instance whose `cube` attribute (a numpy array of shape (v, x, y))
        will be smoothed across the spatial dimensions for each spectral band.
    size : int, optional
        The size of the square window used by `scipy.ndimage.uniform_filter` for
        smoothing. Must be a positive odd integer. Defaults to 3.

    Returns
    -------
    DataCube
        The same DataCube instance, with its `cube` attribute replaced by the
        smoothed data of shape (v, x, y).

    Raises
    ------
    ValueError
        If `size` is not a positive integer.

    Notes
    -----
    - Requires `scipy.ndimage.uniform_filter` to be imported.
    - Smoothing is performed independently on each spectral band.
    - This function updates `dc` in place; no new DataCube is created.

    """
    if not isinstance(size, int) or size < 1:
        raise ValueError("`size` must be a positive integer")
    cube = dc.cube
    for i in range(dc.cube.shape[0]):
        cube[i] = uniform_filter(cube[i], size=size)
    dc.set_cube(cube)
    return dc
