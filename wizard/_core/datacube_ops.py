"""
_core/datacube_ops.py

.. module:: datacube_ops
    :platform: Unix
    :synopsis: DataCube Operations.

Module Overview
---------------

This module contains operation function for processing datacubes.

Functions
---------

.. autofunction:: remove_spikes
.. autofunction:: resize

"""

import cv2
import copy
import rembg
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from scipy.signal import savgol_filter

from . import DataCube

from .._processing.spectral import calculate_modified_z_score, spec_baseline_als
from .._utils.helper import feature_regestration


def _process_slice(spec_out_flat, spikes_flat, idx, window):
    """
    Process a single slice of the data cube to remove spikes by replacing them with the mean
    of neighboring values within a given window.

    Parameters
    ----------
    spec_out_flat : numpy.ndarray
        Flattened output spectrum data from the data cube.
    spikes_flat : numpy.ndarray
        Flattened boolean array indicating where spikes are detected in the cube.
    idx : int
        Index of the current slice to process.
    window : int
        The size of the window used to calculate the mean of neighboring values.

    Returns
    -------
    tuple
        A tuple containing the index of the processed slice and the modified spectrum slice.
    """

    w_h = int(window / 2)
    spike = spikes_flat[idx]
    tmp = np.copy(spec_out_flat[idx])

    for spk_idx in np.where(spike)[0]:
        window_min = max(0, spk_idx - w_h)
        window_max = min(len(tmp), spk_idx + w_h + 1)

        if window_min == spk_idx:
            window_data = tmp[spk_idx + 1:window_max]
        elif window_max == spk_idx + 1:
            window_data = tmp[window_min:spk_idx]
        else:
            window_data = np.concatenate((tmp[window_min:spk_idx], tmp[spk_idx + 1:window_max]))

        if len(window_data) > 0:
            tmp[spk_idx] = np.mean(window_data)

    return idx, tmp


def remove_spikes(dc, threshold: int = 6500, window: int = 5):
    """
    Remove cosmic spikes from the data cube based on a z-score threshold and a smoothing window.

    This function identifies spikes using the modified z-score and replaces the detected spikes
    with the mean of neighboring values within the specified window.

    Parameters
    ----------
    dc : DataCube
        The input `DataCube` from which cosmic spikes are to be removed.
    threshold : int, optional
        The threshold value for detecting spikes based on the modified z-score. Default is 6500.
    window : int, optional
        The size of the window used to calculate the mean of neighboring values
        when replacing spikes. Default is 3.

    Returns
    -------
    DataCube
        The `DataCube` with spikes removed.
    """
    z_spectrum = calculate_modified_z_score(dc.cube.reshape(dc.shape[0], -1))
    spikes = abs(z_spectrum) > threshold

    cube_out = dc.cube.copy()

    spec_out_flat = cube_out.reshape(cube_out.shape[0], -1)

    results = Parallel(n_jobs=-1)(
        delayed(_process_slice)(spec_out_flat, spikes, idx, window) for idx in range(spikes.shape[0]))

    for idx, tmp in results:
        spec_out_flat[idx] = tmp

    dc.set_cube(spec_out_flat.reshape(dc.shape))

    return dc


def remove_background(dc: DataCube, threshold:int = 50, style:str = 'dark') -> DataCube:
    """
    Removes the background from the images in a DataCube using an external algorithm.

    The first image in the DataCube (used as a reference) is processed to generate a mask,
    which is then applied to all images to remove the background.

    :param dc: DataCube containing the image stack.
    :param threshold: Threshold value to define the background.
    :param style: 'dark' or 'bright'
    :return: DataCube with the background removed from all images.
    """

    # Normalize and convert the first image to uint8 for processing
    img = dc.cube[0]
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype('uint8')

    # Apply background removal
    img = Image.fromarray(img)
    img = rembg.remove(img)
    mask = np.array(img.getchannel('A'))  # Extract alpha channel as mask

    # Apply mask to the entire datacube
    cube = dc.cube.copy()
    if style == 'dark':
        cube[:, mask < threshold] = 0  # Vectorized operation for efficiency
    elif style == 'bright':
        cube[:, mask < threshold] = dc.cube.max()
    else:
        raise ValueError('Type must be dark or bright')

    dc.set_cube(cube)
    return dc


def resize(dc, x_new: int, y_new: int, interpolation: str = 'linear') -> None:
    """
    Resize the data cube to new x and y dimensions using the specified interpolation method.

    This function resizes each 2D slice (x, y) of the data cube according to the provided dimensions
    and interpolation method.

    Interpolation methods:
    - `linear`: Bilinear interpolation (ideal for enlarging).
    - `nearest`: Nearest neighbor interpolation (fast but blocky).
    - `area`: Pixel area interpolation (ideal for downscaling).
    - `cubic`: Bicubic interpolation (high quality, slower).
    - `lanczos`: Lanczos interpolation (highest quality, slowest).

    Parameters
    ----------
    dc : DataCube
        The `DataCube` instance to be resized.
    x_new : int
        The new width (x-dimension) of the data cube.
    y_new : int
        The new height (y-dimension) of the data cube.
    interpolation : str, optional
        The interpolation method to use for resizing. Default is 'linear'.

    Raises
    ------
    ValueError
        If the specified interpolation method is not recognized.

    Returns
    -------
    None
        The function modifies the `DataCube` in-place.
    """

    mode = None

    shape = dc.cube.shape

    if shape[1] > x_new:
        print('\033[93mx_new is smaller than the existing cube, you will lose information\033[0m')
    if shape[2] > y_new:
        print('\033[93my_new is smaller than the existing cube, you will lose information\033[0m')

    if interpolation == 'linear':
        mode = cv2.INTER_LINEAR
    elif interpolation == 'nearest':
        mode = cv2.INTER_NEAREST
    elif interpolation == 'area':
        mode = cv2.INTER_AREA
    elif interpolation == 'cubic':
        mode = cv2.INTER_CUBIC
    elif interpolation == 'lanczos':
        mode = cv2.INTER_LANCZOS4
    else:
        raise ValueError(f'Interpolation method `{interpolation}` not recognized.')

    _cube = np.empty(shape=(shape[0], y_new, x_new))
    for idx, layer in enumerate(dc.cube):
        _cube[idx] = cv2.resize(layer, (x_new, y_new), interpolation=mode)
    dc.cube = _cube
    dc._set_cube_shape()


def baseline_als(dc: DataCube = None, lam: float = 1000000, p: float = 0.01, niter: int = 10) -> DataCube:
    """

    :param dc:
    :param lam:
    :param p:
    :param niter:
    :return:
    """
    for x in range(dc.shape[1]):
        for y in range(dc.shape[2]):
            dc.cube[:, x, y] -= spec_baseline_als(
                spectrum=dc.cube[:, x, y],
                lam=lam,
                p=p,
                niter=niter
            )
    return dc


def merge_cubes(dc1: DataCube, dc2: DataCube) -> DataCube:
    """
    Merge to datacubes to a new one.

    :param dc1:
    :param dc2:
    :return:
    """
    c1 = dc1.cube
    c2 = dc2.cube
    wave1 = dc1.wavelengths
    wave2 = dc2.wavelengths

    if c1.shape[:2] == c2.shape[:2]:
        c3 = np.concatenate([c1, c2])
    else:
        raise NotImplementedError('Sorry - '
                                  'This function is not implemented yet.'
                                  'At the moment you just can merge cubes'
                                  ' with the same size x,y.')

    # check for coman memebers in sets
    if set(wave1) & set(wave2):
        raise NotImplementedError('Sorry - your wavelengths are overlapping,'
                                  ' we working on a solution')
    else:
        wave3 = np.concatenate((wave1, wave2))

    dc1.set_cube(c3)
    dc1.set_wavelengths(wave3)
    return dc1


def inverse(dc: DataCube) -> DataCube:
    """
    Invert the datacube, handy for flipping transmission and reflextion data

    :param dc: wizard.DataCube
    :return: dc
    """
    tmp = dc.cube
    tmp *= -1
    tmp += - tmp.min()

    dc.set_cube(tmp)
    return dc


def register_layers(dc: DataCube, max_features: int = 5000, match_percent: float = 0.1) -> DataCube:
    """
    Align the images within a datacube by using a feature-based image registration.

    :param dc: Datacube
    :param exclude: List of specific names of datacubes, which shall not be registered
    :param max_features: Int value of the maximum number of keypoint regions
    :param match_percent: Float percentage of keypoint matches to consider
    :return: Registered datacube
    """
    o_img = dc.cube[0, :, :]
    for i in range(dc.cube.shape[0]):
        if i > 0:
            a_img = copy.deepcopy(dc.cube[i, :, :])
            _, h = feature_regestration(o_img=o_img, a_img=a_img, max_features=max_features, match_percent=match_percent)

            height, width = o_img.shape
            aligned_img = cv2.warpPerspective(a_img, h, (width, height))
            dc.cube[i, :, :] = aligned_img

    return dc


def remove_vingetting(dc: DataCube, axis: int = 1, slice_params: dict = None) -> DataCube:
    """
    Process a DataCube by summing along the specified axis, fitting a best-guess polygon to each layer,
    and plotting the results.

    Parameters
    ----------
    dc : DataCube
        The DataCube instance to process.
    axis : int, optional
        The axis along which to sum the data (default is 1).
    slice_params : dict, optional
        A dictionary specifying the slicing behavior. Keys can include:
            - "start": int, starting index for slicing (default is None, meaning no slicing at the start).
            - "end": int, ending index for slicing (default is None, meaning no slicing at the end).
            - "step": int, step size for slicing (default is 1).

    Returns
    -------
    DataCube
    The processed DataCube with vingetting removed.
    """
    if dc.cube is None:
        raise ValueError("The DataCube is empty. Please provide a valid cube.")

    # Handle slicing dynamically based on slice_params
    if slice_params is None:
        slice_params = {"start": None, "end": None, "step": 1}

    start = slice_params.get("start", None)
    end = slice_params.get("end", None)
    step = slice_params.get("step", 1)

    # Summing along the specified axis with dynamic slicing
    if axis == 1:
        sliced_data = dc.cube[:, start:end:step]
    elif axis == 2:
        sliced_data = dc.cube[:, :, start:end:step]
    else:
        raise ValueError('Axis cant only be 1 or 2.')

    summed_data = np.mean(sliced_data, axis=axis)

    # Create a copy of the cube to modify
    corrected_cube = dc.cube.copy().astype('float32')

    # Define a polynomial function for fitting
    def polynomial(x, *coeffs):
        return sum(c * x**i for i, c in enumerate(coeffs))

    # Plot each layer
    for i, layer in enumerate(summed_data):

        # Fit a polynomial to the data
        smoothed_layer = savgol_filter(layer, window_length=71, polyorder=1)

        for j in range(dc.cube.shape[axis]):
            if axis == 1:
                corrected_cube[i, j, :] -= smoothed_layer
            elif axis == 2:
                corrected_cube[i, :, j] -= smoothed_layer

    dc.set_cube(corrected_cube)

    return dc
