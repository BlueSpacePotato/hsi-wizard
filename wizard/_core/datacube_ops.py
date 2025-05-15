"""
_core/datacube_ops.py

.. module:: datacube_ops
   :platform: Unix
   :synopsis: DataCube Operations.

## Module Overview
This module contains operation function for processing datacubes.

## Functions
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
from scipy.ndimage import gaussian_filter

from . import DataCube 
from .._processing.spectral import calculate_modified_z_score, spec_baseline_als
from .._utils.helper import _process_slice, feature_registration, RegistrationError, auto_canny, decompose_homography


def remove_spikes(dc: DataCube, threshold: int = 6500, window: int = 5) -> DataCube:
    """
    Remove cosmic spikes from the DataCube.

    Identifies spikes using the modified z-score and replaces them
    with the mean of neighboring values within the specified window.

    Parameters
    ----------
    dc : DataCube
        The input DataCube from which cosmic spikes are to be removed.
    threshold : int, optional
        Threshold for spike detection via modified z-score, defaults to 6500.
    window : int, optional
        Window size for mean calculation when replacing spikes, defaults to 5.

    Returns
    -------
    DataCube
        The DataCube with spikes removed.
    """
    z_spectrum = calculate_modified_z_score(dc.cube.reshape(dc.shape[0], -1))
    spikes = abs(z_spectrum) > threshold
    cube_out = dc.cube.copy()
    spec_out_flat = cube_out.reshape(cube_out.shape[0], -1)
    results = Parallel(n_jobs=-1)(
        delayed(_process_slice)(spec_out_flat, spikes, idx, window)
        for idx in range(spikes.shape[0]))
    for idx, tmp in results:
        spec_out_flat[idx] = tmp
    dc.set_cube(spec_out_flat.reshape(dc.shape))
    return dc


def remove_background(dc: DataCube, threshold: int = 50, style: str = 'dark') -> DataCube:
    """
    Remove background from images in a DataCube.

    Uses an external algorithm (rembg). The first image in the DataCube
    is processed to generate a mask, which is then applied to all images
    to remove the background.

    Parameters
    ----------
    dc : DataCube
        DataCube containing the image stack.
    threshold : int, optional
        Threshold value to define the background from the alpha mask,
        defaults to 50. Pixels with alpha < threshold are considered background.
    style : str, optional
        Style of background removal, 'dark' or 'bright', defaults to 'dark'.
        If 'dark', background pixels are set to 0.
        If 'bright', background pixels are set to the max value of the cube.y

    Returns
    -------
    DataCube
        DataCube with the background removed.

    Raises
    ------
    ValueError
        If style is not 'dark' or 'bright'.
    """
    img = dc.cube[0]
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype('uint8')
    img = Image.fromarray(img)
    img_removed_bg = rembg.remove(img)
    mask = np.array(img_removed_bg.getchannel('A'))

    cube = dc.cube.copy()
    if style == 'dark':
        cube[:, mask < threshold] = 0
    elif style == 'bright':
        cube[:, mask < threshold] = dc.cube.max()
    else:
        raise ValueError("Type must be 'dark' or 'bright'")
    dc.set_cube(cube)
    return dc


def resize(dc: DataCube, x_new: int, y_new: int, interpolation: str = 'linear') -> None:
    """
    Resize the DataCube to new x and y dimensions.

    Resizes each 2D slice (x, y) of the DataCube using the specified
    interpolation method.

    Interpolation methods:
    - ``linear``: Bilinear interpolation (ideal for enlarging).
    - ``nearest``: Nearest neighbor interpolation (fast but blocky).
    - ``area``: Pixel area interpolation (ideal for downscaling).
    - ``cubic``: Bicubic interpolation (high quality, slower).
    - ``lanczos``: Lanczos interpolation (highest quality, slowest).

    Parameters
    ----------
    dc : DataCube
        The DataCube instance to be resized.
    x_new : int
        The new width (x-dimension).
    y_new : int
        The new height (y-dimension).
    interpolation : str, optional
        Interpolation method, defaults to 'linear'.
        Options: 'linear', 'nearest', 'area', 'cubic', 'lanczos'.

    Returns
    -------
    None
        The DataCube is modified in-place.

    Raises
    ------
    ValueError
        If the interpolation method is not recognized.
    """
    mode = None
    shape = dc.cube.shape
    if shape[2] > x_new:
        print('\033[93mx_new is smaller than the existing cube, you will lose information\033[0m')
    if shape[1] > y_new:
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


def baseline_als(dc: DataCube, lam: float = 1000000, p: float = 0.01, niter: int = 10) -> DataCube:
    """
    Apply Adaptive Smoothness (ALS) baseline correction.

    Iterates through each pixel (spectrum) in the DataCube and subtracts
    the baseline calculated by the `spec_baseline_als` function.

    Parameters
    ----------
    dc : DataCube
        The input DataCube.
    lam : float, optional
        The smoothness parameter for ALS, defaults to 1000000.
        Larger lambda makes the baseline smoother.
    p : float, optional
        The asymmetry parameter for ALS, defaults to 0.01.
        Value between 0 and 1. Controls how much the baseline is pushed
        towards the data (0 for minimal, 1 for maximal).
    niter : int, optional
        The number of iterations for the ALS algorithm, defaults to 10.

    Returns
    -------
    DataCube
        The DataCube with baseline correction applied.
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
    Merge two DataCubes by concatenating spectral bands.

    Concatenates the cube data and wavelengths of two DataCubes along the
    spectral (first) axis. This function requires that the spatial
    dimensions (height, width) of the cubes are the same.

    Wavelengths are concatenated. If both cubes use simple integer indices
    starting from 0 for wavelengths, the new wavelengths will also be a
    continuous range of indices.

    Parameters
    ----------
    dc1 : DataCube
        The first DataCube. This cube will be modified and returned.
    dc2 : DataCube
        The second DataCube to merge with the first.

    Returns
    -------
    DataCube
        The first DataCube (dc1) modified to include data from dc2.

    Raises
    ------
    NotImplementedError
        If spatial dimensions (height, width) do not match,
        or if wavelengths overlap and are not simple integer indices.
    """
    c1 = dc1.cube
    c2 = dc2.cube
    wave1 = dc1.wavelengths
    wave2 = dc2.wavelengths

    if c1.shape[1:] == c2.shape[1:]:  # Compare spatial dimensions (height, width)
        c3 = np.concatenate([c1, c2], axis=0)  # Concatenate along spectral axis
    else:
        raise NotImplementedError(
            'Sorry - This function is not implemented yet. '
            'At the moment you just can merge cubes with the same size x,y.'
        )

    if set(wave1) & set(wave2):
        is_wave1_indexed = list(wave1) == list(range(c1.shape[0]))
        is_wave2_indexed = list(wave2) == list(range(c2.shape[0]))
        if is_wave1_indexed and is_wave2_indexed:
            wave3 = list(range(c1.shape[0] + c2.shape[0]))
        else:
            raise NotImplementedError(
                'Sorry - your wavelengths are overlapping in a complex way, '
                'we are working on a solution for generic wavelength merging.'
            )
    else:
        wave3 = np.concatenate((wave1, wave2))

    dc1.set_cube(c3)
    dc1.set_wavelengths(wave3)
    return dc1


def inverse(dc: DataCube) -> DataCube:
    """
    Invert the DataCube values.

    This operation is useful for converting between transmission and
    reflectance data, or similar inversions. The formula applied is:
    `tmp = cube * -1`
    `tmp += -tmp.min()`
    The data type of the cube is preserved if it's 'uint16' or 'uint8'
    after temporary conversion to 'float16' for calculation.

    Parameters
    ----------
    dc : DataCube
        The DataCube to invert.

    Returns
    -------
    DataCube
        The DataCube with inverted values.
    """
    dtype = dc.cube.dtype
    if dtype == np.uint16 or dtype == np.uint8:  # Use np types for comparison
        cube = dc.cube.astype(np.float16)
    else:
        cube = dc.cube.copy()

    tmp = cube
    tmp *= -1
    tmp += -tmp.min()

    dc.set_cube(tmp.astype(dtype))
    return dc


def register_layers_simple(dc: DataCube, max_features: int = 5000, match_percent: float = 0.1) -> DataCube:
    """
    Align images within a DataCube using simple feature-based registration.

    Each layer in the DataCube is aligned to the first layer (index 0)
    using ORB feature detection and homography estimation via `_feature_registration`.

    Parameters
    ----------
    dc : DataCube
        The DataCube whose layers are to be registered.
    max_features : int, optional
        Maximum number of keypoint regions to detect, defaults to 5000.
    match_percent : float, optional
        Percentage of keypoint matches to consider for homography,
        defaults to 0.1 (10%).

    Returns
    -------
    DataCube
        The DataCube with layers registered.
    """
    o_img = dc.cube[0, :, :]
    for i in range(dc.cube.shape[0]):
        if i > 0:
            a_img = copy.deepcopy(dc.cube[i, :, :])
            try:
                _, h = feature_registration(
                    o_img=o_img, a_img=a_img,
                    max_features=max_features, match_percent=match_percent
                )
                height, width = o_img.shape
                aligned_img = cv2.warpPerspective(a_img, h, (width, height))
                dc.cube[i, :, :] = aligned_img
            except RegistrationError as e:
                print(f"Warning: Could not register layer {i} in simple registration: {e}")
                pass
    return dc


def remove_vignetting_poly(dc: DataCube, axis: int = 1, slice_params: dict = None) -> DataCube:
    """
    Remove vignetting using polynomial fitting along a specified axis.

    Calculates the mean along the specified axis (1 for rows, 2 for columns)
    for each spectral layer, fits a polynomial to this mean profile
    (after Savitzky-Golay smoothing), and subtracts this fitted profile
    from the corresponding rows/columns of the layer to correct for vignetting.

    Parameters
    ----------
    dc : DataCube
        The DataCube instance to process.
    axis : int, optional
        The axis along which to calculate the mean and apply correction.
        1 for correcting along rows (profile used for columns),
        2 for correcting along columns (profile used for rows). Defaults to 1.
    slice_params : dict, optional
        Dictionary for slicing behavior before mean calculation.
        Keys: ``"start"`` (int), ``"end"`` (int), ``"step"`` (int).
        Defaults to full slice with step 1.

    Returns
    -------
    DataCube
        The processed DataCube with vignetting removed.

    Raises
    ------
    ValueError
        If the DataCube is empty or axis is not 1 or 2.
    """
    if dc.cube is None:
        raise ValueError("The DataCube is empty. Please provide a valid cube.")

    if slice_params is None:
        slice_params = {"start": None, "end": None, "step": 1}
    start = slice_params.get("start", None)
    end = slice_params.get("end", None)
    step = slice_params.get("step", 1)

    if axis == 1:
        summed_data = np.mean(dc.cube[:, :, start:end:step], axis=2)
    elif axis == 2:
        summed_data = np.mean(dc.cube[:, start:end:step, :], axis=1)
    else:
        raise ValueError('Axis can only be 1 or 2.')

    corrected_cube = dc.cube.copy().astype(np.float32)

    for i, layer_profile in enumerate(summed_data):
        smoothed_layer_profile = savgol_filter(layer_profile, window_length=71, polyorder=1)
        if axis == 1:
            for j_col in range(dc.cube.shape[2]):
                corrected_cube[i, :, j_col] -= smoothed_layer_profile
        elif axis == 2:
            for j_row in range(dc.cube.shape[1]):
                corrected_cube[i, j_row, :] -= smoothed_layer_profile

    dc.set_cube(corrected_cube)
    return dc


def normalize(dc: DataCube) -> DataCube:
    """
    Normalize spectral information in the data cube to the range [0, 1].

    For each 2D spatial layer in the DataCube, the normalization is performed by:
    `layer = (layer - min_in_layer) / (max_in_layer - min_in_layer)`
    This scales the intensity values of each layer independently across its
    spatial dimensions.

    Parameters
    ----------
    dc : DataCube
        The DataCube instance to normalize.

    Returns
    -------
    DataCube
        The normalized DataCube.
    """
    cube = dc.cube.astype(np.float32)
    min_vals = cube.min(axis=(1, 2), keepdims=True)
    max_vals = cube.max(axis=(1, 2), keepdims=True)

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1

    cube = (cube - min_vals) / range_vals
    dc.set_cube(cube)
    return dc


def normalize_polarity(img: np.ndarray) -> np.ndarray:
    """
    Ensure features are dark-on-light by inverting if necessary.

    If the image is mostly bright (mean pixel value > 0.5 after
    normalization to [0,1]), it inverts the image.
    Handles float or uint8 inputs transparently, returning a float32
    image in the range [0,1].

    Parameters
    ----------
    img : numpy.ndarray
        Input image.

    Returns
    -------
    numpy.ndarray
        Polarity-normalized image as float32 in [0,1].
    """
    if img.dtype == np.uint8:
        img_f = img.astype(np.float32) / 255.0
    else:
        min_val, max_val = img.min(), img.max()
        if min_val < 0.0 or max_val > 1.0:
            if max_val == min_val:
                img_f = np.zeros_like(img, dtype=np.float32)
            else:
                img_f = (img.astype(np.float32) - min_val) / (max_val - min_val)
        else:
            img_f = img.astype(np.float32)

    if np.mean(img_f) > 0.5:
        img_f = 1.0 - img_f
    return img_f


def register_layers_best(
        dc: DataCube,
        ref_layer: int = 0,
        max_features: int = 5000,
        match_percent: float = 0.1,
        rot_thresh: float = 20.0,
        scale_thresh: float = 1.1
) -> DataCube:
    """
    Align DataCube layers with robust registration.

    Aligns each slice of `dc.cube` to a reference layer. It uses
    feature-based registration primarily. If feature-based registration
    yields a degenerate homography (based on rotation and scale thresholds)
    or fails, it falls back to Canny-based edge registration.
    Failed alignments are retried once.

    Parameters
    ----------
    dc : DataCube
        The DataCube to process.
    ref_layer : int, optional
        Index of the reference layer, defaults to 0.
    max_features : int, optional
        Maximum features for ORB, defaults to 5000.
    match_percent : float, optional
        Match percentage for ORB, defaults to 0.1.
    rot_thresh : float, optional
        Rotation threshold (degrees) for homography validation,
        defaults to 20.0.
    scale_thresh : float, optional
        Scale threshold for homography validation, defaults to 1.1.
        Checks if max_scale <= scale_thresh and min_scale >= 1/scale_thresh.

    Returns
    -------
    DataCube
        The DataCube with aligned layers.

    Raises
    ------
    RuntimeError
        If alignment fails for a layer after retry.
    """
    aligned_indices = {ref_layer}
    waitlist = set()
    n_layers, H_dim, W_dim = dc.cube.shape

    def try_align(layer_idx: int, current_aligned_indices: set) -> bool:
        # nonlocal dc
        a_img = dc.cube[layer_idx]
        best_alignment_img = None

        for ref_idx in current_aligned_indices:
            try:
                o_img = dc.cube[ref_idx]
                aligned_img_feat, H_ij = feature_registration(
                    o_img, a_img, max_features, match_percent
                )
                angle, S = decompose_homography(H_ij)
                s_ok = (S.max() <= scale_thresh and S.min() >= 1 / scale_thresh)
                if abs(angle) <= rot_thresh and s_ok:
                    print(f"[Layer {layer_idx}] aligned to {ref_idx}: θ={angle:.1f}°, S={S.round(3)}")
                    best_alignment_img = aligned_img_feat
                    break
                else:
                    print(f"[Layer {layer_idx}] reject vs {ref_idx}: θ={angle:.1f}°, S={S.round(3)}")
            except RegistrationError as e:
                print(f"[Layer {layer_idx}] registration to {ref_idx} failed: {e}")

        if best_alignment_img is None:
            print(f"[Layer {layer_idx}] edge-map fallback to reference layer {ref_layer}")
            try:
                ref_img_for_edge = normalize_polarity(dc.cube[ref_layer])
                tgt_img_for_edge = normalize_polarity(a_img)
                edges_ref = auto_canny(ref_img_for_edge)
                edges_tgt = auto_canny(tgt_img_for_edge)
                aligned_img_edge, H_e = feature_registration(
                    edges_ref.astype(float), edges_tgt.astype(float),
                    max_features, match_percent
                )
                h_orig, w_orig = a_img.shape
                best_alignment_img = cv2.warpPerspective(a_img, H_e, (w_orig, h_orig), flags=cv2.INTER_LINEAR)
                angle_e, S_e = decompose_homography(H_e)
                print(f"[Layer {layer_idx}] edges (vs layer {ref_layer}): θ={angle_e:.1f}°, S={S_e.round(3)}")
            except RegistrationError as e:
                print(f"[Layer {layer_idx}] edge registration failed: {e}")
            except Exception as e:
                print(f"[Layer {layer_idx}] unexpected error in edge registration: {e}")

        if best_alignment_img is not None:
            dc.cube[layer_idx] = best_alignment_img
            return True
        return False

    for i in range(n_layers):
        if i == ref_layer:
            continue
        if try_align(i, aligned_indices.copy()):
            aligned_indices.add(i)
        else:
            waitlist.add(i)

    if waitlist:
        print(f"\nRetrying layers: {list(waitlist)}\n")
        for i in list(waitlist):
            if try_align(i, aligned_indices.copy()):
                aligned_indices.add(i)
                waitlist.remove(i)
            else:
                raise RuntimeError(f"Layer {i}: alignment failed after retry.")
    return dc


def remove_vignetting(dc: DataCube, sigma: float = 50, clip: bool = True, epsilon: float = 1e-6) -> DataCube:
    """
    Remove vignetting from a hyperspectral DataCube.

    Corrects vignetting in each spectral band by estimating a smooth
    background using Gaussian blur and then performing flat-field correction.
    The background is normalized by its mean before correction.

    Parameters
    ----------
    dc : DataCube
        The input DataCube (bands, height, width).
    sigma : float, optional
        Standard deviation for Gaussian blur, controlling smoothness.
        Larger sigma means coarser background estimation. Defaults to 50.
    clip : bool, optional
        If True and the original DataCube dtype is integer,
        clip output values to the valid range of that integer type.
        Defaults to True.
    epsilon : float, optional
        A small constant to add to the background before division
        to prevent division by zero errors. Defaults to 1e-6.

    Returns
    -------
    DataCube
        The DataCube with vignetting corrected. The output cube has the
        same shape and dtype as the input.
    """
    corrected_cube = np.empty_like(dc.cube)
    orig_dtype = dc.cube.dtype
    is_int = np.issubdtype(orig_dtype, np.integer)

    for i in range(dc.cube.shape[0]):
        band = dc.cube[i].astype(np.float64)
        background = gaussian_filter(band, sigma=sigma)
        background = np.maximum(background, epsilon)
        background_mean = background.mean()
        if background_mean > epsilon:
            background /= background_mean
        else:
            background = np.ones_like(background, dtype=np.float64)
        corrected_band = band / background
        if is_int:
            info = np.iinfo(orig_dtype)
            corrected_band = np.round(corrected_band)
            corrected_band = np.clip(corrected_band, info.min, info.max)
        corrected_cube[i] = corrected_band.astype(orig_dtype)
    dc.set_cube(corrected_cube)
    return dc
