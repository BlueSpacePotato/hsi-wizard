"""
_utils/helper.py
=================

.. module:: helper
   :platform: Unix
   :synopsis: Helper functions for processing wave and cube values.

Module Overview
---------------

This module contains helper functions to assist in processing wave and cube values.

Functions
---------

.. autofunction:: find_nex_greater_wave
.. autofunction:: find_nex_smaller_wave

"""

import cv2
import warnings
import numpy as np
from skimage.feature import canny


class RegistrationError(Exception):
    """Custom exception for registration errors."""

    pass


def find_nex_greater_wave(waves, wave_1: int, maximum_deviation: int = 5) -> int:
    """
    Finds the next greater wave value in a list of waves within a specified deviation.

    This function identifies the smallest wave value greater than the specified `wave_1`
    within a range defined by `maximum_deviation`. If no such value exists, it returns -1.

    :param waves: A list of integers representing the available wave values.
    :type waves: list[int]
    :param wave_1: The starting wave value to find the next greater wave for.
    :type wave_1: int
    :param maximum_deviation: The maximum deviation from `wave_1` to consider.
    :type maximum_deviation: int
    :returns: The next greater wave value within the deviation range, or -1 if no such value exists.
    :rtype: int
    """

    wave_next = -1

    for n in range(maximum_deviation):
        wave_n = wave_1 + n

        if wave_n in waves:
            wave_next = wave_n
            break

    return wave_next


def find_nex_smaller_wave(waves, wave_1: int, maximum_deviation: int = 5) -> int:
    """
    Finds the next smaller wave value in a list of waves within a specified deviation.

    This function identifies the largest wave value smaller than the specified `wave_1`
    within a range defined by `maximum_deviation`. If no such value exists, it returns -1.

    :param waves: A list of integers representing the available wave values.
    :type waves: list[int]
    :param wave_1: The starting wave value to find the next smaller wave for.
    :type wave_1: int
    :param maximum_deviation: The maximum deviation from `wave_1` to consider.
    :type maximum_deviation: int
    :returns: The next smaller wave value within the deviation range, or -1 if no such value exists.
    :rtype: int
    """
    
    wave_next = -1

    for n in range(maximum_deviation):
        wave_n = wave_1 - n

        if wave_n in waves:
            wave_next = wave_n
            break

    return wave_next


def normalize_spec(spec):
    """Normalize the spectrum to the range 0-1 if needed."""
    spec_min, spec_max = spec.min(), spec.max()
    return np.clip((spec - spec_min) / (spec_max - spec_min), 0, 1) if spec_max > spec_min else spec


def feature_regestration(o_img: np.ndarray, a_img: np.ndarray, max_features: int = 5000, match_percent: float = 0.1):
    """
    Perform a feature-based registration of two grayscale-images.

    The aligned image as well as the used homography are returned.

    :param o_img: 2D np.ndarray of the reference image
    :param a_img: 2D np.ndarray of the moving image
    :param max_features: Int value of the maximum number of keypoint regions
    :param match_percent: Float percentage of keypoint matches to consider
    :return: Tuple of arrays which define the aligned image as well as the used homography
    """
    orb = cv2.ORB_create(max_features)

    if o_img.dtype != np.uint8:
        o_img = (o_img - o_img.min()) / (o_img.max() - o_img.min())
        o_img = (o_img * 255).astype(np.uint8)

    if a_img.dtype != np.uint8:
        a_img = (a_img - a_img.min()) / (a_img.max() - a_img.min())
        a_img = (a_img * 255).astype(np.uint8)

    a_img_key, a_img_descr = orb.detectAndCompute(a_img, None)
    o_img_key, o_img_descr = orb.detectAndCompute(o_img, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(a_img_descr, o_img_descr, None)

    matches = list(matches)
    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance)

    # Remove not so good matches
    num_good_matches = int(len(matches) * match_percent)
    matches = matches[: num_good_matches]

    # Extract location of good matches
    a_points = np.zeros((len(matches), 2), dtype=np.float32)
    o_points = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        a_points[i, :] = a_img_key[match.queryIdx].pt
        o_points[i, :] = o_img_key[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(a_points, o_points, cv2.RANSAC)

    # Use homography
    height, width = o_img.shape
    aligned_img = cv2.warpPerspective(a_img, h, (width, height))

    return aligned_img, h


def feature_registration(o_img: np.ndarray, a_img: np.ndarray, max_features: int = 5000, match_percent: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform ORB-based feature registration.

    Aligns `a_img` to `o_img` using ORB features and RANSAC
    to find the homography matrix.

    Parameters
    ----------
    o_img : numpy.ndarray
        The target (reference) image.
    a_img : numpy.ndarray
        The source image to be aligned.
    max_features : int, optional
        Maximum number of ORB features to detect, defaults to 5000.
    match_percent : float, optional
        Percentage of best matches to use for homography, defaults to 0.1.

    Returns
    -------
    aligned_img : numpy.ndarray
        The `a_img` warped to align with `o_img`.
    H : numpy.ndarray
        The 3x3 homography matrix mapping points from `a_img` to `o_img`.

    Raises
    ------
    RegistrationError
        If no descriptors are found, not enough good matches are found,
        or homography estimation fails.
    """
    def to_uint8(im: np.ndarray) -> np.ndarray:
        """Convert image to uint8, normalizing if necessary."""
        if im.dtype != np.uint8:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in true_divide")
                warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
                min_val, max_val = im.min(), im.max()
                if min_val == max_val:
                    im_norm = np.zeros_like(im, dtype=np.float32)
                else:
                    im_norm = (im.astype(np.float32) - min_val) / (max_val - min_val)
                im_norm = np.nan_to_num(im_norm, nan=0.0, posinf=1.0, neginf=0.0)
            im = (im_norm * 255).astype(np.uint8)
        return im

    o8, a8 = to_uint8(o_img), to_uint8(a_img)
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(a8, None)
    kp2, des2 = orb.detectAndCompute(o8, None)

    if des1 is None or des2 is None:
        raise RegistrationError("No descriptors found")

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    n_good = max(4, int(len(matches) * match_percent))
    matches = matches[:n_good]

    if len(matches) < 4:
        raise RegistrationError(f"Not enough good matches found ({len(matches)}/{n_good})")

    ptsA = np.float32([kp1[m.queryIdx].pt for m in matches])
    ptsO = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(ptsA, ptsO, cv2.RANSAC)
    if H is None:
        raise RegistrationError("Homography estimation failed")

    H = H / H[2, 2]
    h_o, w_o = o_img.shape
    aligned = cv2.warpPerspective(a_img, H, (w_o, h_o), flags=cv2.INTER_LINEAR)
    return aligned, H


def _process_slice(spec_out_flat: np.ndarray, spikes_flat: np.ndarray, idx: int, window: int) -> tuple:
    """
    Process a single slice to remove spikes.

    Replaces spikes with the mean of neighboring values within a given window.

    Parameters
    ----------
    spec_out_flat : numpy.ndarray
        Flattened output spectrum data from the data cube.
    spikes_flat : numpy.ndarray
        Flattened boolean array indicating spike detections.
    idx : int
        Index of the current slice to process.
    window : int
        Size of the window for mean calculation.

    Returns
    -------
    tuple
        A tuple containing the index of the processed slice and the
        modified spectrum slice.
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


def decompose_homography(H: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Extract rotation angle and singular values from the 2x2 linear part of H.

    Parameters
    ----------
    H : numpy.ndarray
        The 3x3 homography matrix.

    Returns
    -------
    angle : float
        The rotation angle in degrees.
    S : numpy.ndarray
        The singular values (scales) from the 2x2 affine part of H.
    """
    A = H[:2, :2]
    U, S, Vt = np.linalg.svd(A)
    R = U.dot(Vt)
    angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return angle, S


def auto_canny(img: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """
    Automatic Canny edge detection using median-based thresholds.

    Assumes img is a float in the range [0,1].

    Parameters
    ----------
    img : numpy.ndarray
        Input image (float, range [0,1]).
    sigma : float, optional
        Sigma value for threshold calculation, defaults to 0.33.

    Returns
    -------
    numpy.ndarray
        Binary edge map from Canny detector.
    """
    v = np.median(img)
    lower = max(0.0, (1.0 - sigma) * v)
    upper = min(1.0, (1.0 + sigma) * v)
    return canny(img, low_threshold=lower, high_threshold=upper)
