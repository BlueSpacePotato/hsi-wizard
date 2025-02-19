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
import numpy as np


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


def normalize(spec):
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
