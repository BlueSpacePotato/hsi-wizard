"""
_processing/image.py
=========================

.. module:: image
   :platform: Unix
   :synopsis: Provides various methods for image manipulation such as extension, filtering, and color conversions.

Module Overview
---------------

This module contains functions to perform a variety of image processing tasks, including extending and decreasing image size,
generating feature maps, and adjusting brightness and contrast. Functions for converting RGB images to grayscale are also provided.


Functions
---------

.. autofunction:: extend_image
.. autofunction:: decrease_image
.. autofunction:: get_output_size
.. autofunction:: feature_map
.. autofunction:: rgb_to_grayscale_average
.. autofunction:: rgb_to_grayscale_wight
.. autofunction:: brightness
.. autofunction:: contrast

"""

import numpy as np
from .._utils.decorators import check_limits


def extend_image(img: np.array, extend_x: int, extend_y: int) -> np.array:
    """
    Extend the image by 2 * extend_x and 2 * extend_y.

    Adds zero-padding borders of specified size around the image.

    :param img: The input image.
    :param extend_x: The extension size along the x-axis.
    :param extend_y: The extension size along the y-axis.
    :return: The extended image.
    :rtype: np.array
    """
    x, y, z = img.shape

    if x == 0 or y == 0:
        raise ValueError('x and y cant be 0.')
    new_img = np.zeros(
        shape=(x + extend_x * 2, y + extend_y * 2, z),
        dtype=img.dtype
    )
    new_img[extend_x:-extend_x, extend_y:-extend_y] = img

    return new_img


def decrease_image(img: np.array, decrease_x: int, decrease_y: int) -> np.array:
    """
    Decrease the image by 2 * decrease_x and 2 * decrease_y.

    Removes borders of the specified size from the image.

    :param img: The input image.
    :param decrease_x: The size to decrease along the x-axis.
    :param decrease_y: The size to decrease along the y-axis.
    :return: The decreased image.
    :rtype: np.array
    """
    shape = img.shape
    if decrease_x > shape [0] or decrease_y > shape[1]:
        raise ValueError('Decrease value is greater then the image shape.')
    return img[decrease_x:shape[0]-decrease_x, decrease_y:shape[1]-decrease_y]


def get_output_size(image_lengths: int, filter_lengths: int, stride: int) -> int:
    """
    Calculate the length of the feature map.

    This function computes the output size when applying a filter to an image with a specific stride.
    If the computation results in a non-integer value, it returns 0.

    :param image_lengths: The length of one image side.
    :type image_lengths: int
    :param filter_lengths: The length of one side of the filter.
    :type filter_lengths: int
    :param stride: The stride length of the filter.
    :type stride: int
    :return: The feature map length or 0 if the computation is invalid.
    :rtype: int

    >>> get_output_size(10, 2, 1)
    9

    >>> get_output_size(10, 3, 2)
    0
    """
    feature_lengths = (image_lengths - filter_lengths) / stride + 1
    return int(feature_lengths) if feature_lengths.is_integer() else 0


def feature_map(img: np.array, filter: np.array, padding: str = 'const', stride_x: int = 1, stride_y: int = 1):
    """
    Generate a feature map by applying a filter across the image with a specific stride.

    :param img: The input image.
    :param filter: The filter to apply to the image.
    :param padding: The type of padding to apply (currently not implemented).
    :param stride_x: The stride along the x-axis.
    :param stride_y: The stride along the y-axis.
    :return: The resulting feature map.
    :rtype: np.array
    """
    if stride_x == 0:
        raise ValueError('stride_x cannot be 0')
    elif stride_y == 0:
        raise ValueError('stride_y cannot be 0')

    feature_map_len_x = get_output_size(img.shape[0], filter.shape[0], stride_x)
    feature_map_len_y = get_output_size(img.shape[1], filter.shape[1], stride_y)

    if feature_map_len_x == 0:
        while feature_map_len_x == 0:
            stride_x += 1
            feature_map_len_x = get_output_size(img.shape[0], filter.shape[0], stride_x)
        return None
    if feature_map_len_y == 0:
        while feature_map_len_y == 0:
            stride_y += 1
            feature_map_len_y = get_output_size(img.shape[1], filter.shape[1], stride_y)
        return None

    feature_img = np.zeros(
        shape=(
            feature_map_len_x,
            feature_map_len_y,
            img.shape[2]
        )
    )

    for x in range(feature_map_len_x):
        x1 = x * stride_x
        x2 = x * stride_x + filter.shape[0]
        for y in range(feature_map_len_y):
            y1 = y * stride_y
            y2 = y * stride_y + filter.shape[1]
            mini_img = img[x1:x2, y1:y2]
            pixel = np.sum(mini_img * filter)
            feature_img[x, y] = pixel

    feature_img = feature_img / feature_img.max()

    return feature_img


@check_limits
def rgb_to_grayscale_average(image: np.array) -> np.array:
    """
    Convert an RGB image to grayscale by averaging the color channels.

    Each pixel's value is computed as the average of the red, green, and blue channels.

    :param image: The input RGB image.
    :return: The grayscale image.
    :rtype: np.array
    """
    return image[:, :, 0] / 3 + image[:, :, 1] / 3 + image[:, :, 2] / 3


@check_limits
def rgb_to_grayscale_wight(image: np.array, r_weight: float = 0.299, g_weight: float = 0.587,
                           b_weight: float = 0.114) -> np.array:
    """
    Convert an RGB image to grayscale using weighted color channel values.

    :param image: The input RGB image.
    :param r_weight: Weight for the red channel.
    :param g_weight: Weight for the green channel.
    :param b_weight: Weight for the blue channel.
    :return: The grayscale image.
    :rtype: np.array
    """
    return image[:, :, 0] * r_weight + image[:, :, 1] * g_weight + image[:, :, 2] * b_weight


@check_limits
def brightness(image: np.array, delta: int) -> np.array:
    """
    Adjust the brightness of an image by adding a delta value.

    :param image: The input image.
    :param delta: The value to adjust the brightness.
    :return: The brightness-adjusted image.
    :rtype: np.array
    """
    tmp = image + delta
    tmp[tmp < delta] = 255
    return tmp


@check_limits
def contrast(image: np.array, beta: int) -> np.array:
    """
    Adjust the contrast of an image by applying a beta value.

    The contrast is adjusted by altering the pixel values relative to the mean brightness of the image.

    :param image: The input image.
    :param beta: The contrast adjustment factor.
    :return: The contrast-adjusted image.
    :rtype: np.array
    """
    u = np.mean(image, axis=2)
    u_mean = u.mean()

    if beta == 255:
        alpha = np.infty
    else:
        alpha = (255 + beta) / (255 - beta)

    image = ((image[:, :] - u_mean) * alpha + u_mean).astype('int')
    return image
