import numpy as np
from .._utils.decorators import check_limits


def extend_image(img: np.array, extend_x: int, extend_y: int) -> np.array:
    """
    Extend image by 2 * extend_x and 2 * extend_y.

    Extend_x/y adds zero borders in the given size.

    :param img: The input image.
    :param extend_x: The extension size along the x-axis.
    :param extend_y: The extension size along the y-axis.
    :return: Extended image.
    :rtype: np.array
    """
    x, y, z = img.shape

    new_img = np.zeros(
        shape=(x + extend_x * 2, y + extend_y * 2, z),
        dtype=img.dtype
    )
    new_img[extend_x:-extend_x, extend_y:-extend_y] = img

    return new_img


def decrease_image(img: np.array, decrease_x: int, decrease_y: int) -> np.array:
    """
    Decrease image by 2 * decrease_x and 2 * decrease_y.

    Decrease_x/y removes borders from the given size.

    :param img: The input image.
    :param decrease_x: The decrease size along the x-axis.
    :param decrease_y: The decrease size along the y-axis.
    :return: Decreased image.
    :rtype: np.array
    """
    return img[decrease_x:-decrease_x, decrease_y:-decrease_y]


def get_output_size(image_lengths: int, filter_lengths: int, stride: int) -> int:
    """
    Calculate length of the feature map.

    The function should return an integer value.
    If the math is not possible, the function returns 0.

    :param image_lengths: The length of one image side.
    :type image_lengths: int
    :param filter_lengths: The length of one side of the filter.
    :type filter_lengths: int
    :param stride: Filter stride.
    :type stride: int
    :return: Feature lengths as an integer.
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
    Generate a feature map by stepping over the image with a given filter function.

    :param img: The input image.
    :param filter: The filter to apply.
    :param padding: Padding type.
    :param stride_x: Stride along the x-axis.
    :param stride_y: Stride along the y-axis.
    :return: Feature map.
    """
    if stride_x == 0:
        raise ValueError('step_x cant be 0 (zero)')
    elif stride_y == 0:
        raise ValueError('step_y cant be 0 (zero)')

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
    Convert RGB image to grayscale using average values.

    Color/3 to avoid conflicts with uint8 images.

    :param image: The RGB image.
    :return: Grayscale image.
    """
    return image[:, :, 0] / 3 + image[:, :, 1] / 3 + image[:, :, 2] / 3


@check_limits
def rgb_to_grayscale_wight(image: np.array, r_weight: float = 0.299, g_weight: float = 0.587,
                           b_weight: float = 0.114) -> np.array:
    """
    Convert RGB image to grayscale using weighted average.

    :param image: The RGB image.
    :param r_weight: Weight for adjusting red color.
    :param g_weight: Weight for adjusting green color.
    :param b_weight: Weight for adjusting blue value.
    :return: Grayscale image.
    """
    return image[:, :, 0] * r_weight + image[:, :, 1] * g_weight + image[:, :, 2] * b_weight


@check_limits
def brightness(image: np.array, delta: int) -> np.array:
    """
    Adjust brightness of an image.

    :param image: The input image.
    :param delta: The delta value for brightness adjustment.
    :return: Brightness adjusted image.
    """
    tmp = image + delta
    tmp[tmp < delta] = 255
    return tmp


@check_limits
def contrast(image: np.array, beta: int) -> np.array:
    """
    Adjust contrast of an image.

    :param image: The input image.
    :param beta: The beta value for contrast adjustment.
    :return: Contrast adjusted image.
    """
    u = np.mean(image, axis=2)
    u_mean = u.mean()

    if beta == 255:
        alpha = np.infty
    else:
        alpha = (255 + beta) / (255 - beta)

    image = ((image[:, :] - u_mean) * alpha + u_mean).astype('int')
    return image
