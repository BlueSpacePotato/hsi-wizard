




import pytest
import numpy as np
from wizard._processing.image import (
    extend_image,
    decrease_image,
    get_output_size,
    feature_map,
    rgb_to_grayscale_average,
    rgb_to_grayscale_wight,
    brightness,
    contrast,
)

class TestProcessingImage:

    # Test extend_image function
    def test_extend_image_basic(self):
        img = np.ones((5, 5, 3))
        extended_img = extend_image(img, 1, 1)
        assert extended_img.shape == (7, 7, 3)
        assert np.array_equal(extended_img[1:-1, 1:-1], img)

    def test_extend_image_zero_extension(self):
        img = np.ones((5, 5, 3))
        with pytest.raises(ValueError):
            extended_img = extend_image(img, 0, 0)

    def test_extend_image_negative(self):
        img = np.ones((5, 5, 3))
        with pytest.raises(ValueError):
            extend_image(img, -1, -1)

    # Test decrease_image function
    def test_decrease_image_basic(self):
        img = np.ones((5, 5, 3))
        decreased_img = decrease_image(img, 1, 1)
        assert decreased_img.shape == (3, 3, 3)

    def test_decrease_image_zero_decrease(self):
        img = np.ones((5, 5, 3))
        decreased_img = decrease_image(img, 0, 0)
        assert np.array_equal(decreased_img, img)

    def test_decrease_image_large_decrease(self):
        img = np.ones((5, 5, 3))
        with pytest.raises(ValueError):
            decrease_image(img, 6, 3)  # More decrease than image size

    # Test get_output_size function
    def test_get_output_size_basic(self):
        assert get_output_size(10, 3, 1) == 8
        assert get_output_size(10, 3, 2) == 0

    def test_get_output_size_division_zero(self):
        assert get_output_size(10, 3, 4) == 0  # Feature map size would be fractional

    # Test feature_map function
    def test_feature_map_basic(self):
        img = np.random.rand(5, 5, 3)
        filt = np.ones((3, 3, 3))
        fmap = feature_map(img, filt)
        assert fmap is not None

    def test_feature_map_invalid_stride(self):
        img = np.random.rand(5, 5, 3)
        filt = np.ones((3, 3, 3))
        with pytest.raises(ValueError):
            feature_map(img, filt, stride_x=0, stride_y=1)

    # Test rgb_to_grayscale_average function
    def test_rgb_to_grayscale_average(self):
        img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        grayscale = rgb_to_grayscale_average(img)
        assert grayscale.shape == (1, 3)

    # Test rgb_to_grayscale_wight function
    def test_rgb_to_grayscale_wight(self):
        img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        grayscale = rgb_to_grayscale_wight(img)
        assert grayscale.shape == (1, 3)

    # Test brightness function
    def test_brightness_basic(self):
        img = np.ones((5, 5, 3)) * 100
        brightened_img = brightness(img, 50)
        assert np.all(brightened_img <= 255)  # Ensure values are capped at 255

    # Test contrast function
    def test_contrast_basic(self):
        img = np.ones((5, 5, 3)) * 100
        contrast_img = contrast(img, 50)
        assert contrast_img.shape == img.shape

