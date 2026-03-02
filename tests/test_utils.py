from wizard._utils import helper, decorators

from wizard._core.datacube import DataCube


import time
import pytest
import numpy as np

VALID_PATH = '.'

@pytest.fixture
def sample_data_cube():
    # Define a sample DataCube for testing
    data = np.random.rand(10, 11, 12)  # 3D array as a placeholder for data cube
    wavelengths = [i for i in range(data.shape[0])]       # Example wavelengths
    name = "TestCube"
    notation = "nm"
    record = False
    return DataCube(cube=data, wavelengths=wavelengths, name=name, notation=notation, record=record)

# Simulate the content of an FSM file
@pytest.fixture
def mock_fsm_file():
    fsm_file_content = b'\x00\x00\x00\x00Description of FSM file'  # Simulated header
    fsm_file_content += b'\x00\x00\x00\x00'  # Simulated metadata
    fsm_file_content += b'\x00\x00\x00\x00'  # Simulated block with size 0
    fsm_file_content += b'\x00\x00\x00\x00'  # Simulated block with size 0
    fsm_file_content += b'\x00\x00\x00\x00'  # More data or spectrum
    return fsm_file_content


# Test case for non-existing file
@pytest.fixture
def non_existing_file():
    return "non_existing_file.tdms"



class TestHelperFindNextGreaterWave:
    def test_find_next_greater_wave_valid(self):
        """
        Test with a valid wave list where a greater wave exists within the deviation.
        """
        waves = [100, 105, 110, 115]
        wave_1 = 102
        maximum_deviation = 10

        result = helper.find_nex_greater_wave(waves, wave_1, maximum_deviation)
        assert result == 105, f"Expected 105, but got {result}"

    def test_find_next_greater_wave_no_match(self):
        """
        Test with a wave list where no greater wave exists within the deviation.
        """
        waves = [100, 105, 110, 115]
        wave_1 = 120
        maximum_deviation = 5

        result = helper.find_nex_greater_wave(waves, wave_1, maximum_deviation)
        assert result == -1, f"Expected -1, but got {result}"

    def test_find_next_greater_wave_exact_match(self):
        """
        Test with a wave list where the next wave is exactly at the starting value.
        """
        waves = [100, 105, 110, 115]
        wave_1 = 105
        maximum_deviation = 5

        result = helper.find_nex_greater_wave(waves, wave_1, maximum_deviation)
        assert result == 105, f"Expected 105, but got {result}"

    def test_find_next_greater_wave_empty_list(self):
        """
        Test with an empty wave list.
        """
        waves = []
        wave_1 = 100
        maximum_deviation = 5

        result = helper.find_nex_greater_wave(waves, wave_1, maximum_deviation)
        assert result == -1, f"Expected -1, but got {result}"

    def test_find_next_greater_wave_zero_deviation(self):
        """
        Test with zero maximum deviation.
        """
        waves = [100, 105, 110, 115]
        wave_1 = 100
        maximum_deviation = 0

        result = helper.find_nex_greater_wave(waves, wave_1, maximum_deviation)
        assert result == -1, f"Expected -1, but got {result}"

    def test_find_next_greater_wave_negative_deviation(self):
        """
        Test with a negative maximum deviation (should behave the same as zero deviation).
        """
        waves = [100, 105, 110, 115]
        wave_1 = 100
        maximum_deviation = -5

        result = helper.find_nex_greater_wave(waves, wave_1, maximum_deviation)
        assert result == -1, f"Expected -1, but got {result}"

class TestFindNextSmallerWave:
    def test_find_next_smaller_wave_valid(self):
        """
        Test with a valid wave list where a smaller wave exists within the deviation.
        """
        waves = [90, 95, 100, 105]
        wave_1 = 102
        maximum_deviation = 10

        result = helper.find_nex_smaller_wave(waves, wave_1, maximum_deviation)
        assert result == 100, f"Expected 100, but got {result}"

    def test_find_next_smaller_wave_no_match(self):
        """
        Test with a wave list where no smaller wave exists within the deviation.
        """
        waves = [90, 95, 100, 105]
        wave_1 = 85
        maximum_deviation = 5

        result = helper.find_nex_smaller_wave(waves, wave_1, maximum_deviation)
        assert result == -1, f"Expected -1, but got {result}"

    def test_find_next_smaller_wave_exact_match(self):
        """
        Test with a wave list where the next smaller wave is exactly at the starting value.
        """
        waves = [90, 95, 100, 105]
        wave_1 = 100
        maximum_deviation = 5

        result = helper.find_nex_smaller_wave(waves, wave_1, maximum_deviation)
        assert result == 100, f"Expected 100, but got {result}"

class TestDecorators:

    #  Function called with valid path
    def test_valid_path(self):
        @decorators.check_path
        def dummy_func(path):
            return True

        assert dummy_func(VALID_PATH) == True

    #  Function called with valid path and additional arguments
    def test_valid_path_with_args(self):
        @decorators.check_path
        def dummy_func(path, arg1, arg2):
            return True

        assert dummy_func(VALID_PATH, 'arg1', 'arg2') == True

    #  Function called with valid path and keyword argument
    def test_valid_path_with_kwarg(self):
        @decorators.check_path
        def dummy_func(path, kwarg=None):
            return True

        assert dummy_func(VALID_PATH, kwarg='value') == True

    #  Function called with valid path and multiple keyword arguments
    def test_valid_path_with_multiple_kwargs(self):
        @decorators.check_path
        def dummy_func(path, kwarg1=None, kwarg2=None):
            return True

        assert dummy_func(VALID_PATH, kwarg1='value1', kwarg2='value2') == True

    #  Function called with empty string path
    def test_empty_string_path(self):
        @decorators.check_path
        def dummy_func(path):
            return True

        with pytest.raises(ValueError):
            dummy_func('')

    #  Function called with non-existent path
    def test_nonexistent_path(self):
        @decorators.check_path
        def dummy_func(path):
            return True

        with pytest.raises(FileNotFoundError):
            dummy_func('/path/to/nonexistent')

    #  The function receives an image with all values equal to the upper limit and returns the same image.
    def test_all_values_equal_to_upper_limit_returns_same_image(self):
        # Arrange
        image = np.array([[1, 1], [1, 1]], dtype='float32')

        @decorators.check_limits
        def dummy_func(image):
            return image

        # Act
        result = dummy_func(image)

        # Assert
        assert np.array_equal(result, image)

    #  The function receives an image with all values equal to the lower limit and returns the same image.
    def test_all_values_equal_to_lower_limit_returns_same_image(self):
        # Arrange
        image = np.array([[0, 0], [0, 0]], dtype='float32')

        @decorators.check_limits
        def dummy_func(image):
            return image

        # Act
        result = dummy_func(image)

        # Assert
        assert np.array_equal(result, image)

    #  The function receives an image with a single value and returns the same image.
    def test_single_value_image_returns_same_image(self):
        # Arrange
        image = np.array([100], dtype='float32')

        @decorators.check_limits
        def dummy_func(image):
            return image

        # Act
        result = dummy_func(image)

        # Assert
        assert np.array_equal(result, [1.])

    #  The function receives an image with negative values and returns the clipped image.
    def test_negative_values_returns_clipped_image(self):
        # Arrange
        image = np.array([[-100, -200], [150, 250]], dtype='float32')

        @decorators.check_limits
        def dummy_func(image):
            return image

        # Act
        result = dummy_func(image)

        # Assert
        assert np.array_equal(result, np.array([[0., 0.], [1., 1.]], dtype='float32'))

    #  The function receives an image with values above the upper limit and returns the clipped image.
    def test_values_above_upper_limit_returns_clipped_image(self):
        # Arrange
        image = np.array([[300, 400], [500, 600]], dtype='float32')

        @decorators.check_limits
        def dummy_func(image):
            return image

        # Act
        result = dummy_func(image)

        # Assert
        assert np.array_equal(result, np.array([[1, 1], [1, 1]], dtype='float32'))


    #  The decorator function should return a function.
    def test_decorator_returns_function(self):

        class MyClass:
            pass

        @decorators.add_method(MyClass)
        def my_method():
            return None

        assert callable(MyClass.my_method)

    def test_check_load_dc_valid(self, sample_data_cube):

        @decorators.check_load_dc
        def mock_loader():
            return sample_data_cube

        result = mock_loader()
        assert isinstance(result, DataCube)
        assert result.cube.shape == (10, 11, 12)

    def test_check_load_dc_invalid_return_type(self):

        @decorators.check_load_dc
        def mock_loader():
            return "invalid return type"

        with pytest.raises(ValueError, match='Loading function should return a DataCube'):
            mock_loader()

    def test_check_load_dc_invalid_shape(self):

        @decorators.check_load_dc
        def mock_loader():
            data = np.random.rand(1, 100)  # Invalid shape
            return DataCube(cube=data, wavelengths=[], name="InvalidCube", notation="nm", record=False)

        with pytest.raises(ValueError, match='The return shape should be \\(v\\|x\\|y\\).'):
            mock_loader()

    def test_check_path_valid_path(self, tmp_path):

        valid_path = tmp_path / "test.txt"
        valid_path.touch()  # Create the file

        @decorators.check_path
        def mock_function(path):
            return path

        assert mock_function(path=str(valid_path)) == str(valid_path)

    def test_check_path_no_path_provided(self):

        @decorators.check_path
        def mock_function(path=None):
            return path

        with pytest.raises(ValueError, match='No path provided.'):
            mock_function()

    def test_check_path_invalid_path(self):

        @decorators.check_path
        def mock_function(path):
            return path

        with pytest.raises(FileNotFoundError, match='Invalid path: .*'):
            mock_function(path="nonexistent_path.txt")

    def test_add_method(self):

        class MyClass:
            pass

        @decorators.add_method(MyClass)
        def new_method(self):
            return "method added"

        instance = MyClass()
        assert instance.new_method() == "method added"

    def test_track_execution_time(self, capsys):

        @decorators.track_execution_time
        def slow_function():
            time.sleep(0.1)
            return "done"

        result = slow_function()
        captured = capsys.readouterr()
        assert "Function 'slow_function' executed in" in captured.out
        assert result == "done"

    def test_check_limits_clips_values(self):

        @decorators.check_limits
        def process_image(image):
            return image * 2  # Exaggerate to go beyond limits

        image = np.array([0.5, 0.7, 1.5, -0.5], dtype=np.float32)
        result = process_image(image)
        np.testing.assert_array_equal(result, np.clip(image*2, 0, 1))




class TestFeatureRegistration:
    def test_feature_registration_identity(self):
        """
        Test feature registration with identical images.
        """
        # Create a dummy grayscale image
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        # Call the feature registration function
        aligned_img, homography = helper.feature_regestration(img, img)

        # Assert that the aligned image is identical to the input
        assert np.array_equal(aligned_img, img), "Aligned image should be identical to the input image for identical inputs."

        # Assert that the homography is close to the identity matrix
        np.testing.assert_array_almost_equal(homography, np.eye(3), decimal=5, err_msg="Homography should be close to the identity matrix for identical inputs.")

    def test_feature_registration_different_images(self):
        """
        Test feature registration with slightly different images.
        """
        # Create two dummy grayscale images
        img1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img2 = np.roll(img1, shift=5, axis=1)  # Shifted version of img1

        # Call the feature registration function
        aligned_img, homography = helper.feature_regestration(img1, img2)

        # Assert that the aligned image is not None
        assert aligned_img is not None, "Aligned image should not be None."

        # Assert that the homography is not None
        assert homography is not None, "Homography should not be None."

    def test_feature_registration_invalid_input(self):
        """
        Test feature registration with invalid inputs.
        """
        # Create dummy invalid inputs
        img1 = np.random.randint(0, 256, (100, 100)).astype('float32')  # Float32 instead of uint8
        img2 = np.random.randint(0, 256, (100, 100)).astype('uint8')

        # Call the feature registration function
        aligned_img, h = helper.feature_regestration(img1, img2)

        # Assert that the aligned image is not None
        assert aligned_img is not None, "Aligned image should not be None."

        # Assert that the homography is not None
        assert h is not None, "Homography should not be None."
