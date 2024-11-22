# Generated by CodiumAI

from wizard._utils._loader._helper import get_files_by_extension, make_path_absolute, to_cube
from wizard import read
import wizard

from wizard._utils.decorators import check_path
from wizard._utils.decorators import track_execution_time
from wizard._utils.decorators import check_limits
from wizard._utils.decorators import add_method

from wizard._core.datacube import DataCube


import os
import time
import pytest
import tempfile
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

class TestGetFilesByExtension:

    #  Returns a sorted list of filenames with the given extension from a valid directory.
    def test_valid_directory(self):
        # Arrange
        path = '/path/to/directory'
        extension = '.csv'

        # Act
        result = get_files_by_extension(path, extension)

        # Assert
        assert isinstance(result, list)
        assert all(isinstance(filename, str) for filename in result)
        assert all(filename.endswith(extension) for filename in result)
        assert sorted(result) == result

    #  Returns an empty list if the extension is not provided.
    def test_no_extension_provided(self):
        # Arrange
        path = '/path/to/directory'
        extension = ''

        # Act
        result = get_files_by_extension(path, extension)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 0

    #  Accepts a valid directory path with a trailing slash.
    def test_directory_with_trailing_slash(self):
        # Arrange
        path = '/path/to/directory/'
        extension = '.csv'

        # Act
        result = get_files_by_extension(path, extension)

        # Assert
        assert isinstance(result, list)
        assert all(isinstance(filename, str) for filename in result)
        assert all(filename.endswith(extension) for filename in result)
        assert sorted(result) == result


class TestMakePathAbsolute:

    #  The function receives a valid absolute path as input and returns the same path.
    def test_valid_absolute_path(self):
        # Arrange
        path = "/home/user/file.txt"

        # Act
        result = make_path_absolute(path)

        # Assert
        assert result == path

    #  The function receives an empty string as input and raises a ValueError.
    def test_empty_string_input(self):
        # Arrange
        path = ""

        # Act and Assert
        with pytest.raises(ValueError):
            make_path_absolute(path)

    #  The function receives a non-string input and raises a ValueError.
    def test_non_string_input(self):
        # Arrange
        path = 123

        # Act and Assert
        with pytest.raises(ValueError):
            make_path_absolute(path)

class TestToCube:

    #  Should transform a 2d numpy array to a 3d numpy array with the correct shape
    def test_transform_2d_to_3d(self):
        # Arrange
        data = np.random.rand(6,6)
        len_x = 3
        len_y = 3

        # Act
        result = to_cube(data, len_x, len_y)

        # Assert
        assert result.shape == (4, 3, 3)

class TestCheckPath:

    #  Function called with valid path
    def test_valid_path(self):
        @check_path
        def dummy_func(path):
            return True

        assert dummy_func(VALID_PATH) == True

    #  Function called with valid path and additional arguments
    def test_valid_path_with_args(self):
        @check_path
        def dummy_func(path, arg1, arg2):
            return True

        assert dummy_func(VALID_PATH, 'arg1', 'arg2') == True

    #  Function called with valid path and keyword argument
    def test_valid_path_with_kwarg(self):
        @check_path
        def dummy_func(path, kwarg=None):
            return True

        assert dummy_func(VALID_PATH, kwarg='value') == True

    #  Function called with valid path and multiple keyword arguments
    def test_valid_path_with_multiple_kwargs(self):
        @check_path
        def dummy_func(path, kwarg1=None, kwarg2=None):
            return True

        assert dummy_func(VALID_PATH, kwarg1='value1', kwarg2='value2') == True

    #  Function called with empty string path
    def test_empty_string_path(self):
        @check_path
        def dummy_func(path):
            return True

        with pytest.raises(ValueError):
            dummy_func('')

    #  Function called with non-existent path
    def test_nonexistent_path(self):
        @check_path
        def dummy_func(path):
            return True

        with pytest.raises(FileNotFoundError):
            dummy_func('/path/to/nonexistent')

class TestCheckLimits:

    #  The function receives an image with all values equal to the upper limit and returns the same image.
    def test_all_values_equal_to_upper_limit_returns_same_image(self):
        # Arrange
        image = np.array([[1, 1], [1, 1]], dtype='float32')

        @check_limits
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

        @check_limits
        def dummy_func(image):
            return image

        # Act
        result = dummy_func(image)

        # Assert
        assert np.array_equal(result, image)

    #  The function receives an empty image and returns an empty image.
    def test_empty_image_returns_empty_image(self):
        # Arrange
        image = np.array([], dtype='float32')

        @check_limits
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

        @check_limits
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

        @check_limits
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

        @check_limits
        def dummy_func(image):
            return image

        # Act
        result = dummy_func(image)

        # Assert
        assert np.array_equal(result, np.array([[1, 1], [1, 1]], dtype='float32'))


class TestAddMethod:

    #  The decorator function should return a function.
    def test_decorator_returns_function(self):

        class MyClass:
            pass

        @add_method(MyClass)
        def my_method():
            return None

        assert callable(MyClass.my_method)

class TestLoader:

    def test_read_write_nrrd(self, sample_data_cube):

        from wizard._utils._loader import nrrd

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.nrrd', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Write the DataCube to the temporary file
            nrrd._write_nrrd(dc=sample_data_cube, path=temp_path)

            # Read the DataCube from the temporary file
            loaded_data_cube = nrrd._read_nrrd(path=temp_path)

            # Assertions to ensure the loaded data matches the original data
            np.testing.assert_array_almost_equal(loaded_data_cube.cube, sample_data_cube.cube)
            np.testing.assert_array_equal(loaded_data_cube.wavelengths, sample_data_cube.wavelengths)
            assert loaded_data_cube.name == sample_data_cube.name
            assert loaded_data_cube.notation == sample_data_cube.notation
            assert loaded_data_cube.record == sample_data_cube.record

        finally:
            # Clean up the temporary file
            os.remove(temp_path)

    def test_read_write_pickle(self, sample_data_cube):

        from wizard._utils._loader import pickle

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Write the DataCube to the temporary file
            pickle._write_pickle(data=sample_data_cube, path=temp_path)

            # Read the DataCube from the temporary file
            loaded_data_cube = pickle._read_pickle(path=temp_path)
            print(sample_data_cube.cube.shape)
            print(loaded_data_cube.cube.shape)
            print(loaded_data_cube.wavelengths)

            # Assertions to ensure the loaded data matches the original data
            # np.testing.assert_array_almost_equal(loaded_data_cube.cube, sample_data_cube.cube)
            np.testing.assert_array_equal(loaded_data_cube.wavelengths, sample_data_cube.wavelengths)

        finally:
            # Clean up the temporary file
            os.remove(temp_path) 


class TestHelper:
    def test_find_nex_greater_wave_within_deviation(self):
        waves = [100, 102, 104, 108]
        wave_1 = 101
        maximum_deviation = 5
        result = wizard._utils.helper.find_nex_greater_wave(waves, wave_1, maximum_deviation)
        assert result == 102, f"Expected 102, got {result}"

    def test_find_nex_greater_wave_outside_deviation(self):
        waves = [100, 102, 104, 108]
        wave_1 = 101
        maximum_deviation = 1
        result = wizard._utils.helper.find_nex_greater_wave(waves, wave_1, maximum_deviation)
        assert result == -1, f"Expected -1, got {result}"

    def test_find_nex_greater_wave_exact_match(self):
        waves = [100, 101, 102]
        wave_1 = 101
        result = wizard._utils.helper.find_nex_greater_wave(waves, wave_1)
        assert result == 101, f"Expected 101, got {result}"

    def test_find_nex_greater_wave_no_valid_wave(self):
        waves = [100, 102, 104, 106]
        wave_1 = 110
        result = wizard._utils.helper.find_nex_greater_wave(waves, wave_1)
        assert result == -1, f"Expected -1, got {result}"

    def test_find_nex_smaller_wave_within_deviation(self):
        waves = [100, 102, 104, 108]
        wave_1 = 105
        maximum_deviation = 5
        result = wizard._utils.helper.find_nex_smaller_wave(waves, wave_1, maximum_deviation)
        assert result == 104, f"Expected 104, got {result}"

    def test_find_nex_smaller_wave_outside_deviation(self):
        waves = [100, 102, 104, 108]
        wave_1 = 103
        maximum_deviation = 1
        result = wizard._utils.helper.find_nex_smaller_wave(waves, wave_1, maximum_deviation)
        assert result == -1, f"Expected -1, got {result}"

    def test_find_nex_smaller_wave_exact_match(self):
        waves = [99, 101, 103]
        wave_1 = 101
        result = wizard._utils.helper.find_nex_smaller_wave(waves, wave_1)
        assert result == 101, f"Expected 101, got {result}"

    def test_find_nex_smaller_wave_no_valid_wave(self):
        waves = [100, 102, 104, 108]
        wave_1 = 90
        result = wizard._utils.helper.find_nex_smaller_wave(waves, wave_1)
        assert result == -1, f"Expected -1, got {result}"

class TestDecorators:

    def test_check_load_dc_valid(self, sample_data_cube):
        from wizard._utils.decorators import check_load_dc

        @check_load_dc
        def mock_loader():
            return sample_data_cube

        result = mock_loader()
        assert isinstance(result, DataCube)
        assert result.cube.shape == (10, 11, 12)

    def test_check_load_dc_invalid_return_type(self):
        from wizard._utils.decorators import check_load_dc

        @check_load_dc
        def mock_loader():
            return "invalid return type"

        with pytest.raises(ValueError, match='Loading function should return a DataCube'):
            mock_loader()

    def test_check_load_dc_invalid_shape(self):
        from wizard._utils.decorators import check_load_dc

        @check_load_dc
        def mock_loader():
            data = np.random.rand(1, 100)  # Invalid shape
            return DataCube(cube=data, wavelengths=[], name="InvalidCube", notation="nm", record=False)

        with pytest.raises(ValueError, match='The return shape should be \\(v\\|x\\|y\\).'):
            mock_loader()

    def test_check_path_valid_path(self, tmp_path):
        from wizard._utils.decorators import check_path

        valid_path = tmp_path / "test.txt"
        valid_path.touch()  # Create the file

        @check_path
        def mock_function(path):
            return path

        assert mock_function(path=str(valid_path)) == str(valid_path)

    def test_check_path_no_path_provided(self):
        from wizard._utils.decorators import check_path

        @check_path
        def mock_function(path=None):
            return path

        with pytest.raises(ValueError, match='No path provided.'):
            mock_function()

    def test_check_path_invalid_path(self):
        from wizard._utils.decorators import check_path

        @check_path
        def mock_function(path):
            return path

        with pytest.raises(FileNotFoundError, match='Invalid path: .*'):
            mock_function(path="nonexistent_path.txt")

    def test_add_method(self):
        from wizard._utils.decorators import add_method

        class MyClass:
            pass

        @add_method(MyClass)
        def new_method(self):
            return "method added"

        instance = MyClass()
        assert instance.new_method() == "method added"

    def test_track_execution_time(self, capsys):
        from wizard._utils.decorators import track_execution_time

        @track_execution_time
        def slow_function():
            time.sleep(0.1)
            return "done"

        result = slow_function()
        captured = capsys.readouterr()
        assert "Function 'slow_function' executed in" in captured.out
        assert result == "done"

    def test_check_limits_clips_values(self):
        from wizard._utils.decorators import check_limits

        @check_limits
        def process_image(image):
            return image * 2  # Exaggerate to go beyond limits

        image = np.array([0.5, 0.7, 1.5, -0.5], dtype=np.float32)
        result = process_image(image)
        np.testing.assert_array_equal(result, np.clip(image, 0, 1))

    def test_add_to_workflow(self, capsys):
        from wizard._utils.decorators import add_to_workflow

        @add_to_workflow
        def sample_function(arg1, arg2):
            print(f"{arg1}, {arg2}")
            return "workflow added"

        result = sample_function("arg1_value", "arg2_value")
        captured = capsys.readouterr()
        assert "arg1_value, arg2_value" in captured.out
        assert result == "workflow added"