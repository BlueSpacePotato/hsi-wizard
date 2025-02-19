# Generated by CodiumAI



import wizard

from wizard._utils import helper, _loader, decorators
from wizard._core.datacube import DataCube


import os
import time
import pytest
import tempfile
import numpy as np
from matplotlib import pyplot as plt

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



class TestLoaderHelper:

    #  Returns a sorted list of filenames with the given extension from a valid directory.
    def test_valid_directory(self):
        # Arrange
        path = '/path/to/directory'
        extension = '.csv'

        # Act
        result = _loader._helper.get_files_by_extension(path, extension)

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
        result = _loader._helper.get_files_by_extension(path, extension)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 0

    #  Accepts a valid directory path with a trailing slash.
    def test_directory_with_trailing_slash(self):
        # Arrange
        path = '/path/to/directory/'
        extension = '.csv'

        # Act
        result = _loader._helper.get_files_by_extension(path, extension)

        # Assert
        assert isinstance(result, list)
        assert all(isinstance(filename, str) for filename in result)
        assert all(filename.endswith(extension) for filename in result)
        assert sorted(result) == result

    #  The function receives a valid absolute path as input and returns the same path.
    def test_valid_absolute_path(self):
        # Arrange
        path = "/home/user/file.txt"

        # Act
        result = _loader._helper.make_path_absolute(path)

        # Assert
        assert result == path

    #  The function receives an empty string as input and raises a ValueError.
    def test_empty_string_input(self):
        # Arrange
        path = ""

        # Act and Assert
        with pytest.raises(ValueError):
            _loader._helper.make_path_absolute(path)

    #  The function receives a non-string input and raises a ValueError.
    def test_non_string_input(self):
        # Arrange
        path = 123

        # Act and Assert
        with pytest.raises(ValueError):
            _loader._helper.make_path_absolute(path)

    #  Should transform a 2d numpy array to a 3d numpy array with the correct shape
    def test_transform_2d_to_3d(self):
        # Arrange
        data = np.random.rand(6,6)
        len_x = 3
        len_y = 3

        # Act
        result = _loader._helper.to_cube(data, len_x, len_y)

        # Assert
        assert result.shape == (4, 3, 3)

    def test_normalize_valid_input(self):
        """
        Test the normalize function with valid input.
        """
        spec = np.array([10, 20, 30, 40, 50], dtype=np.float32)
        normalized = helper.normalize(spec)

        # Expected normalization: (x - min) / (max - min)
        expected = (spec - spec.min()) / (spec.max() - spec.min())
        np.testing.assert_array_almost_equal(normalized, expected, decimal=5, err_msg="Normalization output is incorrect.")

    def test_normalize_constant_input(self):
        """
        Test the normalize function with constant input.
        """
        spec = np.array([10, 10, 10, 10, 10], dtype=np.float32)
        normalized = helper.normalize(spec)

        # Expected output: all zeros since max == min
        expected = spec
        np.testing.assert_array_almost_equal(normalized, expected, decimal=5, err_msg="Normalization of constant input is incorrect.")

    def test_feature_registration_valid_input(self):
        """
        Test the feature_regestration function with valid input.
        """
        # Create two dummy grayscale images
        img1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img2 = np.roll(img1, shift=5, axis=1)  # Shifted version of img1

        # Call the feature registration function
        aligned_img, homography = helper.feature_regestration(img1, img2)

        # Assert the aligned image and homography are not None
        assert aligned_img is not None, "Aligned image should not be None."
        assert homography is not None, "Homography should not be None."

    def test_feature_registration_non_uint8_input(self):
        """
        Test the feature_regestration function with non-uint8 input.
        """
        # Create two dummy grayscale images with float32 type
        img1 = np.random.rand(100, 100).astype(np.float32) * 255
        img2 = np.roll(img1, shift=5, axis=1)

        # Call the feature registration function
        aligned_img, homography = helper.feature_regestration(img1, img2)

        # Assert the aligned image and homography are not None
        assert aligned_img is not None, "Aligned image should not be None."
        assert homography is not None, "Homography should not be None."

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

class TestLoaderNrrd:

    def test_read_write_nrrd(self, sample_data_cube):

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.nrrd', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Write the DataCube to the temporary file
            _loader.nrrd._write_nrrd(dc=sample_data_cube, path=temp_path)

            # Read the DataCube from the temporary file
            loaded_data_cube = _loader.nrrd._read_nrrd(path=temp_path)

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

            # Assertions to ensure the loaded data matches the original data
            # np.testing.assert_array_almost_equal(loaded_data_cube.cube, sample_data_cube.cube)
            np.testing.assert_array_equal(loaded_data_cube.wavelengths, sample_data_cube.wavelengths)

        finally:
            # Clean up the temporary file
            os.remove(temp_path) 

class TestLoaderFolder:
    def test_read_folder_filter_image_files(self):
        files = ["image.jpg", "document.pdf", "photo.png", "archive.zip", "picture.TIFF"]
        expected = ["image.jpg", "photo.png", "picture.TIFF"]
        assert _loader.folder.filter_image_files(files) == expected


    def test_read_folder_load_image(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
            plt.imsave(temp_path, np.random.rand(10, 10, 3))

        try:
            img = _loader.folder.load_image(temp_path)
            assert isinstance(img, np.ndarray)
            assert img.shape == (10,10,3)  # Ensure it is a 3-channel image
        finally:
            os.remove(temp_path)


    def test_read_folder_image_to_dc_single(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
            plt.imsave(temp_path, np.random.rand(10, 10, 3))

        try:
            dc = _loader.folder.image_to_dc(temp_path)
            print(dc.shape)
            assert isinstance(dc, DataCube)
            assert dc.cube.shape[0] == 3  # Ensure channel-first format
        finally:
            os.remove(temp_path)


    def test_read_folder_image_to_dc_batch(self):
        temp_files = []
        for _ in range(3):
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_files.append(temp_file.name)
            plt.imsave(temp_file.name, np.random.rand(8, 7, 3)) # x=8, y=7, v=3
            temp_file.close()

        try:
            dc = _loader.folder.image_to_dc(temp_files)

            assert isinstance(dc, DataCube)
            assert dc.shape == (9, 8, 7)  # v=3, x=8, y=7
        finally:
            for temp_path in temp_files:
                os.remove(temp_path)


    def test_read_folder_image_to_dc_batch_pusbroom(self):
        temp_files = []
        for _ in range(3):
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_files.append(temp_file.name)
            plt.imsave(temp_file.name, np.random.rand(8, 7, 3))
            temp_file.close()

        try:
            dc = _loader.folder.image_to_dc(temp_files, type='pushbroom')

            print(dc.shape)

            assert isinstance(dc, DataCube)
            assert dc.shape == (7,9,8)   # v=7, x=9, y=8
        finally:
            for temp_path in temp_files:
                os.remove(temp_path)


    def test_read_folder_no_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="No valid image files found in the directory."):
                _loader.folder._read_folder(temp_dir)

class TestLoaderTDMS:

    def test_read_tdms_invalid_file(self):
        with pytest.raises(FileNotFoundError):
            _loader.tdms._read_tdms("non_existent_file.tdms")


class TestLoaderXLSX:

    def test_write_and_read_xlsx(self, sample_data_cube, tmp_path):
        """Tests whether the write and read functions are consistent."""
        test_cube = sample_data_cube
        test_file =  tmp_path / "test.xlsx"

        # Write DataCube to Excel
        _loader.xlsx._write_xlsx(test_cube, str(test_file))

        # Read back the DataCube
        read_cube = wizard.read(str(test_file))

        # Verify dimensions
        assert test_cube.shape == read_cube.shape, "Shapes do not match!"

        # Verify wavelengths
        np.testing.assert_array_equal(test_cube.wavelengths, read_cube.wavelengths) # "Wavelengths do not match!"

        # Verify data content
        np.testing.assert_allclose(test_cube.cube, read_cube.cube, atol=1e-6), "Data cubes do not match!"


class TestLoaderCSV:

    def test_write_and_read_csv(self, sample_data_cube, tmp_path):
        """Tests whether the write and read functions are consistent for CSV."""
        test_cube = sample_data_cube
        test_file = tmp_path / "test.csv"

        # Write DataCube to CSV
        _loader.csv._write_csv(test_cube, str(test_file))

        # Read back the DataCube
        read_cube = wizard.read(str(test_file))

        # Verify dimensions
        assert test_cube.shape == read_cube.shape, "Shapes do not match!"

        # Verify wavelengths
        np.testing.assert_array_equal(test_cube.wavelengths, read_cube.wavelengths)  # "Wavelengths do not match!"

        # Verify data content
        np.testing.assert_allclose(test_cube.cube, read_cube.cube, atol=1e-6), "Data cubes do not match!"

class TestLoaderFSM:

    def test_wrong_len_block_info(self):
        """
        check if _block_info raises excaption if data len != 6
        """
        data = b'blablablablablab'

        with pytest.raises(ValueError):
            _loader.fsm._block_info(data)

    def test_right_len_block_info(self):
        """
        check if _block_info dosnt raise an excaption if data len == 6
        """
        data = b'blabla'
        try:
            _loader.fsm._block_info(data)
        except:
            assert False
        assert True


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

class TestHelper:


    def test_normalize_basic(self):
        spec = np.array([0, 5, 10])
        normalized = helper.normalize(spec)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(normalized, expected, atol=1e-6)


    def test_normalize_already_normalized(self):
        spec = np.array([0.0, 0.5, 1.0])
        normalized = helper.normalize(spec)
        np.testing.assert_array_equal(normalized, spec)


    def test_normalize_constant_array(self):
        spec = np.array([3, 3, 3])
        normalized = helper.normalize(spec)
        np.testing.assert_array_equal(normalized, spec)  # Should remain unchanged


    def test_normalize_negative_values(self):
        spec = np.array([-10, 0, 10])
        normalized = helper.normalize(spec)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(normalized, expected, atol=1e-6)


    def test_normalize_large_range(self):
        spec = np.array([1e9, 2e9, 3e9])
        normalized = helper.normalize(spec)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(normalized, expected, atol=1e-6)


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
