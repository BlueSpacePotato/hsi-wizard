
# Generated by CodiumAI
import numpy as np

from hsi_wizard.datacube import DataCube

import pytest


class TestDataCube:

    #  DataCube can be initialized with cube, wavelengths, name, and notation.
    def test_initialize_with_parameters(self):
        cube = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        wavelengths = np.array([400, 500])
        name = "Cube1"
        notation = "A"
    
        data_cube = DataCube(cube=cube, wavelengths=wavelengths, name=name, notation=notation)
    
        assert data_cube.cube is not None
        assert data_cube.wavelengths is not None
        assert data_cube.name == name
        assert data_cube.notation == notation

    #  DataCube can be concatenated with another DataCube using the '+' operator.
    def test_concatenate_datacubes(self):
        cube1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        wavelengths1 = np.array([400, 500])
        name1 = "Cube1"
        notation1 = "A"
    
        cube2 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
        wavelengths2 = np.array([600, 700])
        name2 = "Cube2"
        notation2 = "B"
    
        data_cube1 = DataCube(cube=cube1, wavelengths=wavelengths1, name=name1, notation=notation1)
        data_cube2 = DataCube(cube=cube2, wavelengths=wavelengths2, name=name2, notation=notation2)
    
        result = data_cube1 + data_cube2
    
        assert result.cube is not None
        assert result.wavelengths is not None
        assert result.name == name1
        assert result.notation == notation1

    #  DataCube's shape can be accessed.
    def test_access_shape(self):
        cube = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        data_cube = DataCube(cube=cube)
    
        shape = data_cube.shape
    
        assert shape == (2, 2, 2)

    #  DataCube's length can be accessed.
    def test_access_length(self):
        cube = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        data_cube = DataCube(cube=cube)
    
        length = len(data_cube)
    
        assert length == 2

    #  DataCube's layers can be accessed using the [] operator.
    def test_access_layers(self):
        cube = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        data_cube = DataCube(cube=cube)
    
        layer = data_cube[0]
    
        assert np.array_equal(layer, np.array([[1, 2], [3, 4]]))

    #  DataCube can be initialized with no parameters.
    def test_initialize_with_no_parameters(self):
        data_cube = DataCube()
    
        assert data_cube.cube is None
        assert data_cube.wavelengths is None
        assert data_cube.name is None
        assert data_cube.notation is None

    #  DataCube can be initialized with a cube but no wavelengths.
    def test_initialize_with_cube_no_wavelengths(self):
        cube = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        data_cube = DataCube(cube=cube)
    
        assert data_cube.cube is not None
        assert data_cube.wavelengths is not None

    #  DataCube can be concatenated with another DataCube with different x and y shapes.
    def test_concatenate_datacubes_different_shapes(self):
        cube1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        wavelengths1 = np.array([400, 500])
        name1 = "Cube1"
        notation1 = "A"
    
        cube2 = np.array([[[9, 10, 11], [12, 13, 14]], [[15, 16, 17], [18, 19, 20]]])
        wavelengths2 = np.array([600, 700])
        name2 = "Cube2"
        notation2 = "B"
    
        data_cube1 = DataCube(cube=cube1, wavelengths=wavelengths1, name=name1, notation=notation1)
        data_cube2 = DataCube(cube=cube2, wavelengths=wavelengths2, name=name2, notation=notation2)
    
        with pytest.raises(ValueError):
            result = data_cube1 + data_cube2

    # Initializes a DataCube instance with all parameters.
    def test_all_parameters(self):
        cube = np.zeros((10, 100, 100))
        wavelengths = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490]
        name = "Example DataCube"
        notation = "wavelengths"
        record = True

        data_cube = DataCube(cube=cube, wavelengths=wavelengths, name=name, notation=notation, record=record)

        assert data_cube.name == name
        assert data_cube.shape == cube.shape
        assert data_cube.dim is None
        assert np.array_equal(data_cube.wavelengths, np.array(wavelengths))
        assert np.array_equal(data_cube.cube, cube)
        assert data_cube.notation == notation
        assert data_cube.record is True

    # Add a datacube and an int and check if the ValueError gets raised.
    def test_add_with_int_raises_valueerror(self):
        cube1 = DataCube(cube=np.ones((3, 2, 2)))
        cube2 = 5
        with pytest.raises(ValueError):
            cube1 + cube2

    # Adding two DataCubes with same shape and one of them has empty wavelengths should return a new DataCube with
    # concatenated cube and None wavelengths, and a warning message
    def test_same_shape_one_empty_wavelengths(self):
        dc1 = DataCube(cube=np.ones((10, 100, 100)), wavelengths=np.arange(10), name="dc1")
        dc2 = DataCube(cube=np.ones((10, 100, 100)), wavelengths=None, name="dc2")
        dc2.wavelengths = None

        with pytest.warns(UserWarning):
            result = dc1 + dc2

        assert isinstance(result, DataCube)
        assert result.cube.shape == (20, 100, 100)
        assert result.cube.shape[0] == len(result.wavelengths)

    #  Set a valid 3D numpy array to the cube attribute
    def test_set_valid_3d_array(self):
        cube_data = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        dc = DataCube()
        dc.set_cube(cube_data)
        assert np.array_equal(dc.cube, cube_data)

    #  Set a valid 4D numpy array to the cube attribute
    def test_set_valid_4d_array(self):
        cube_data = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
        dc = DataCube()
        dc.set_cube(cube_data)
        assert np.array_equal(dc.cube, cube_data)

    #  Set a valid 2D numpy array to the cube attribute
    def test_set_valid_2d_array(self):
        cube_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dc = DataCube()
        dc.set_cube(cube_data)
        expected_cube = np.zeros(shape=(1, cube_data.shape[0], cube_data.shape[1]), dtype=cube_data.dtype)
        expected_cube[0] = cube_data
        assert np.array_equal(dc.cube, expected_cube)

    #  Set a valid 3D numpy array to the cube attribute with the same shape as an existing cube
    def test_set_valid_3d_array_same_shape(self):
        existing_cube = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        dc = DataCube(cube=existing_cube)
        cube_data = np.array([[[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
        dc.set_cube(cube_data)
        assert np.array_equal(dc.cube, cube_data)

    #  Set a valid 4D numpy array to the cube attribute with the same shape as an existing cube
    def test_set_valid_4d_array_same_shape(self):
        existing_cube = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
        dc = DataCube(cube=existing_cube)
        cube_data = np.array([[[[10, 11, 12], [13, 14, 15], [16, 17, 18]]]])
        dc.set_cube(cube_data)
        assert np.array_equal(dc.cube, cube_data)

    #  Set an empty numpy array to the cube attribute
    def test_set_empty_array(self):
        cube_data = np.array([])
        dc = DataCube()
        with pytest.raises(AttributeError):
            dc.set_cube(cube_data)
        assert np.array_equal(dc.cube, None)

    #  Set a list to the cube attribute
    def test_set_list(self):
        cube_data = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
        dc = DataCube()
        dc.set_cube(cube_data)
        assert np.array_equal(dc.cube, np.array(cube_data))

    # test __iter__ and __next__
    def test_returns_next_item_and_wavelength(self):
        dc = DataCube(cube=np.random.rand(10, 100, 100), wavelengths=np.arange(10))
        iterator = iter(dc)
        next_value = next(iterator)
        assert isinstance(next_value, tuple)
        assert len(next_value) == 2
        assert isinstance(next_value[0], np.ndarray)

    # Should return a string with the name and shape of the DataCube
    def test_return_string_with_name_and_shape(self):
        dc = DataCube(cube=np.ones((3, 4, 5)), wavelengths=[400, 500, 600], name="Example")
        expected_output = ("Name: Example\n"
                           "Shape: (3, 4, 5)\n"
                           "Wavelengths\n"
                           "Num: 3\n"
                           "From: 400\n"
                           "To: 600\n"
                           "Cube:\n"
                           "[[[1. 1. 1. 1. 1.]\n  [1. 1. 1. 1. 1.]\n  [1. 1. 1. 1. 1.]\n  [1. 1. 1. 1. 1.]]\n\n"
                           " [[1. 1. 1. 1. 1.]\n  [1. 1. 1. 1. 1.]\n  [1. 1. 1. 1. 1.]\n  [1. 1. 1. 1. 1.]]\n\n"
                           " [[1. 1. 1. 1. 1.]\n  [1. 1. 1. 1. 1.]\n  [1. 1. 1. 1. 1.]\n  [1. 1. 1. 1. 1.]]]\n")

        assert dc.__str__() == expected_output

    # Resizing a cube with valid x_new and y_new values and linear interpolation
    def test_valid_resize_linear_interpolation(self):
        cube = np.random.rand(10, 100, 100)
        dc = DataCube(cube=cube)
        dc.resize(50, 50, interpolation='linear')
        assert dc.cube.shape == (10, 50, 50)

    # Resizing a cube with valid x_new and y_new values and nearest interpolation
    def test_valid_resize_nearest_interpolation(self):
        cube = np.random.rand(10, 100, 100)
        dc = DataCube(cube=cube)
        dc.resize(50, 50, interpolation='nearest')
        assert dc.cube.shape == (10, 50, 50)

    # Resizing a cube with x_new and y_new values
    def test_resize_to_up_value(self):
        cube = np.random.rand(10, 100, 100)
        dc = DataCube(cube=cube)
        dc.resize(200, 200, interpolation='linear')
        assert dc.cube.shape == (10, 200, 200)

    #  Method sets wavelength data when given a valid 1D numpy array.
    def test_valid_wavelengths_array(self):
        wavelengths = np.array([400, 450, 500, 550, 600])
        dc = DataCube()
        dc.set_wavelengths(wavelengths)
        assert np.array_equal(dc.wavelengths, wavelengths)

    #  Method sets wavelength data when given an empty 1D numpy array.
    def test_empty_wavelengths_array(self):
        wavelengths = np.array([])
        dc = DataCube()
        dc.set_wavelengths(wavelengths)
        assert np.array_equal(dc.wavelengths, wavelengths)

    #  Method raises AttributeError when given a non-convertible object.
    def test_non_convertible_object(self):
        wavelengths = "invalid"
        dc = DataCube()
        with pytest.raises(AttributeError):
            dc.set_wavelengths(wavelengths)

    #  Method raises AttributeError when given a numpy array with ndim != 1.
    def test_ndim_not_1(self):
        wavelengths = np.array([[400, 450, 500], [550, 600, 650]])
        dc = DataCube()
        with pytest.raises(AttributeError):
            dc.set_wavelengths(wavelengths)

    #  Method raises AttributeError when given a numpy array with ndim == 0.
    def test_ndim_0(self):
        wavelengths = np.array(400)
        dc = DataCube()
        with pytest.raises(AttributeError):
            dc.set_wavelengths(wavelengths)