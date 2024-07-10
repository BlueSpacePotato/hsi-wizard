# Generated by CodiumAI

from wizard._utils.fileHandler import get_files_by_extension
from wizard._utils.fileHandler import make_path_absolute
from wizard._utils.fileHandler import to_cube

import pytest

import numpy as np


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
        data = np.array(
            [
            [
                [1, 2, 3],
                [4, 5, 6],
                [1, 2, 3]
            ],
            [
                [4, 5, 6],
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [1, 2, 3],
                [4, 5, 6],
                [1, 2, 3]
                ],[
                [4, 5, 6],
                [1, 2, 3],
                [4, 5, 6]
            ]
        ]
        )
        len_x = 3
        len_y = 3

        # Act
        result = to_cube(data, len_x, len_y)

        # Assert
        assert result.shape == (4, 3, 3)
