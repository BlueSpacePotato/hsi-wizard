"""
_core/datacube.py
=================

.. module:: datacube
   :platform: Unix
   :synopsis: DataCube class for storing HSI data.

Module Overview
---------------

This module provides the `DataCube` class, which is used
to store hyperspectral imaging (HSI) data. The `DataCube`
is a 3D array where the x and y axes represent pixels,
and the v axis stores measured values like counts or
wavelengths.

Classes
-------
.. autoclass:: DataCube
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------
Here is an example of how to use this module:

.. code-block:: python

    from wizard import DataCube
    dc1 = DataCube(cube=cube_data)
    print(dc1)

"""



import warnings

import cv2
from rich import print
import numpy as np
import pickle

from .._utils import decorators
from .._utils.tracker import TrackExecutionMeta


class DataCube(metaclass=TrackExecutionMeta):
    """
    The `DataCube` class stores hyperspectral imaging (HSI) data as a 3D array.

    The cube is a 3D array of shape (v, x, y):
    - `x` and `y` represent the pixel coordinates.
    - `v` represents measured values, such as counts, channels, or wavelengths.

    In most cases, the data cube contains wavelength information, which can be in units such as nm or cm⁻¹.
    The `notation` parameter allows you to specify this information.

    Attributes:
    -----------
    - `cube`: 3D numpy array storing the actual data.
    - `wavelengths`: 1D numpy array representing the wavelengths.
    - `name`: Optional name for the data cube.
    - `notation`: Specifies whether the wavelength data is in nm or cm⁻¹.
    """



    def __init__(self, cube=None, wavelengths=None, name=None, notation=None, record: bool = False) -> None:
        """
        Initialize a new `DataCube` instance.

        :param cube: 3D numpy array representing the data cube. Default is None.
        :type cube: np.ndarray, optional
        :param wavelengths: List of wavelengths corresponding to the `v` axis of the data cube.
        :type wavelengths: list, optional
        :param name: Name of the data cube. Default is None.
        :type name: str, optional
        :param notation: Specifies whether the wavelength data is in nm or cm⁻¹. Default is None.
        :type notation: str, optional
        :param record: If True, execution of the methods will be recorded. Default is False.
        :type record: bool
        :rtype: None
        """

        self.name = name  # name of the dc
        self.shape = None if cube is None else cube.shape  # shape of the dc
        self.dim = None  # get dimension of the dc 2d, 3d, 4d ...
        self.wavelengths = np.array(wavelengths) if wavelengths is not None \
            else np.arange(0, cube.shape[0], dtype=int) if cube is not None \
            else None
        self.cube = None if cube is None else cube
        self.notation = notation

        self.record = record
        if self.record:
            self.start_recording()

    def __add__(self, other):
        """
        Add two `DataCube` instances.

        This method concatenates the cubes along the `v` axis. The x and y dimensions of both cubes must match.

        :param other: Another `DataCube` instance to add.
        :type other: DataCube
        :raises ValueError: If the x and y dimensions of the cubes do not match or if the cube contains None values.
        :rtype: DataCube
        """

        if not isinstance(other, DataCube):
            raise ValueError('Cant add DataCube and none DataCube.')

        new_wavelengths = None

        if self.cube.shape[1:] != other.cube.shape[1:]:
            raise ValueError(
                f'DataCubes needs to have the same `x` an `y` shape.\n'
                f'Cube1: {self.cube.shape}, Cube2: {other.cube.shape}'
                f'You can use the DataCube.resize function to adjust the cubes'
            )

        # wavelengths cant be empty atm
        if self.wavelengths is None or other.wavelengths is None:
            warnings.warn('One of the two DataCubes does not contain the'
                          ' wavelength information. Adding them will work,'
                          ' but you will lose this information.')
        else:
            new_wavelengths = self.wavelengths + other.wavelengths

        if self.cube is None or other.cube is None:
            raise ValueError("Cannot add DataCubes with None values.")
        new_cube = np.concatenate((self.cube, other.cube), axis=0)
        return DataCube(cube=new_cube, wavelengths=new_wavelengths,
                        name=self.name, notation=self.notation)

    def __len__(self) -> int:
        """
        Return the number of layers (v dimension) in the data cube.

        :rtype: int
        """

        return self.shape[0] if self.cube is not None else 0

    def __getitem__(self, idx):
        """
        Magic Method to get an item.

        :param idx:
        :return:

        """
        return self.cube[idx]

    def __setitem__(self, idx, value) -> None:
        """
        Magic Method to set an item.

        :param idx:
        :param value:
        :return: None
        """
        self.cube[idx] = value

    def __iter__(self):
        """
        Magic Method to iter ofer DataCube.

        :return:
        """
        self.idx = 0
        return self

    def __next__(self):
        """
        Magic Method for next.

        :return:
        """
        if self.idx >= len(self.cube):
            raise StopIteration
        else:
            self.idx += 1
            return self.cube[self.idx - 1], self.wavelengths[self.idx - 1]

    # def __sizeof__(self):
    #   pass

    def __str__(self) -> str:
        """
        Magic Method, print dc information.

        :return : string with dc information
        :rtype: str
        """
        n = '\n'
        _str = ''
        _str += f'Name: {self.name}' + n
        _str += f'Shape: {self.shape}' + n
        if self.wavelengths is not None:
            _str += 'Wavelengths' + n
            _str += f'Num: {len(self.wavelengths)}' + n
            _str += f'From: {self.wavelengths.min()}' + n
            _str += f'To: {self.wavelengths.max()}' + n
        _str += 'Cube:' + n
        _str += f'{self.cube}' + n
        return _str

    def execute_template(self, template_data) -> None:
        """
        Execute Template.

        :return:
        """
        for method_name, args, kwargs in template_data:
            method = getattr(self, method_name)
            method(*args, **kwargs)

    @decorators.check_load_dc
    def load(self, *args, **kwargs) -> None:
        """
        Empty load Function to override.

        This is a template function. You can implemnt your own load functions.

        :return:
        """
        raise NotImplementedError('Subclasses must implement the `load` method')

    def resize(self, x_new: int, y_new: int, interpolation: str = 'linear') -> None:
        """
        Resize the data cube to new dimensions using the specified interpolation method.

        Interpolation methods:
        - `cv2.INTER_LINEAR`: Standard bilinear interpolation (ideal for enlarging).
        - `cv2.INTER_NEAREST`: Nearest neighbor interpolation (fast but blocky).
        - `cv2.INTER_AREA`: Pixel area interpolation (ideal for downscaling).
        - `cv2.INTER_CUBIC`: Bicubic interpolation (high quality, slower).
        - `cv2.INTER_LANCZOS4`: Lanczos interpolation (highest quality, slowest).

        :param interpolation: The interpolation method to use (e.g., 'linear', 'nearest').
        :type interpolation: str
        :param x_new: The new width (x-dimension) of the cube.
        :type x_new: int
        :param y_new: The new height (y-dimension) of the cube.
        :type y_new: int
        :rtype: None
        """

        mode = None

        shape = self.cube.shape

        if shape[1] > x_new:
            print('\033[93mx_new is smaller then the exising cube,'
                  'you lose information\033[0m')
        if shape[2] > y_new:
            print('\033[93my_new is smaller then the exising cube,'
                  'you lose information\033[0m')

        if interpolation == 'linear':
            mode = cv2.INTER_LINEAR
        elif interpolation == 'nearest':
            mode = cv2.INTER_NEAREST
        elif interpolation == 'area':
            mode = cv2.INTER_AREA
        elif interpolation == 'cubic':
            mode = cv2.INTER_CUBIC
        elif interpolation == 'Lanczos':
            mode = cv2.INTER_LANCZOS4

        _cube = np.empty(shape=(shape[0], y_new, x_new))
        for idx, layer in enumerate(self.cube):
            _cube[idx] = cv2.resize(layer, (x_new, y_new), interpolation=mode)
        self.cube = _cube
        self.update_cube_shape()

    # todo: implemnt
    def shift_layers(self, num_layer: int) -> None:
        """
        Shift layers.

        :return: None
        """
        if num_layer > self.shape[0]:
            raise ValueError(f'`Num_layer` {num_layer} must me <= then the'
                             'layer deeps of the DataCube {self.shape[0]}')
        raise NotImplementedError('Sorry - Not Implemented')

    def set_wavelengths(self, wavelengths: np.array) -> None:
        """
        Set wavelength data.

        :return: None
        """
        if not isinstance(wavelengths, np.ndarray):
            try:
                # todo: better error handling
                if np.array(wavelengths).ndim == 1:
                    self.wavelengths = np.array(wavelengths)
                else:
                    raise AttributeError
            except AttributeError:
                raise AttributeError('Your wavelengths didnt match an'
                                     '1d np.array')

        else:
            if wavelengths.ndim == 1:
                self.wavelengths = wavelengths
            else:
                raise AttributeError('Your wavelengths didnt match an'
                                     '1d np.array')

    def set_cube(self, cube: np.array) -> None:
        """
        Set cube data.

        :return: None
        """
        if not isinstance(cube, np.ndarray):
            try:
                # todo: better error handling
                cube = np.array(cube)
            except AttributeError:
                raise AttributeError('Your cube is not convertable to a'
                                     'np.array')
        if 3 <= cube.ndim <= 4:
            self.cube = cube
        elif cube.ndim == 2:
            self.cube = np.zeros(shape=(1, cube.shape[0], cube.shape[1]),
                                 dtype=cube.dtype)
            self.cube[0] = cube
            print(f'\033[93mYour cube got forced to {self.cube.shape}\033[0m')
        else:
            raise AttributeError('Cube Data is not ndim 2,3 or 4')
        self.update_cube_shape()

    def update_cube_shape(self) -> None:
        """
        Update cube shape.

        :return: None
        """
        self.shape = self.cube.shape

    def start_recording(self):
        """
        Start Recording.

        :return: None
        """
        self.record = True
        TrackExecutionMeta.start_recording()

    def stop_recording(self) -> None:
        """
        Stop Recording.

        :return: None
        """
        self.record = False
        TrackExecutionMeta.stop_recording()

    @staticmethod
    def save_template(filename) -> None:
        """
        Save template from executed functions.

        :return: None
        """
        if not filename.endswith('.pickle'):
            filename = filename + '.pickle'
        with open(filename, 'wb') as template_file:
            pickle.dump(TrackExecutionMeta.recorded_methods, template_file)

    def load_template(self, filenmae) -> None:
        """
        Load template and execute function.

        :return: None
        """
        with open(filenmae, 'rb') as template_file:
            template_data = pickle.load(template_file)
        self.execute_template(template_data)
