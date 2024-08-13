"""DataCube class for storring HSI data."""
import warnings

import cv2
from rich import print
import numpy as np
import pickle

from .._utils import decorators
from .._utils.tracker import TrackExecutionMeta


class DataCube(metaclass=TrackExecutionMeta):
    """
    DataCube class to store several important information for the hsi data.

    The cube is a 3D array with shape `vxy`.
    x, y values describe the `pixel`.
    v values are for the measured counts, channels, values, ...

    In the most cases you have some kind of wavelengths information in your dc.
    These can be nm or cm^-1. You can applie your description in notation for
    the dc and cast it.
    """

    def __init__(self, cube=None, wavelengths=None, name=None,
                 notation=None, record: bool = False) -> None:
        """
        Magic Methods to init a new instance.

        :param cube: dc
        :param wavelengths: wavelengths of the spectral information as list
        :param name: name of the dc
        :param notation: wavelengths or wavenumbers
        :return: None
        :rtype return: None
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
        Magic Method to add two Datacubes.

        :param other: DataCube that should be added
        :return: None
        :type: None
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
        Magic Method for getting length of axis as int.

        :param axis: int for selection axis
        :return: length of DataCube for given axis
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
        raise NotImplementedError('Subclasses must implement the `load`'
                                  'method')

    def resize(self, x_new: int, y_new: int,
               interpolation: str = 'linear') -> None:
        """
        Resize DataCube.cube.

        cv2.INTER_LINEAR 	The standard bilinear interpolation, ideal for
                            enlarged images.
        cv2.INTER_NEAREST 	The nearest neighbor interpolation, which, though
                            fast to run, creates blocky images.
        cv2.INTER_AREA 	 The interpolation for the pixel area, which scales
                            down images.
        cv2.INTER_CUBIC 	The bicubic interpolation with 4×4-pixel
                            neighborhoods, which, though slow to run, generates
                            high-quality instances.
        cv2.INTER_LANCZOS4 	The Lanczos interpolation with an 8×8-pixel
                            neighborhood, which generates images of the highest
                            quality but is the slowest to run.

        :param interpolation:
        :param x_new:
        :param y_new:
        :return:
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
