"""
core.datacube.py
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

import inspect
import warnings

import yaml
import numpy as np
from rich import print

# from traitlets import ValidateHandler

from .._utils.tracker import TrackExecutionMeta


class DataCube(metaclass=TrackExecutionMeta):
    """
    The `DataCube` class stores hyperspectral imaging (HSI) data as a 3D array.

    The cube is a 3D array of shape (v, x, y):

    - `x` and `y` represent the pixel coordinates.
    - `v` represents measured values, such as counts, channels, or wavelengths.

    In most cases, the data cube contains wavelength information, which can be in units such as nm or cm⁻¹.
    The `notation` parameter allows you to specify this information.

    Attributes
    ----------
    cube : np.ndarray, optional
        3D numpy array representing the data cube. Default is None.
    wavelengths : list | np.ndarray, optional
        List of wavelengths corresponding to the `v` axis of the data cube.
    name : str, optional
        Name of the data cube. Default is None.
    notation : str, optional
        Specifies whether the wavelength data is in nm or cm⁻¹. Default is None.
    record : bool, optional
        If True, execution of the methods will be recorded. Default is False.

    Methods added dynamically:
    - remove_spikes()
    - resize()
    - remove_background()
    - baseline_als()
    - inverse()

    These methods are injected via `__init__` for ease of use. Example:

    ```python
    dc = DataCube()
    dc.remove_spikes()  # Works directly
    ```
    """

    def __init__(self, cube=None, wavelengths=None, name=None, notation=None, record: bool = False, registered: bool = False) -> None:
        """
        Initialize a new `DataCube` instance.

        Parameters
        ----------
        cube : np.ndarray, optional
            3D numpy array representing the data cube. Default is None.
        wavelengths : list | np.ndarray, optional
            List of wavelengths corresponding to the `v` axis of the data cube.
        name : str, optional
            Name of the data cube. Default is None.
        notation : str, optional
            Specifies whether the wavelength data is in nm or cm⁻¹. Default is None.
        record : bool, optional
            If True, execution of the methods will be recorded. Default is False.
        registered: bool optional
            If True images are allready registered. Default is False
        """
        self.name = name  # name of the dc
        self.shape = None if cube is None else cube.shape  # shape of the dc
        self.dim = None  # get dimension of the dc 2d, 3d, 4d ...
        self.wavelengths = np.array(wavelengths) if wavelengths is not None \
            else np.arange(0, cube.shape[0], dtype=int) if cube is not None \
            else None
        self.cube = None if cube is None else cube
        self.notation = notation
        self.registered = registered

        self.record = record
        if self.record:
            self.start_recording()

    def __add__(self, other):
        """
        Add two `DataCube` instances.

        This method concatenates the cubes along the `v` axis. The x and y dimensions of both cubes must match.

        Parameters
        ----------
        other : DataCube
            Another `DataCube` instance to add.

        Raises
        ------
        ValueError
            If the x and y dimensions of the cubes do not match or if the cube contains None values.

        Returns
        -------
        DataCube
            New `DataCube` instance with combined data.
        """
        if not isinstance(other, DataCube):
            raise ValueError('Cant add DataCube and none DataCube.')

        if self.cube is None or other.cube is None:
            raise ValueError("Cannot add DataCubes with None values.")

        new_wavelengths = None

        if self.cube.shape[1:] != other.cube.shape[1:]:
            raise ValueError(
                f'DataCubes needs to have the same `x` an `y` shape.\n'
                f'Cube1: {self.cube.shape}, Cube2: {other.cube.shape}'
                f'You can use the DataCube.resize function to adjust the cubes'
            )

        if self.wavelengths is None or other.wavelengths is None:
            warnings.warn('One of the two DataCubes does not contain the'
                          ' wavelength information. Adding them will work,'
                          ' but you will lose this information.')
        else:
            new_wavelengths = self.wavelengths + other.wavelengths

        new_cube = np.concatenate((self.cube, other.cube), axis=0)
        return DataCube(cube=new_cube, wavelengths=new_wavelengths,
                        name=self.name, notation=self.notation)

    def __len__(self) -> int:
        """
        Return the number of layers (v dimension) in the data cube.

        Returns
        -------
        int
            Number of layers in the data cube.
        """
        return self.shape[0] if self.cube is not None else 0

    def __getitem__(self, idx):
        """
        Get an item from the data cube.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns
        -------
        np.ndarray
            Selected item from the data cube.
        """
        return self.cube[idx]

    def __setitem__(self, idx, value) -> None:
        """
        Set an item in the data cube.

        Parameters
        ----------
        idx : int
            Index where the value should be set.
        value : np.ndarray
            Value to be set at the given index.
        """
        self.cube[idx] = value

    def __iter__(self):
        """
        Return an iterator for the data cube.

        Returns
        -------
        DataCube
            Iterator for the data cube.
        """
        self.idx = 0
        return self

    def __next__(self):
        """
        Return the next item in the data cube during iteration.

        Returns
        -------
        tuple
            Next cube layer and corresponding wavelength.

        Raises
        ------
        StopIteration
            If no more items are available in the data cube.
        """
        if self.idx >= len(self.cube):
            raise StopIteration
        else:
            self.idx += 1
            return self.cube[self.idx - 1]

    def __str__(self) -> str:
        """
        Return a string representation of the `DataCube`.

        Returns
        -------
        str
            String containing information about the data cube.
        """
        n = '\n'
        _str = ''
        _str += f'Name: {self.name}' + n
        _str += f'Shape: {self.shape}' + n
        if self.wavelengths is not None:
            _str += 'Wavelengths:' + n
            _str += f'\tLen: {len(self.wavelengths)}' + n
            _str += f'\tFrom: {self.wavelengths.min()}' + n
            _str += f'\tTo: {self.wavelengths.max()}' + n
        if self.notation is not None:
            _str += 'Notaion: ' + self.notation
        return _str

    def custom_read(self, *args, **kwargs) -> None:
        """
        Placeholder method to read data into the `DataCube`.

        Parameters
        ----------
        *args : tuple
            Positional arguments for future extension.
        **kwargs : dict
            Keyword arguments for future extension.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        if hasattr(self, 'reading_func'):
            _dc = self.reading_func(*args, **kwargs)

            # stupid but easly readble, mapping would be overkill...
            self.set_cube(_dc.cube)
            if _dc.wavelengths is not None:
                self.set_wavelengths(_dc.wavelengths)
            if _dc.notation is not None:
                self.set_notation(_dc.notation)
            if _dc.name is not None:
                self.set_name(_dc.name)
        else:
            raise NotImplementedError('You need to implement a `custom_read` method and set it with `set_custom_read(fuc).')

    def set_custom_reader(self, reading_func):
        """
        Set a custom reading function for loading data.
        :param reading_func: A callable that takes a path as input and returns the loaded data.
        """
        self.reading_func = reading_func

    def set_name(self, name:str) -> None:
        """
        Set a name for the DataCube.

        :param name: Name as string.
        """
        if name and isinstance(name, str):
            self.name = name
        else:
            raise AttributeError('Name musste be a string.')

    def set_wavelengths(self, wavelengths: np.ndarray) -> None:
        """
        Set wavelength data for the `DataCube`.

        Parameters
        ----------
        wavelengths : np.ndarray or list
            1D numpy array or list of wavelength data. If a list is provided, it will be converted to a numpy array.

        Raises
        ------
        AttributeError
            If the input wavelengths are not a 1D array or list.
        """
        if not isinstance(wavelengths, np.ndarray):
            try:
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

    def set_cube(self, cube: np.ndarray) -> None:
        """
        Set data for the `DataCube`.

        Parameters
        ----------
        cube : np.ndarray
            2D, 3D, or 4D numpy array of data. If a 2D array is provided, it will be expanded to 3D.

        Raises
        ------
        AttributeError
            If the input cube is not a numpy array or cannot be converted to one,
            or if its dimensionality is not 2, 3, or 4.
        """
        if not isinstance(cube, np.ndarray):
            try:
                cube = np.array(cube)
            except ValueError:
                print('oh no')
                raise ValueError('Your cube is not convertable to a `np.array`')

        if 3 <= cube.ndim <= 4:
            self.cube = cube
        elif cube.ndim == 2:
            self.cube = np.zeros(shape=(1, cube.shape[0], cube.shape[1]),
                                 dtype=cube.dtype)
            self.cube[0] = cube
            print(f'\033[93mYour cube got forced to {self.cube.shape}\033[0m')
        else:
            raise AttributeError('Cube Data is not ndim 2,3 or 4')
        self._set_cube_shape()

    def _set_cube_shape(self) -> None:
        """Update the shape of the data cube."""
        self.shape = self.cube.shape

    def set_notation(self, notation:str) -> None:
        """
        Update the notation for the DataCube.

        Parameters
        ----------
        notation : str
            The notation describing the units for wavelength data, such as 'nm' for nanometers
            or 'cm⁻¹' for inverse centimeters.
        """

        self.notation = notation

    def start_recording(self) -> None:
        """Start recording method execution for the `DataCube`."""
        self.record = True
        TrackExecutionMeta.start_recording()

    def stop_recording(self) -> None:
        """Stop recording method execution for the `DataCube`."""
        self.record = False
        TrackExecutionMeta.stop_recording()

    def save_template(self, filename) -> None:
        """
        Save a template of recorded methods to a YAML file.

        Parameters
        ----------
        filename : str
            Name of the YAML file where the template will be saved. If the filename does not
            end with `.yml`, it will automatically be appended.

        Raises
        ------
        AttributeError
            If the filename is None or not a string.
        """
        if not filename:
            raise AttributeError('Filename can\'t be `None`')
        elif not isinstance(filename, str):
            t = type(filename)
            raise AttributeError(f'Filename must be string not {t}')

        if not (filename.endswith('.yml') or filename.endswith('yaml')):
            filename = filename + '.yaml'

        cleaned_data = self._clean_data(TrackExecutionMeta.recorded_methods)
        y = yaml.dump(cleaned_data,default_flow_style=False, sort_keys=False)
        with open(filename, 'w') as template_file:
            template_file.write(y)
        print(f'Saved templated: {filename}')

    def _map_args_to_kwargs(sef, func, args, kwargs):
        """
        Map positional arguments (args) to keyword arguments (kwargs)
        based on the function signature.

        Parameters
        ----------
        func : function
            Function to be analyzed.
        args : tuple
            Positional arguments passed to the function.
        kwargs : dict
            Keyword arguments passed to the function.

        Returns
        -------
        dict
            Combined keyword arguments.
        """
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        new_kwargs = kwargs.copy()  # Start with the provided kwargs
        for i, arg in enumerate(args):
            if i < len(params):
                param_name = params[i].name
                new_kwargs[param_name] = arg

        return new_kwargs

    def _clean_data(self, recorded_methods):
        """
        Clean recorded method data by removing unnecessary entries.

        Parameters
        ----------
        recorded_methods : list
            List of recorded methods and their arguments.

        Returns
        -------
        list
            Cleaned method data for template saving.
        """
        cleaned_data = []

        for method_name, args, kwargs in recorded_methods:
            # Clean out DataCube instances from the arguments
            cleaned_args = [arg for arg in args if not isinstance(arg, DataCube)]

            # Get the actual function object from the DataCube class
            func = getattr(DataCube, method_name, None)

            if func is not None:
                # Map positional args to kwargs using cleaned args
                full_kwargs = self._map_args_to_kwargs(func, cleaned_args, kwargs)
            else:
                print(f"Warning: Method {method_name} not found in DataCube.")
                full_kwargs = kwargs

            cleaned_entry = {
                'method': method_name,
                'kwargs': full_kwargs
            }

            cleaned_data.append(cleaned_entry)

        return cleaned_data

    def execute_template(self, filename) -> None:
        """
        Load a template and execute the corresponding methods.

        Parameters
        ----------
        filename : str
            Name of the YAML file containing the template.

        """
        with open(filename, 'rb') as template_file:
            template_data = yaml.safe_load(template_file)

        for i in range(len(template_data)):
            method = getattr(self, template_data[i]['method'])
            kwargs = template_data[i]['kwargs']
            method(**kwargs)

    def remove_spikes(self, threshold: int = 6500, window: int = 5):
        """
        Remove impulsive spike artifacts from the datacube.

        Parameters
        ----------
        threshold : int, default=6500
            Intensity threshold used to detect spikes.
        window : int, default=5
            Neighborhood window size used to filter/replace detected spikes.

        Returns
        -------
        DataCube
            The processed datacube (typically the same instance) to allow chaining.
        """
        from ..processing.denoise import remove_spikes
        return remove_spikes(self, threshold=threshold, window=window)

    def remove_background(self, threshold: int = 50, style: str = "dark"):
        """
        Remove background pixels based on an intensity threshold.

        Parameters
        ----------
        threshold : int, default=50
            Threshold used to identify background pixels.
        style : {"dark", "light"}, default="dark"
            Background assumption. Use ``"dark"`` if background is near zero,
            ``"light"`` if background is bright.

        Returns
        -------
        DataCube
            The processed datacube (typically the same instance) to allow chaining.
        """
        from ..processing.background import remove_background
        return remove_background(self, threshold=threshold, style=style)

    def resize(self, x_new: int, y_new: int, interpolation: str = "linear"):
        """
        Resize the spatial dimensions of the datacube.

        Parameters
        ----------
        x_new : int
            New width (x-dimension).
        y_new : int
            New height (y-dimension).
        interpolation : str, default="linear"
            Interpolation method used for resampling.

        Returns
        -------
        DataCube
            The resized datacube (same instance) to allow chaining.
        """
        from ..processing.rescale import resize
        resize(self, x_new=x_new, y_new=y_new, interpolation=interpolation)
        return self  # resize mutates self

    def baseline_als(self, lam: float = 1_000_000, p: float = 0.01, niter: int = 10):
        """
        Apply ALS (Asymmetric Least Squares) baseline correction to spectra.

        Parameters
        ----------
        lam : float, default=1_000_000
            Smoothness parameter (larger values produce smoother baselines).
        p : float, default=0.01
            Asymmetry parameter (typically small, e.g. 0.001–0.1).
        niter : int, default=10
            Number of ALS iterations.

        Returns
        -------
        DataCube
            Baseline-corrected datacube (typically the same instance) to allow chaining.
        """
        from ..processing.baseline import baseline_als
        return baseline_als(self, lam=lam, p=p, niter=niter)

    def merge_cubes(self, other: "DataCube", register: bool = False):
        """
        Merge another datacube into this datacube.

        Parameters
        ----------
        other : DataCube
            The other datacube to merge with this one.
        register : bool, default=False
            If True, register/align the cubes before merging (implementation-dependent).

        Returns
        -------
        DataCube
            The merged datacube (typically the same instance) to allow chaining.
        """
        from ..processing.geometry import merge_cubes
        return merge_cubes(self, other, register=register)

    def inverse(self):
        """
        Invert intensities of the datacube.

        Returns
        -------
        DataCube
            The inverted datacube (typically the same instance) to allow chaining.
        """
        from wizard.processing.normalization import inverse
        return inverse(self)

    def register_layers_simple(self, max_features: int = 5000, match_percent: float = 0.1):
        """
        Register (align) spectral layers using a simple feature-based approach.

        Parameters
        ----------
        max_features : int, default=5000
            Maximum number of features to detect per layer.
        match_percent : float, default=0.1
            Fraction of best matches to keep when estimating transforms.

        Returns
        -------
        DataCube
            The registered datacube (typically the same instance) to allow chaining.
        """
        from ..processing.registration import register_layers_simple
        return register_layers_simple(self, max_features=max_features, match_percent=match_percent)

    def remove_vignetting_poly(self, axis: int = 1, slice_params: dict | None = None):
        """
        Correct vignetting using a polynomial model estimated along a given axis.

        Parameters
        ----------
        axis : int, default=1
            Axis along which the correction profile is estimated.
        slice_params : dict or None, default=None
            Optional slicing parameters to restrict the region used for estimation.

        Returns
        -------
        DataCube
            The vignetting-corrected datacube (typically the same instance) to allow chaining.
        """
        from ..processing.background import remove_vignetting_poly
        return remove_vignetting_poly(self, axis=axis, slice_params=slice_params)

    def normalize(self) -> "DataCube":
        """
        Normalize datacube intensities (implementation-dependent).

        Returns
        -------
        DataCube
            The normalized datacube (typically the same instance) to allow chaining.
        """
        from wizard.processing.normalization import normalize
        return normalize(self)

    def register_layers_best(self, ref_layer: int = 0, max_features: int = 5000, match_percent: float = 0.1, rot_thresh: float = 20.0, scale_thresh: float = 1.1):
        """
        Register (align) spectral layers using a more robust/best-effort approach.

        Parameters
        ----------
        ref_layer : int, default=0
            Index of the reference layer to which other layers are aligned.
        max_features : int, default=5000
            Maximum number of features to detect per layer.
        match_percent : float, default=0.1
            Fraction of best matches to keep when estimating transforms.
        rot_thresh : float, default=20.0
            Rotation threshold used to reject implausible transforms.
        scale_thresh : float, default=1.1
            Scale threshold used to reject implausible transforms.

        Returns
        -------
        DataCube
            The registered datacube (typically the same instance) to allow chaining.
        """
        from ..processing.registration import register_layers_best
        return register_layers_best(self, ref_layer=ref_layer, max_features=max_features, match_percent=match_percent, rot_thresh=rot_thresh, scale_thresh=scale_thresh)

    def remove_vignetting(self, sigma: float = 50, clip: bool = True, epsilon: float = 1e-6):
        """
        Correct vignetting by estimating and removing a smooth illumination field.

        Parameters
        ----------
        sigma : float, default=50
            Smoothing parameter used when estimating the illumination field.
        clip : bool, default=True
            If True, clip output values to a valid range (implementation-dependent).
        epsilon : float, default=1e-6
            Small constant to avoid division by zero.

        Returns
        -------
        DataCube
            The vignetting-corrected datacube (typically the same instance) to allow chaining.
        """
        from ..processing.background import remove_vignetting
        return remove_vignetting(self, sigma=sigma, clip=clip, epsilon=epsilon)

    def upscale_datacube_edsr(self, scale: int, model_path: str):
        """
        Upscale the datacube spatially using an EDSR super-resolution model.

        Parameters
        ----------
        scale : int
            Upscaling factor (e.g., 2, 3, 4).
        model_path : str
            Path to the pre-trained EDSR model.

        Returns
        -------
        DataCube
            The upscaled datacube (typically the same instance) to allow chaining.
        """
        from ..processing.rescale import upscale_datacube_edsr
        return upscale_datacube_edsr(self, scale=scale, model_path=model_path)

    def upscale_datacube_espcn(self, scale: int, model_path: str):
        """
        Upscale the datacube spatially using an ESPCN super-resolution model.

        Parameters
        ----------
        scale : int
            Upscaling factor (e.g., 2, 3, 4).
        model_path : str
            Path to the pre-trained ESPCN model.

        Returns
        -------
        DataCube
            The upscaled datacube (typically the same instance) to allow chaining.
        """
        from ..processing.rescale import upscale_datacube_espcn
        return upscale_datacube_espcn(self, scale=scale, model_path=model_path)

    def upscale_datacube_with_reference(self, reference_image):
        """
        Upscale the datacube using a reference image as guidance.

        Parameters
        ----------
        reference_image : array-like
            Reference image used to guide upscaling (type depends on implementation).

        Returns
        -------
        DataCube
            The upscaled datacube (typically the same instance) to allow chaining.
        """
        from ..processing.rescale import upscale_datacube_with_reference
        return upscale_datacube_with_reference(self, reference_image=reference_image)

    def upscale_datacube_fsrcnn(self, scale: int, model_path: str):
        """
        Upscale the datacube spatially using an FSRCNN super-resolution model.

        Parameters
        ----------
        scale : int
            Upscaling factor (e.g., 2, 3, 4).
        model_path : str
            Path to the pre-trained FSRCNN model.

        Returns
        -------
        DataCube
            The upscaled datacube (typically the same instance) to allow chaining.
        """
        from ..processing.rescale import upscale_datacube_fsrcnn
        return upscale_datacube_fsrcnn(self, scale=scale, model_path=model_path)

    def remove_vignette(self, vignette_map, flip: bool = False):
        """
        Apply a provided vignette correction map to the datacube.

        Parameters
        ----------
        vignette_map : array-like
            Multiplicative (or divisive) correction field used to compensate vignetting.
        flip : bool, default=False
            If True, flip the vignette map before applying (orientation correction).

        Returns
        -------
        DataCube
            The vignette-corrected datacube (typically the same instance) to allow chaining.
        """
        from ..processing.background import remove_vignette
        return remove_vignette(self, vignette_map=vignette_map, flip=flip)

    def uniform_filter_dc(self, size: int = 3):
        """
        Apply a uniform (mean) filter to each layer of the datacube.

        Parameters
        ----------
        size : int, default=3
            Filter kernel size.

        Returns
        -------
        DataCube
            The filtered datacube (typically the same instance) to allow chaining.
        """
        from ..processing.denoise import uniform_filter_dc
        return uniform_filter_dc(self, size=size)
