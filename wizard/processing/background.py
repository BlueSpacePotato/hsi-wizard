from ..core import DataCube

import rembg
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter


def remove_background(dc: DataCube, threshold: int = 50, style: str = 'dark') -> DataCube:
    """
    Remove background from images in a DataCube.

    Uses an external algorithm (rembg). The first image in the DataCube
    is processed to generate a mask, which is then applied to all images
    to remove the background.

    Parameters
    ----------
    dc : DataCube
    DataCube containing the image stack.
    threshold : int, optional
    Threshold value to define the background from the alpha mask,
    defaults to 50. Pixels with alpha < threshold are considered background.
    style : str, optional
    Style of background removal, 'dark' or 'bright', defaults to 'dark'.
    If 'dark', background pixels are set to 0.
    If 'bright', background pixels are set to the max value of the cube.y

    Returns
    -------
    DataCube
    DataCube with the background removed.

    Raises
    ------
    ValueError
    If style is not 'dark' or 'bright'.

    Examples
    --------
    >>> import wizard
    >>> dc = wizard.read("example.fsm")
    >>> dc.remove_background(threshold=50, style='bright') # or style 'dark'
    """
    img = dc.cube[0]
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype('uint8')
    img = Image.fromarray(img)
    img_removed_bg = rembg.remove(img)
    mask = np.array(img_removed_bg.getchannel('A'))

    cube = dc.cube.copy()
    if style == 'dark':
        cube[:, mask < threshold] = 0
    elif style == 'bright':
        cube[:, mask < threshold] = dc.cube.max()
    else:
        raise ValueError("Type must be 'dark' or 'bright'")
    dc.set_cube(cube)
    return dc



def remove_vignetting_poly(dc: DataCube, axis: int = 1, slice_params: dict = None) -> DataCube:
    """
    Remove vignetting using polynomial fitting along a specified axis.

    Calculates the mean along the specified axis (1 for rows, 2 for columns)
    for each spectral layer, fits a polynomial to this mean profile
    (after Savitzky-Golay smoothing), and subtracts this fitted profile
    from the corresponding rows/columns of the layer to correct for vignetting.

    Parameters
    ----------
    dc : DataCube
    The DataCube instance to process.
    axis : int, optional
    The axis along which to calculate the mean and apply correction.
    1 for correcting along rows (profile used for columns),
    2 for correcting along columns (profile used for rows). Defaults to 1.
    slice_params : dict, optional
    Dictionary for slicing behavior before mean calculation.
    Keys: ``"start"`` (int), ``"end"`` (int), ``"step"`` (int).
    Defaults to full slice with step 1.

    Returns
    -------
    DataCube
    The processed DataCube with vignetting removed.

    Raises
    ------
    ValueError
    If the DataCube is empty or axis is not 1 or 2.

    Examples
    --------
    >>> import wizard
    >>> dc = wizard.read('example.fsm')
    >>> params = {"start":25, "end":50}
    >>> dc.remove_vignetting_poly(slice_params=params, axis=2)
    """
    if dc.cube is None:
        raise ValueError("The DataCube is empty. Please provide a valid cube.")

    if slice_params is None:
        slice_params = {"start": None, "end": None, "step": 1}
    start = slice_params.get("start", None)
    end = slice_params.get("end", None)
    step = slice_params.get("step", 1)

    if axis == 1:
        summed_data = np.mean(dc.cube[:, :, start:end:step], axis=2)
    elif axis == 2:
        summed_data = np.mean(dc.cube[:, start:end:step, :], axis=1)
    else:
        raise ValueError('Axis can only be 1 or 2.')

    corrected_cube = dc.cube.copy().astype(np.float32)

    for i, layer_profile in enumerate(summed_data):
        smoothed_layer_profile = savgol_filter(layer_profile, window_length=71, polyorder=1)
        if axis == 1:
            for j_col in range(dc.cube.shape[2]):
                corrected_cube[i, :, j_col] -= smoothed_layer_profile
        elif axis == 2:
            for j_row in range(dc.cube.shape[1]):
                corrected_cube[i, j_row, :] -= smoothed_layer_profile

    dc.set_cube(corrected_cube)
    return dc


def remove_vignetting(dc: DataCube, sigma: float = 50, clip: bool = True, epsilon: float = 1e-6) -> DataCube:
    """
    Remove vignetting from a hyperspectral DataCube.

    Corrects vignetting in each spectral band by estimating a smooth
    background using Gaussian blur and then performing flat-field correction.
    The background is normalized by its mean before correction.

    Parameters
    ----------
    dc : DataCube
    The input DataCube (bands, height, width).
    sigma : float, optional
    Standard deviation for Gaussian blur, controlling smoothness.
    Larger sigma means coarser background estimation. Defaults to 50.
    clip : bool, optional
    If True and the original DataCube dtype is integer,
    clip output values to the valid range of that integer type.
    Defaults to True.
    epsilon : float, optional
    A small constant to add to the background before division
    to prevent division by zero errors. Defaults to 1e-6.

    Returns
    -------
    DataCube
    The DataCube with vignetting corrected. The output cube has the
    same shape and dtype as the input.

    Examples
    --------
    >>> import wizard
    >>> dc = wizard.read('example.fsm')
    >>> dc.remove_vignetting()
    """
    corrected_cube = np.empty_like(dc.cube)
    orig_dtype = dc.cube.dtype
    is_int = np.issubdtype(orig_dtype, np.integer)

    for i in range(dc.cube.shape[0]):
        band = dc.cube[i].astype(np.float64)
        background = gaussian_filter(band, sigma=sigma)
        background = np.maximum(background, epsilon)
        background_mean = background.mean()
        if background_mean > epsilon:
            background /= background_mean
        else:
            background = np.ones_like(background, dtype=np.float64)
        corrected_band = band / background
        if is_int:
            info = np.iinfo(orig_dtype)
            corrected_band = np.round(corrected_band)
            corrected_band = np.clip(corrected_band, info.min, info.max)
        corrected_cube[i] = corrected_band.astype(orig_dtype)
    dc.set_cube(corrected_cube)
    return dc


def remove_vignette(dc: DataCube, vignette_map: np.ndarray, flip: bool = False) -> DataCube:
    """
    Subtract a vignette pattern from every spectral layer.

    Removes spatial vignetting by subtracting the provided vignette_map
    from each (x, y) layer in the data cube. If flip=True, the vignette
    pattern is inverted (dark center → bright center) before subtraction.

    Parameters
    ----------
    dc: DataCube
    An instance of the DataCube class. Must have attributes:
    - .cube: numpy array of shape (v, x, y)
    - .wavelength: list of length v
    vignette_map : np.ndarray
    2D array of shape (x, y) representing the vignette intensity to subtract.
    Values should be on the same scale as the cube’s pixel intensities.
    flip : bool, default=False
    If True, invert the vignette_map before subtraction.

    Returns
    -------
    None
    Modifies the DataCube.cube in-place.

    Raises
    ------
    ValueError
    If vignette_map.shape does not match the spatial dimensions of the cube.

    Notes
    -----
    - After subtraction, any negative values in the cube are clipped to zero.
    - Assumes cube and vignette_map share the same intensity scale.

    Examples
    --------
    >>> dc = DataCube(cube=np.random.rand(10, 256, 256), wavelength=list(np.linspace(400,700,10)))
    >>> # Remove standard vignette (darker at edges, brighter center)
    >>> dc.remove_vignette(vignette_map=my_vignette_image)
    >>> # Remove flipped vignette (darker center, brighter edges)
    >>> dc.remove_vignette(vignette_map=my_vignette_image, flip=True)
    """
    # Check that the vignette map matches spatial dims
    if vignette_map.shape != dc.cube.shape[1:]:
        raise ValueError(
            f"vignette_map shape {vignette_map.shape} does not match cube spatial shape {dc.cube.shape[1:]}"
        )

    cube = dc.cube.copy()

    # Optionally invert the vignette pattern
    if flip:
        vignette_map = vignette_map.max() - vignette_map

    # Subtract vignette from each layer
    # Using broadcasting: vignette_map has shape (x, y), expand to (1, x, y)
    cube -= vignette_map[np.newaxis, :, :]

    # Clip negative values to zero
    np.clip(cube, a_min=0, a_max=None, out=cube)

    dc.set_cube(cube)

    return dc

