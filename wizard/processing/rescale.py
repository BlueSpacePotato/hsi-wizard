from ..core import DataCube

import cv2
import numpy as np

def resize(dc: DataCube, x_new: int, y_new: int, interpolation: str = 'linear') -> None:
    """
    Resize the DataCube to new x and y dimensions.
    dc.shape is v,x,y

    Resizes each 2D slice (x, y) of the DataCube using the specified
    interpolation method.

    Interpolation methods:
    * ``linear``: Bilinear interpolation (ideal for enlarging).
    * ``nearest``: Nearest neighbor interpolation (fast but blocky).
    * ``area``: Pixel area interpolation (ideal for downscaling).
    * ``cubic``: Bicubic interpolation (high quality, slower).
    * ``lanczos``: Lanczos interpolation (highest quality, slowest).

    Parameters
    ----------
    dc : DataCube
    The DataCube instance to be resized.
    x_new : int
    The new width (x-dimension).
    y_new : int
    The new height (y-dimension).
    interpolation : str, optional
    Interpolation method, defaults to 'linear'.
    Options: 'linear', 'nearest', 'area', 'cubic', 'lanczos'.

    Returns
    -------
    None
    The DataCube is modified in-place.

    Raises
    ------
    ValueError
    If the interpolation method is not recognized.

    Examples
    --------
    >>> import wizard
    >>> dc = wizard.read("example.fsm")
    >>> dc.resize(x_new=500, y_new=500)
    """
    mode = None
    shape = dc.cube.shape
    if shape[1] > x_new:
        print('\033[93mx_new is smaller than the existing cube, you will lose information\033[0m')
    if shape[2] > y_new:
        print('\033[93my_new is smaller than the existing cube, you will lose information\033[0m')

    if interpolation == 'linear':
        mode = cv2.INTER_LINEAR
    elif interpolation == 'nearest':
        mode = cv2.INTER_NEAREST
    elif interpolation == 'area':
        mode = cv2.INTER_AREA
    elif interpolation == 'cubic':
        mode = cv2.INTER_CUBIC
    elif interpolation == 'lanczos':
        mode = cv2.INTER_LANCZOS4
    else:
        raise ValueError(f'Interpolation method `{interpolation}` not recognized.')

    _cube = np.empty(shape=(shape[0], x_new, y_new))
    for idx, layer in enumerate(dc.cube):
        _cube[idx] = cv2.resize(layer, (y_new, x_new), interpolation=mode)
    dc.cube = _cube
    dc._set_cube_shape()


def upscale_datacube_edsr(dc: DataCube, scale: int, model_path: str):
    """
    Upscale the spatial dimensions of a DataCube using the EDSR super-resolution model.

    This function applies the EDSR (Enhanced Deep Super-Resolution) model from OpenCV’s
    dnn_superres module to each spectral slice of the given DataCube. Since EDSR
    expects a 3-channel input, each single-band slice is duplicated across three
    channels before processing. The output slices are then reassembled into a new
    DataCube with upscaled spatial dimensions.

    Parameters
    ----------
    dc : DataCube
    An instance of the DataCube class. Must have attributes:
    - datacube.cube: numpy array of shape (v, x, y)
    - datacube.wavelength: list of length v
    scale : int
    The upscaling factor (e.g., 2, 3, or 4) supported by the EDSR model.
    model_path : str
    Filesystem path to the pretrained EDSR `.pb` model file
    (e.g., `"EDSR_x4.pb"`).

    Returns
    -------
    DataCube
    A new DataCube instance whose `.cube` has shape
    (v, x * scale, y * scale) and the same `.wavelength` list.

    Raises
    ------
    FileNotFoundError
    If the specified `model_path` does not exist or is unreadable.
    cv2.error
    If OpenCV fails to load the model or perform upsampling.
    ValueError
    If the `scale` is not one of the factors supported by the loaded model.

    Notes
    -----
    - Requires `opencv-contrib-python>=4.3.0`.
    - Processing is done slice-by-slice and may be slow for large cubes.
    - Memory usage grows by approximately `scale^2`.

    Examples
    --------
    >>> import wizard
    >>> dc = wizard.read("hyperspectral.fsm")
    >>> dc.upscale_datacube_edsr(scale=4, model_path="EDSR_x4.pb")
    >>> print(dc.cube.shape)
    (v, x*4, y*4)
    """
    # Create EDSR super-res object
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", scale)

    v, x, y = dc.shape
    new_x, new_y = x * scale, y * scale
    up_cube = np.empty((v, new_x, new_y), dtype=dc.cube.dtype)

    for i in range(v):
        # Extract single-band slice
        band = dc.cube[i, :, :]

        # Convert to 3-channel BGR by stacking
        bgr = cv2.merge([band, band, band])

        # Upsample
        up_bgr = sr.upsample(bgr)

        # Take one channel (all channels are identical)
        up_band = up_bgr[:, :, 0]

        up_cube[i, :, :] = up_band

    # Build new DataCube
    dc.set_cube(up_cube)
    return dc


def upscale_datacube_espcn(dc: DataCube, scale: int, model_path: str):
    """
    Upscale the spatial dimensions of a DataCube using the ESPCN super-resolution model.

    This function applies the ESPCN (Efficient Sub-Pixel Convolutional Neural Network)
    model from OpenCV’s dnn_superres module to each spectral slice of the given DataCube.
    Since ESPCN expects a 3-channel input, each single-band slice is duplicated across
    three channels before processing. The output slices are then reassembled into a new
    DataCube with upscaled spatial dimensions.

    Parameters
    ----------
    dc : DataCube
    An instance of the DataCube class. Must have attributes:
    - .cube: numpy array of shape (v, x, y)
    - .wavelength: list of length v
    scale : int
    The upscaling factor (2, 3, or 4) supported by the ESPCN model.
    model_path : str
    Filesystem path to the pretrained ESPCN `.pb` model file
    (e.g., "ESPCN_x3.pb").

    Returns
    -------
    DataCube
    A new DataCube instance whose `.cube` has shape
    (v, x * scale, y * scale) and the same `.wavelength` list.

    Raises
    ------
    FileNotFoundError
    If the specified `model_path` does not exist.
    ValueError
    If `scale` is not in {2, 3, 4}.
    cv2.error
    If OpenCV fails to load the model or perform upsampling.

    Notes
    -----
    - Requires `opencv-contrib-python>=4.3.0`.
    - Processing is done slice-by-slice and may be slow for large cubes.
    - Memory usage grows by approximately `scale**2`.

    Examples
    --------
    >>> import wizard
    >>> dc = wizard.read("test.fsm")
    >>> dc.upscale_datacube_espcn(scale=3, model_path="ESPCN_x3.pb")
    >>> print(dc.cube.shape)
    (v, x*3, y*3)
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if scale not in (2, 3, 4):
        raise ValueError(f"ESPCN only supports scale factors 2, 3, or 4, got {scale}")

    # Initialize ESPCN super-resolver
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("espcn", scale)

    v, x, y = dc.shape
    new_x, new_y = x * scale, y * scale
    up_cube = np.empty((v, new_x, new_y), dtype=dc.cube.dtype)

    for i in range(v):
        # extract single-band slice
        band = dc.cube[i, :, :]

        # create 3-channel image
        bgr = cv2.merge([band, band, band])

        # run ESPCN upsampling
        up_bgr = sr.upsample(bgr)

        # channels are identical: pick one
        up_band = up_bgr[:, :, 0]
        up_cube[i, :, :] = up_band

    # assemble new DataCube
    dc.set_cube(up_cube)
    return dc


def upscale_datacube_with_reference(dc: DataCube, reference_image: np.ndarray) -> 'DataCube':
    """
    Upscales a DataCube to match the spatial resolution of a reference image using ESRGAN.

    Each spectral band of the DataCube is individually upscaled using the ESRGAN model
    by treating the single band as a pseudo-grayscale image. The resulting DataCube has
    the same spatial resolution as the reference image.

    Parameters
    ----------
    dc : DataCube
    The input DataCube to be upscaled. Expected shape (v, x, y).

    reference_image : np.ndarray
    A high-resolution reference image (e.g., RGB) that defines the target (x, y) resolution.

    Returns
    -------
    DataCube
    A new DataCube with the same number of bands, but upscaled to match the reference image's spatial resolution.

    Raises
    ------
    ValueError
    If the reference image resolution is smaller than the datacube resolution.

    Notes
    -----
    This method uses the ESRGAN model via TensorFlow Hub. The model is applied to each spectral band independently.

    Examples
    --------
    >>> upscaled_cube = upscale_datacube_with_reference(cube, ref_image)
    """
    import tensorflow as tf
    import tensorflow_hub as hub

    # Load ESRGAN model
    model_url = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
    model = hub.load(model_url)

    # Target resolution from reference image
    target_x, target_y = reference_image.shape[:2]

    v, x, y = dc.shape

    if x >= target_x or y >= target_y:
        raise ValueError("Reference image must be larger than the DataCube in spatial dimensions.")

    upscaled_bands = []

    for i in range(v):
        band = dc.cube[i, :, :]

        # Normalize and expand dims for ESRGAN
        img = (band / band.max() * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_rgb = tf.convert_to_tensor(img_rgb, dtype=tf.float32)

        # Ensure divisible by 4
        imageSize = (tf.convert_to_tensor(img_rgb.shape[:-1]) // 4) * 4
        cropped_image = tf.image.crop_to_bounding_box(
            img_rgb, 0, 0, imageSize[0], imageSize[1]
        )
        preprocessed_image = tf.expand_dims(cropped_image, 0)

        # Run super-resolution
        sr_result = model(preprocessed_image)
        sr_image = tf.squeeze(sr_result).numpy() / 255.0

        # Convert back to single band and resize to match reference
        sr_gray = cv2.cvtColor((sr_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        resized_band = cv2.resize(sr_gray.astype(np.float32), (target_y, target_x), interpolation=cv2.INTER_CUBIC)

        upscaled_bands.append(resized_band)

    upscaled_cube_array = np.stack(upscaled_bands, axis=0)

    dc.set_cube(upscaled_cube_array)
    return dc


def upscale_datacube_fsrcnn(dc, scale, model_path):
    """
    Upscale the spatial dimensions of a DataCube using the FSRCNN super-resolution model.

    This function applies the FSRCNN (Fast Super-Resolution Convolutional Neural Network)
    model from OpenCV’s dnn_superres module to each spectral slice of the given DataCube.
    Since FSRCNN expects a 3-channel input, each single-band slice is duplicated across three
    channels before processing. The output slices are then reassembled into a new DataCube
    with upscaled spatial dimensions.

    Parameters
    ----------
    dc : DataCube
    An instance of the DataCube class. Must have attributes:
    - .cube: numpy array of shape (v, x, y)
    - .wavelength: list of length v
    scale : int
    The upscaling factor (2, 3, or 4) supported by the FSRCNN model.
    model_path : str
    Filesystem path to the pretrained FSRCNN `.pb` model file
    (e.g., "FSRCNN_x3.pb").

    Returns
    -------
    DataCube
    A new DataCube instance whose `.cube` has shape
    (v, x * scale, y * scale) and the same `.wavelength` list.

    Raises
    ------
    FileNotFoundError
    If the specified `model_path` does not exist.
    ValueError
    If `scale` is not in {2, 3, 4}.
    cv2.error
    If OpenCV fails to load the model or perform upsampling.

    Notes
    -----
    - Requires `opencv-contrib-python>=4.3.0`.
    - Processing is done slice-by-slice and may be slow for large cubes.
    - Memory usage grows by approximately `scale**2`.

    Examples
    --------
    >>> dc = wizard.read("hyperspectral.npy", wavelengths=[...])
    >>> up_dc = upscale_datacube_fsrcnn(dc, scale=3, model_path="FSRCNN_x3.pb")
    >>> print(up_dc.cube.shape)
    (v, x*3, y*3)
    """
    # Validate model file
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # Validate supported scale factors
    if scale not in (2, 3, 4):
        raise ValueError(f"FSRCNN only supports scale factors 2, 3, or 4, got {scale}")

    # Initialize FSRCNN super-resolver
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", scale)

    # Prepare new cube
    v, x, y = dc.cube.shape
    new_x, new_y = x * scale, y * scale
    up_cube = np.empty((v, new_x, new_y), dtype=dc.cube.dtype)

    # Upscale each spectral band
    for i in range(v):
        band = dc.cube[i, :, :]
        bgr = cv2.merge([band, band, band])
        up_bgr = sr.upsample(bgr)
        up_band = up_bgr[:, :, 0]
        up_cube[i, :, :] = up_band

    # Create and return new DataCube
    dc.set_cube(up_cube)
    return dc

