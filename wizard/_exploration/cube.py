"""
cube.py
==========

.. module:: cube
:platform: Unix
:synopsis: cube plotting module for the hsi-wizard package.

Module Overview
--------------

This module provides functionalities for visualizing data cubes.

"""

from wizard import DataCube
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_datacube_3d(datacube, sample_rate=1, vmin=None, vmax=None, cmap='viridis'):
    """
    Plot a 3D volumetric representation of a DataCube.

    This function visualizes the spectral cube data in 3D space using a scatter plot,
    with axes corresponding to spatial dimensions (x, y) and the spectral (wavelength) axis.
    Colors represent the intensity values of the cube.
    Useful for exploratory visualization of spectral–spatial data.

    Parameters
    ----------
    datacube : DataCube
        An instance of the DataCube class with attributes `cube` (numpy.ndarray of shape (v, x, y))
        and `wavelength` (sequence of length v).
    sample_rate : int, optional
        Factor by which to subsample the data along each axis for performance.
        Only every n-th point along each dimension is plotted. Default is 1 (no subsampling).
    vmin : float, optional
        Minimum intensity value for color normalization. Defaults to the minimum of the sampled data.
    vmax : float, optional
        Maximum intensity value for color normalization. Defaults to the maximum of the sampled data.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use for the scatter points. Default is 'viridis'.

    Returns
    -------
    None
        Displays a 3D scatter plot of the DataCube.

    Raises
    ------
    TypeError
        If `datacube` does not have the required `cube` or `wavelength` attributes.
    ValueError
        If `sample_rate` is less than 1 or not an integer, or if the cube and wavelength lengths mismatch.

    Notes
    -----
    - This visualization may be slow for large DataCubes. Consider increasing `sample_rate` for
    larger datasets.
    - Requires `matplotlib` with the `mplot3d` toolkit.

    Examples
    --------
    >>> import numpy as np
    >>> from wizard import DataCube
    >>> dc = DataCube(cube=np.random.rand(50, 100, 100),
    ...               wavelength=np.linspace(400, 700, 50))
    >>> plot_datacube_3d(dc, sample_rate=5, vmin=0.1, vmax=0.9, cmap='plasma')
    """

    # Validate inputs
    if not hasattr(datacube, 'cube') or not hasattr(datacube, 'wavelengths'):
        raise TypeError("datacube must have 'cube' and 'wavelength' attributes")
    if not isinstance(sample_rate, int) or sample_rate < 1:
        raise ValueError("sample_rate must be an integer >= 1")

    cube = datacube.cube
    wl = datacube.wavelengths
    print(cube.shape)
    if cube.ndim != 3 or cube.shape[0] != wl.shape[0]:
        raise ValueError("cube must be shape (v, x, y) and len(wavelengths) == v")

    # Subsample indices
    v, x_dim, y_dim = cube.shape
    w_idx = np.arange(0, v, sample_rate)
    x_idx = np.arange(0, x_dim, sample_rate)
    y_idx = np.arange(0, y_dim, sample_rate)

    # Create 3D grid of points
    X, Y, W = np.meshgrid(x_idx, y_idx, wl[w_idx], indexing='xy')
    # Extract and reorder intensities to match meshgrid shape
    vals = cube[w_idx][:, x_idx][:, :, y_idx].transpose(2, 1, 0).flatten()

    # Flatten coordinate arrays
    xs = X.flatten()
    ys = Y.flatten()
    zs = W.flatten()

    # Color normalization
    if vmin is None:
        vmin = vals.min()
    if vmax is None:
        vmax = vals.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xs, ys, zs, c=vals, cmap=cmap, norm=norm, marker='o', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Wavelength')
    fig.colorbar(sc, ax=ax, label='Intensity')
    plt.tight_layout()
    plt.show()


def plot_datacube_solid(datacube: DataCube,
                        wl_index: int = None,
                        x_index: int = None,
                        y_index: int = None,
                        vmin: float = None,
                        vmax: float = None,
                        cmap: str = 'viridis'):
    """
    Plot a solid “corner” of a DataCube as three colored orthogonal faces.

    This function renders a block consisting of the bottom XY plane at a chosen
    wavelength, the back X–W plane at the maximum Y index, and the right Y–W
    plane at the maximum X index—producing a solid-looking cube corner. Color
    encodes intensity.

    Parameters
    ----------
    datacube : DataCube
        A `wizard.DataCube` with `cube` (numpy.ndarray of shape (v, x, y))
        and `wavelength` (sequence of length v).
    wl_index : int, optional
        Spectral slice for the bottom (XY) face. Defaults to 0 (lowest wavelength).
    x_index : int, optional
        Spatial X slice for the right (Y–W) face. Defaults to last X index.
    y_index : int, optional
        Spatial Y slice for the back  (X–W) face. Defaults to last Y index.
    vmin : float, optional
        Minimum intensity for colormap. Defaults to global minimum.
    vmax : float, optional
        Maximum intensity for colormap. Defaults to global maximum.
    cmap : str or Colormap, optional
        Matplotlib colormap name or instance. Default is 'viridis'.

    Returns
    -------
    None
        Displays a 3D plot of three orthogonal colored faces forming a solid cube corner.

    Raises
    ------
    TypeError
        If `datacube` lacks `cube` or `wavelength` attributes.
    ValueError
        If slice indices are out of valid ranges or cube shape mismatches wavelength.

    Notes
    -----
    - Uses Matplotlib's `mplot3d` toolkit.
    - Defaults show the “visible” corner: bottom at min wavelength, back and right faces.
    - You can override `wl_index`, `x_index`, and `y_index` to explore interior slices.
    - For very large cubes, consider downsampling before plotting.

    Examples
    --------
    >>> from wizard import DataCube
    >>> import numpy as np
    >>> dc = DataCube(cube=np.random.rand(50, 30, 30),
    ...               wavelength=np.linspace(400, 700, 50))
    >>> plot_datacube_solid(dc)
    """
    # Validate inputs
    if not hasattr(datacube, 'cube') or not hasattr(datacube, 'wavelengths'):
        raise TypeError("datacube must have 'cube' and 'wavelength' attributes")
    cube = np.asarray(datacube.cube)
    wl = np.asarray(datacube.wavelengths)
    if cube.ndim != 3 or cube.shape[0] != wl.shape[0]:
        raise ValueError("cube must be shape (v, x, y) and len(wavelength) == v")

    v, x_dim, y_dim = cube.shape

    # Default slice indices: bottom at min wavelength, back and right faces at maxima
    wl_index = 0 if wl_index is None else int(wl_index)
    x_index = x_dim - 1 if x_index is None else int(x_index)
    y_index = y_dim - 1 if y_index is None else int(y_index)

    # Check slice ranges
    for name, idx, limit in (('wl_index', wl_index, v), ('x_index', x_index, x_dim), ('y_index', y_index, y_dim)):
        if not (0 <= idx < limit):
            raise ValueError(f"{name}={idx} out of range [0, {limit})")

    # Colormap normalization
    if vmin is None:
        vmin = float(cube.min())
    if vmax is None:
        vmax = float(cube.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)

    # Coordinate arrays
    x = np.arange(x_dim)
    y = np.arange(y_dim)

    # Bottom face (XY) at wl_index
    X_xy, Y_xy = np.meshgrid(x, y, indexing='ij')
    Z_xy = np.full_like(X_xy, wl[wl_index], dtype=float)
    C_xy = cube[wl_index, :, :]

    # Back face (X–W) at y_index
    W_xw, X_xw = np.meshgrid(wl, x, indexing='ij')
    Y_xw = np.full_like(W_xw, y_index, dtype=float)
    C_xw = cube[:, :, y_index]

    # Right face (Y–W) at x_index
    W_yw, Y_yw = np.meshgrid(wl, y, indexing='ij')
    X_yw = np.full_like(W_yw, x_index, dtype=float)
    C_yw = cube[:, x_index, :]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        X_xy, Y_xy, Z_xy,
        facecolors=cmap(norm(C_xy)),
        rstride=1, cstride=1, shade=False
    )
    ax.plot_surface(
        X_xw, Y_xw, W_xw,
        facecolors=cmap(norm(C_xw)),
        rstride=1, cstride=1, shade=False
    )
    ax.plot_surface(
        X_yw, Y_yw, W_yw,
        facecolors=cmap(norm(C_yw)),
        rstride=1, cstride=1, shade=False
    )

    # Labels and colorbar
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Wavelength')
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Intensity')

    plt.tight_layout()
    plt.show()
