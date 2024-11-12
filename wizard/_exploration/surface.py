
from wizard import DataCube
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


def dc_cut_by_value(dc: DataCube, val: int, type: str) -> DataCube:
    """
    Cut cube by defined value.

    :param dc: DataCube
    :param val: Value
    :return:
    """
    dc.cube[dc.cube <= val] = 0
    return dc


def get_z_surface(cube, v):
    """
    Calculate the Surface for the Plot.

    :param cube: DataCube.cube data
    :param v: slice value
    """
    # Create an empty array for z with the same shape as the 2D slice of the cube
    z = np.zeros((cube.shape[1], cube.shape[2]))
    
    # Extract the v-th slice
    slice_v = cube[v, :, :]
    
    # Apply a mask for elements greater than 0
    mask = slice_v > 0
    
    # Assign values from the slice to z where the mask is True
    z[mask] = slice_v[mask]
    
    return z


def plot_surface(dc: DataCube, v:int):
    """
    Plot a surface from a DataCube Slice

    :param dc: DataCube with beatuifull data
    :param v: Index Value for the DataCube Slice
    """

    z = get_z_surface(dc.cube, 250)
    z = (z - z.min()) / (z.max() - z.min())
    X = range(dc.shape[1])
    Y = range(dc.shape[2])
    x, y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z.T, cmap=cm.coolwarm)
    plt.show()
