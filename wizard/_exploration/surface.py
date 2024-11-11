
from wizard import DataCube
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib import cm

# matplotlib.use('TkAgg')


def dc_cut_by_value(dc: DataCube, val: int, type: str) -> DataCube:
    """
    cut cube by value

    :param dc:
    :param val:
    :return:
    """
    dc.cube[dc.cube <= val] = 0
    return dc


def plotting(dc: DataCube):
    """

    :param dc:
    :return:
    """
    color_map = plt.get_cmap('spring')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = dc.cube.nonzero()
    view = ax.scatter(x, y, z, cmap=cm.coolwarm, c=z)
    plt.show()

"""
def get_z_surface(cube, v):
    z = np.empty(shape=(cube.shape[1], cube.shape[2]))
    
    for y in range(cube.shape[1]):
        for x in range(cube.shape[2]):
                if cube[v , y, x] > 0:
                    z[y, x] =  cube[v , y, x]
    return z
"""

def get_z_surface(cube, v):
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

    z = get_z_surface(dc.cube, 250)
    z = (z - z.min()) / (z.max() - z.min())
    X = range(dc.shape[1])
    Y = range(dc.shape[2])
    x, y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z.T, cmap=cm.coolwarm)
    plt.show()
