
from ..core import DataCube
import numpy as np

def inverse(dc: DataCube) -> DataCube:
    """
Invert the DataCube values.

This operation is useful for converting between transmission and
reflectance data, or similar inversions. The formula applied is:
`tmp = cube * -1`
`tmp += -tmp.min()`
The data type of the cube is preserved if it's 'uint16' or 'uint8'
after temporary conversion to 'float16' for calculation.

Parameters
----------
dc : DataCube
The DataCube to invert.

Returns
-------
DataCube
The DataCube with inverted values.

Examples
--------
>>> import wizard
>>> dc = wizard.read('example.fsm')
>>> dc.inverse()
    """
    dtype = dc.cube.dtype
    if dtype == np.uint16 or dtype == np.uint8:  # Use np types for comparison
        cube = dc.cube.astype(np.float32)
    else:
        cube = dc.cube.copy()

    tmp = cube
    tmp *= -1
    tmp += -tmp.min()

    dc.set_cube(tmp.astype(dtype))
    return dc



def normalize(dc: DataCube) -> DataCube:
    """
Normalize spectral information in the data cube to the range [0, 1].

For each 2D spatial layer in the DataCube, the normalization is performed by:
`layer = (layer - min_in_layer) / (max_in_layer - min_in_layer)`
This scales the intensity values of each layer independently across its
spatial dimensions.

Parameters
----------
dc : DataCube
The DataCube instance to normalize.

Returns
-------
DataCube
The normalized DataCube.

Examples
--------
>>> import wizard
>>> dc = wizard.read('example.fsm')
>>> dc.normalize()
    """
    cube = dc.cube.astype(np.float32)
    min_vals = cube.min(axis=(1, 2), keepdims=True)
    max_vals = cube.max(axis=(1, 2), keepdims=True)

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1

    cube = (cube - min_vals) / range_vals
    dc.set_cube(cube)
    return dc

