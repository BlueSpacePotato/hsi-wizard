.. _datacube:

DataCube
--------
.. module:: datacube
   :platform: Unix
   :synopsis: DataCube class for storing hyperspectral imaging (HSI) data.

Module Overview
***************
This module provides the `DataCube` class, which is designed to store and manage hyperspectral imaging (HSI) data. The `DataCube` is represented as a 3D array, where the `x` and `y` axes represent spatial dimensions (pixels), and the `v` axis represents values like spectral counts, channels, or wavelengths.

The `DataCube` class offers methods for manipulating and interacting with hyperspectral data, including arithmetic operations between cubes, data access, and wavelength management. Additionally, the class can track method executions and save them as reusable templates

Classes
*******
DataCube Methods
---------------

.. automethod:: wizard._core.datacube.DataCube.__init__

Initialize a new `DataCube` instance.

.. automethod:: wizard._core.datacube.DataCube.__add__

Add two `DataCube` instances.

.. automethod:: wizard._core.datacube.DataCube.__len__

Return the number of layers (v dimension) in the data cube.

.. automethod:: wizard._core.datacube.DataCube.__getitem__

Get an item from the data cube.

.. automethod:: wizard._core.datacube.DataCube.__setitem__

Set an item in the data cube.

.. automethod:: wizard._core.datacube.DataCube.set_name

Set a name for the `DataCube`.

.. automethod:: wizard._core.datacube.DataCube.set_wavelengths

Set wavelength data for the `DataCube`.

.. automethod:: wizard._core.datacube.DataCube.set_cube

Set data for the `DataCube`.

.. automethod:: wizard._core.datacube.DataCube.set_notation

Update the notation for the `DataCube`.

.. automethod:: wizard._core.datacube.DataCube.start_recording

Start recording method execution for the `DataCube`.

.. automethod:: wizard._core.datacube.DataCube.stop_recording

Stop recording method execution for the `DataCube`.

.. automethod:: wizard._core.datacube.DataCube.save_template

Save a template of recorded methods to a YAML file.

.. automethod:: wizard._core.datacube.DataCube.execute_template

Load a template and execute the corresponding methods.

Examples
********
Here is an example of how to use the `DataCube` class:

.. code-block:: python

    from wizard._core.datacube import DataCube

    # Create a DataCube instance
    cube_data = np.random.random((10, 100, 100))
    wavelengths = np.linspace(400, 700, 10)
    dc = DataCube(cube=cube_data, wavelengths=wavelengths, name="ExampleCube", notation="nm")
    dc.set_name("TestDataCube")

    # Access data
    print(dc[0])  # Access the first layer
    print(dc.wavelengths)  # Access wavelengths

    # Record method and save
    dc.start_recording()
    dc.resize(120, 120)
    dc.save_template("methods_template.yml")
    dc.stop_recording()

    # Create new DataCube
    dc2 = DataCube(cube=cube_data, wavelengths=wavelengths, name="ExampleCube", notation="nm")

    # Execute Template
    dc2.execute_template("methods_template.yml")