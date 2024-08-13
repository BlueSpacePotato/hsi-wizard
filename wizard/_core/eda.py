import copy
import numpy as np
import matplotlib.pyplot as plt
from wizard import DataCube
from .._utils.helper_functions import find_nex_smaller_wave

try:
    from IPython import get_ipython
    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except ImportError:
    import matplotlib
    matplotlib.use('TkAgg')
except Exception as e:
    print(f"An unexpected error occurred: {e}")


def normalize_layer(layer: np.array) -> np.array:
    """
    Normalize a 2D or 3D numpy array (layer) by adjusting its values to a 0-1 range.

    This function first checks if the maximum value in the layer is more than 10 times greater than the mean value and prints a warning if so. It then normalizes the layer by removing any offset and scaling the values to the range [0, 1]. The resulting layer is rounded and cast to `float16` for consistency.

    :param layer: A numpy array representing the layer to be normalized.
    :type layer: np.ndarray
    :returns: The normalized layer as a numpy array with values scaled to the range [0, 1] and rounded to 10 decimal places.
    :rtype: np.ndarray
    :raises ValueError: If the input `layer` is not a numpy array.
    """
    if layer.max() > layer.mean() * 10:
        print('\033[93mThe layer max value is more than 10 times greater than the mean value.'
              ' If you don’t see anything, try the spike removing tool.\033[0m')
    
    layer_copy = copy.deepcopy(layer)

    if layer_copy.min() != 0:
        layer_copy -= layer.min()

    if layer_copy.max() != 0:
        layer_copy /= layer_copy.max()

    layer_copy = np.round(layer_copy, decimals=10).astype('float16')

    return layer_copy


def plotter(dc: DataCube) -> None:
    """
    Interactive plotter for visualizing and analyzing data from a DataCube.

    This function creates an interactive plot with two subplots: one for displaying an image layer from the DataCube and another for displaying a spectral plot. Users can interact with the plots to select different image layers and regions of interest, which updates both the image and spectral plot accordingly.

    :param dc: A DataCube object containing the data to be visualized. The DataCube should have attributes `wavelengths` and `cube`, where `cube` is a 3D numpy array and `wavelengths` is a list or array of wavelength values corresponding to the layers of the cube.
    :type dc: DataCube
    :returns: None
    :raises AttributeError: If the DataCube object does not have the required attributes (`wavelengths` and `cube`).
    :raises ValueError: If the DataCube object does not have a `name` attribute, but the code assumes it exists.

    Example usage:
    >>> plotter(my_datacube)
    """
    global layer_id
    global x_id
    global y_id

    layer_id = dc.wavelengths[0]
    x_id = 0
    y_id = 0

    def update_plot(val):
        layer = dc.cube[np.where(dc.wavelengths == layer_id)[0][0]]
        layer = normalize_layer(layer)
        imshow.set_data(layer)

        spec = dc.cube[:, x_id, y_id]
        wave = range(dc.cube.shape[0]) if dc.wavelengths is None else dc.wavelengths
        s_plot.set_data(wave, spec)
        r = (spec.max() - spec.min()) * 0.1
        ax[1].set_ylim(spec.min() - r, spec.max() + r)
        line.set_data(layer_id, (layer.min(), layer.max()))
        fig.canvas.draw_idle()

    def onclick_select(event):
        global layer_id
        global x_id
        global y_id

        if event.inaxes == ax[0]:
            y_id = int(event.xdata)
            x_id = int(event.ydata)
            update_plot(event)

        elif event.inaxes == ax[1]:
            tmp_id = int(event.xdata)
            layer_id = find_nex_smaller_wave(dc.wavelengths, tmp_id, 10)
            if layer_id != -1:
                update_plot(event)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)
    fig.suptitle('Datacube' if dc.name is None else dc.name)

    layer = normalize_layer(dc.cube[0])
    imshow = ax[0].imshow(layer)

    spec = dc.cube[:, 0, 0]
    wave = range(dc.cube.shape[0]) if dc.wavelengths is None else dc.wavelengths

    line = ax[1].axvline(
        x=layer_id,
        color='lightgrey',
        linestyle='dashed',
    )

    s_plot, = ax[1].plot(wave, spec)

    fig.canvas.mpl_connect("button_press_event", onclick_select)

    update_plot(None)
    plt.show()
