import os
import sys
import copy
import argparse
from pathlib import Path

import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.widgets import Slider, Button, TextBox

from wizard import DataCube
from .._utils.helper_functions import find_nex_smaller_wave

import imageio

from tqdm import tqdm

try:
    from IPython import get_ipython
    if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
        raise ImportError("console")
except:
    matplotlib.use('TkAgg')


def normalize_layer(layer: np.array) -> np.array:
    """
    deepcopy layer and scale form 0.0 / 1.0
    :param layer:
    :return:
    """
    
    if layer.max() > layer.mean() * 10:
        print(f'\033[93mThe layer max value is more then 10 times greater then the mean value.'
              f' If you dont see anything try the spike removing tool.\033[0m')
    
    l = copy.deepcopy(layer)

    # remove offset
    if l.min() != 0:
        #todo: some errors?
        l = l - layer.min()

    # scale to 1.0
    if l.max() != 0:
        l= l/l.max()

    # some cleaning
    l = np.round(l, decimals=10).astype('float16')

    return l

def plotter(dc: DataCube) -> None:

    global layer_id
    global x_id
    global y_id

    layer_id = dc.wavelengths[0]
    x_id = 0
    y_id = 0

    def update_plot(val):
        #imshow.set_cmap(build_cmap(int(slider_threshold.val)))

        # new image
        layer = dc.cube[np.where(dc.wavelengths == layer_id)[0][0]] # todo [0] or [0][0]
        layer = normalize_layer(layer)
        imshow.set_data(layer)


        #new spec
        spec = dc.cube[:, x_id, y_id]
        wave = range(0, dc.cube.shape[0]) if dc.wavelengths is None else dc.wavelengths
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
            # todo : check whats wrong with axis x, y
            y_id = int(event.xdata)
            x_id = int(event.ydata)
            update_plot(event)

        elif event.inaxes == ax[1]:
            tmp_id = int(event.xdata)
            layer_id = find_nex_smaller_wave(dc.wavelengths, tmp_id, 10)
            if layer_id != -1:
                update_plot(event)

    # define subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)
    fig.suptitle('Datacube' if dc.name is None else dc.name )

    # image
    layer = normalize_layer(dc.cube[0])
    imshow = ax[0].imshow(layer)

    # spec
    spec = dc.cube[:, 0, 0]
    wave = range(0, dc.cube.shape[0]) if dc.wavelengths is None else dc.wavelengths

    line = ax[1].axvline(
        x=layer_id,
        color='lightgrey',
        linestyle='dashed',
    )


    s_plot, = ax[1].plot(wave, spec)
    # add widgets
    axcolor = 'lightgoldenrodyellow'

    # connect clicked
    fig.canvas.mpl_connect("button_press_event", onclick_select)

    update_plot(None)
    plt.show()

