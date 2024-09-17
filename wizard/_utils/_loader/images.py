"""
_utils/_loader/images.py
====================================

.. module:: images
   :platform: Unix
   :synopsis: Provides images reader and writer.

Module Overview
---------------

This module includes reader and writer for images files.


Functions
---------

.. autofunction:: filter_image_files
.. autofunction:: images_from_folder_to_dc
.. autofunction:: load_image
.. autofunction:: image_to_dc

"""

import os
import numpy as np

from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from ..decorators import check_path
from ..._core import DataCube

def filter_image_files(files):
    """
    Filters a list of filenames, returning only those that have image file extensions.

    The function checks for the following image file extensions (case-insensitive):
    - .jpg
    - .jpeg
    - .png
    - .gif
    - .bmp
    - .tiff

    :param files: A list of filenames to be filtered for image file extensions.
    :type files: list[str]
    :returns: A list of filenames that have image file extensions.
    :rtype: list[str]

    :Example:

    >>> files = ["image.jpg", "document.pdf", "photo.png", "archive.zip"]
    >>> image_files = filter_image_files(files)
    >>> print(image_files)  # Output: ['image.jpg', 'photo.png']
    """

    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    return [file for file in files if any(file.lower().endswith(ext) for ext in image_extensions)]


@check_path
def images_from_folder_to_dc(path: str, **kwargs) -> DataCube:
    """
    Load a folder of images into a DataCube.

    #todo: add exclude parameter to avoid some files or to an
    include_only parameter

    :param path:
    :return:
    """
    _files = [os.path.join(path, f) for f in os.listdir(path)]

    _files_filtert = filter_image_files(_files)

    _dc = image_to_dc(_files_filtert, **kwargs)

    # put data in DataCube and return
    return _dc


def load_image(path):
    """
    Load an image from a specified file path.

    Parameters
    ----------
    path : str
        The file path to the image to be loaded.

    Returns
    -------
    ndarray
        The image read from the file, represented as a NumPy array.

    Examples
    --------
    >>> img = load_image('path/to/image.png')
    >>> plt.imshow(img)
    >>> plt.show()
    """
    return plt.imread(path)


def image_to_dc(path: str | list, **kwargs) -> DataCube:
    """
    Load image(s) into a DataCube.

    This function loads one or more images into a DataCube. It supports both a single image file path or a list of image file paths. The images are processed based on the specified type, which determines the transpose operation applied to the data.

    :param path: Path to an image file or a list of image file paths. If a list is provided, images are loaded concurrently.
    :type path: str or list[str]
    :param kwargs: Optional keyword arguments.
        - type: Specifies the transpose operation to apply to the data. Can be 'default' (default behavior) or 'pushbroom' (for pushbroom images).
        - Other keyword arguments may be accepted depending on the implementation of `load_image`.

    :returns: A DataCube object containing the image data.
    :rtype: DataCube

    :raises TypeError: If `path` is neither a string nor a list of strings.
    """

    type = kwargs.get('type', 'default')
    name = kwargs.get('name', None)

    if isinstance(path, str):
        img = load_image(path)
        data = np.transpose(np.array(img), (2, 0, 1))
        
    elif isinstance(path, list):

        def process_image(idx_file):
            idx, file = idx_file
            _img = load_image(file)
            return _img

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_image, enumerate(path)))

        data = np.array(results)

        if type == 'pushbroom':
            data = np.transpose(data, (1, 0, 2))
        else:
            data = np.transpose(data, (2, 0, 1))

    else:
        raise TypeError('Path must be string to a file or a list of files')
    
    return DataCube(data, name=name)
