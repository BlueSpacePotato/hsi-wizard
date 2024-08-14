import glob
import os

import re
import pathlib
import numpy as np
import pandas as pd

from tqdm import tqdm

from .fsm_reader import _read_fsm

from nptdms import TdmsFile
    
from matplotlib import pyplot as plt

from concurrent.futures import ThreadPoolExecutor

from .._core.datacube import DataCube

from wizard._utils.decorators import check_path, add_method


def get_files_by_extension(path: str, extension: str) -> list:
    """
    Return a sorted list of filenames of a given extension from a directory.

    :param path: path of a directory
    :type path: str
    :param extension: extension, for example `.csv`
    :rtype extension: str
    :return: list with filenames
    :rtype: list
    """
    # check if extension exists
    if not extension:
        return []

    # check if path is valid
    if not os.path.isdir(path):
        return []

    # check if extension doesn't start with `.`
    if not extension.startswith('.'):
        extension = '.' + extension

    return sorted(glob.glob(os.path.join(path, '*' + extension.lower())))


def make_path_absolute(path: str) -> str:
    """
    Check if the path is absolute. If not, convert it to an absolute path.

    :param path: Path to the file or directory
    :return: Absolute path to the file or directory
    :rtype: str
    """
    if isinstance(path, str):
        if not os.path.isabs(path):
            raise ValueError(f"Invalid path: {path}")
        return os.path.abspath(path).lower()
    else:
        raise ValueError("Input path must be a string.")


def to_cube(data, len_x, len_y) -> np.array:
    """
    Transform a 2d numpy array to a dc like array.

    data stored in Fortran-like index ordering, so using f-order instate c

    :param data: list of data in fortran order
    :param len_x: len dc x / pixel size x
    :param len_y: len dc y / pixel size y
    :return: transformed dc
    :rtype: np.array
    """
    len_v = data.shape[0]
    return data.reshape(len_v, len_x, len_y, order='F')


@check_path
@add_method(DataCube)
def read(path: str, datatype: str = 'auto', **kwargs) -> DataCube:
    """
    Read functions for importing data from different file types.

    :param path: data path to file
    :param datatype: data type of the file
    :param kwargs: additional keyword arguments
    :return: DataCube object
    """
    _datacube = None

    if datatype == 'auto':
        suffix = pathlib.Path(path).suffix
    else:
        suffix = datatype

    if suffix == '.csv':
        _datacube = __read_csv__(path, **kwargs)
    elif suffix == '.xlsx':
        _datacube = __read_xlsx__(path, **kwargs)
    elif suffix == '.fsm':
        _datacube = __read_fsm__(path, **kwargs)
    elif suffix == '.tdms':
        sample, dark, wave = read_tdms(path, **kwargs)
        _datacube = DataCube(sample, wavelengths=wave)
    elif suffix == '.jpg':
        _datacube = image_to_dc(path, **kwargs)
    elif suffix == '.npy':
        _datacube = load_npy(path, **kwargs)
    elif os.listdir(path):
        _datacube = images_from_folder_to_dc(path, **kwargs)
    else:
        raise NotImplementedError(f'No loader for {suffix}, '
                                  f'please parse your data manually.')

    return _datacube


def load_npy(path) -> DataCube:
    """
    Load numpy data.

    :param path:
    :return:
    """
    data = np.load(path)
    data = np.transpose(data, (2, 0, 1))

    # put data in DataCube and return
    return DataCube(data)


def load(data) -> DataCube:
    """
    Function for loading existing data into a DataCube.

    This function initializes a DataCube object from the provided data. It supports both numpy arrays and lists as input data formats.

    :param data: The data to be loaded into the DataCube. It can be either a numpy array or a list.
    :type data: numpy.ndarray or list
    :returns: A DataCube object initialized with the provided data, or None if the data format is unsupported.
    :rtype: DataCube or None
    """

    _datacube = None

    if isinstance(data, np.ndarray):
        _datacube = DataCube(cube=data)
    elif isinstance(data, list):
        _datacube = DataCube(cube=data)

    return _datacube


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
    _files = [os.path.join(path, f) for f in tqdm(os.listdir(path), desc='List Files')]

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

    if isinstance(path, str):
        img = load_image(path)
        data = np.transpose(np.array(img), (2, 0, 1))
        
    elif isinstance(path, list):

        def process_image(idx_file):
            idx, file = idx_file
            _img = load_image(file)
            return _img

        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_image, enumerate(path)), total=len(path), desc='Load Images'))

        data = np.array(results)

        if type == 'pushbroom':
            data = np.transpose(data, (1, 0, 2))
        else:
            data = np.transpose(data, (2, 0, 1))

    else:
        raise TypeError('Path must be string to a file or a list of files')
    
    return DataCube(data)


# todo
def __read_csv__(path: str) -> DataCube:
    """
    Read csv file.

    :param path: path to csv file
    :return:
    """
    raise NotImplementedError('Not Implemented yet')
    _datacube = DataCube(cube=None)

    return _datacube


# todo
def __read_xlsx__(path: str) -> DataCube:
    """
    Read xlsx file.

    :param path: path to xslx file
    :return:
    """
    raise NotImplementedError('Not Implemented yet')
    _datacube = DataCube(cube=None)

    return _datacube


def __read_fsm__(path: str) -> DataCube:
    """
    Read function for fsm-files from perkin elmer. Tested with FTIR-Data.

    ToDo: missing error handling if `path` is wrong.

    :param path: str, path to file
    :return:
    """
    # load data - load spectra from path
    # fsm_spectra = specread(path)
    fsm_spectra, fsm_wave, fsm_meta = _read_fsm(path)

    # load data - load x&y lens
    fsm_len_x = fsm_meta['n_x']
    fsm_len_y = fsm_meta['n_y']
    # fsm_len_y = fsm_meta['n_z']

    # load data - start and stop from the wavelength
    # fsm_wave_start = fsm_meta['z_start']
    # fsm_wave_end = fsm_meta['z_end']

    # load data - wavelength
    fsm_wave = fsm_wave.astype('int')

    # load data - data
    # fsm_data = fsm_spectra.amplitudes

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # convert 2d df in 3d np.array
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fsm_data_cube = to_cube(data=fsm_spectra.T, len_x=fsm_len_x, len_y=fsm_len_y)

    # may return with transposed date
    _datacube = DataCube(fsm_data_cube, wavelengths=fsm_wave,
                         name='.fsm', notation='cm-1')
    return _datacube


def read_tdms(path: str = None) -> DataCube:
    """
    Read function for tdms file.

    The functions reads and pareses the in `tdms_file` defined file and
    returns the data in a defined way. The return value can be `np` or df`.

    :param path: Path in string format with reference to the file to
                 be read.
    :return: datacube
    :rtype: wizard.DataCube
    """
    # type for automatic detection
    data_type = ''
    wave_col = 0
    len_col = 0

    # get values
    file = TdmsFile(path)

    # build df
    tdms_df = file.as_dataframe()

    # copy cols
    col = tdms_df.columns
    col_new = []
    col_sample = []
    col_raw = []

    # sort by dark and normal
    for i in col:

        i = i.replace(' ', '').replace('\'', '')

        if i.find('RAW') >= 1:
            col_raw.append(i)
        elif (i.find('DarkCurrent') >= 1 or i.find('cm') >= 1 or i.find('nm') >= 1):
            pass
        else:
            col_sample.append(i)

        col_new.append(i)

    # rename cols
    tdms_df.columns = col_new

    if any("RAMAN" in s for s in col_new):
        data_type = 'raman'
        wave_col = 1
        len_col = 4

    if any("NIR" in s for s in col_new) or any("KNIR" in s for s in col_new):
        data_type = 'nir'
        wave_col = 1
        len_col = 3

    if any("VIS" in s for s in col_new) or any("KVIS" in s for s in col_new):
        data_type = 'vis'
        wave_col = 1
        len_col = 3

    # get wave length
    # wave_col: - 1 raman for cm-¹; -2 for nm
    wave = np.array(tdms_df[tdms_df.columns[-wave_col]])

    # parse length information
    # len_col: -3 for other,  -4 for raman
    len_xy = re.findall(r'\d+', col_new[-len_col])
    # len_v = wave.shape[0]
    len_x = int(len_xy[0]) + 1
    len_y = int(len_xy[1]) + 1

    # set index
    tdms_df = tdms_df.set_index(tdms_df.columns[-2])

    # cops into new df
    tdms_sample_df = tdms_df[col_sample].copy()
    # tdms_raw_df = tdms_df[col_raw].copy()

    # clean up
    del tdms_df

    tdms_sample = np.array(tdms_sample_df)
    # tdms_raw = np.array(tdms_raw_df)

    tdms_sample_cube = to_cube(data=tdms_sample, len_x=len_x, len_y=len_y)
    # tdms_raw_cube = to_cube(data=tdms_raw, len_x=len_x, len_y=len_y)

    wave = wave.astype('int')

    return DataCube(
        cube=tdms_sample_cube,
        wavelengths=wave,
        name=data_type
    )


def write_xlsx(datacube: np.array, wavelenghts: np.array, filename: str):
    """
    Write out a .xlsx file.

    :param datacube:
    :param wavelenghts:
    :param filename:
    :return:
    """
    shape = datacube.shape

    df = pd.DataFrame()

    cols = []

    for i in wavelenghts:
        cols.append(str(i))

    idx = []

    for y in range(shape[1]):
        for x in range(shape[0]):
            spec_ = datacube[x, y, :]

            df_tmp = pd.DataFrame(spec_).T

            df = df.append(df_tmp)

            idx.append(f'x:{x}; y:{y}')

    df.columns = cols

    df.insert(0, column='Point', value=idx)

    df = df.set_index('Point')

    df.to_excel(f'{filename}.xlsx')


def merge_cubes(cube1: DataCube, cube2: DataCube) -> DataCube:
    """
    Merge to datacubes to a new one.

    :param cube1:
    :param cube2:
    :return:
    """
    c1 = cube1.cube
    c2 = cube2.cube
    if c1.shape[:2] == c2.shape[:2]:
        c3 = np.concatenate(c1, c2)
    else:
        c3 = None
        raise NotImplementedError('Sorry - '
                                  'This function is not implemented yet.'
                                  'At the moment you just can merge cubes'
                                  ' with the same size x,y.')
    return DataCube(c3)


def merge_waves(wave1: list, wave2: list) -> list:
    """
    Merge two wave lists.

    todo: better merge algorithms

    :param wave1: first list with waves
    :param wave2: second list with waves
    :return: merged waves
    :rtype: list
    """
    if common_members(wave1, wave2):
        raise NotImplementedError('Sorry - your wavelengths are overlapping,'
                                  ' we working on a solution')

    return wave1 + wave2


def common_members(a: list, b: list) -> set:
    """
    Check for comon members between two lists.

    :param a: list a
    :param b: list to compare to a
    :return: return a set of common members
    :rtype: set
    """
    a_set = set(a)
    b_set = set(b)
    return a_set & b_set
