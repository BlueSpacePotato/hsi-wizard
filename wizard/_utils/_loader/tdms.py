"""
_utils/_loader/tdms.py
===========

.. module:: tdms
   :platform: Unix
   :synopsis: read and write .tdms-files

Module Overview
---------------

This module includes reader and writer for .tdms-files.

Functions
---------
.. autofunction:: read_tdms

"""

import re
import numpy as np
from nptdms import TdmsFile

from ._helper import to_cube
from ..._core import DataCube

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