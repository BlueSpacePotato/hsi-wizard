"""
_core/datacube_ops.py

.. module:: datacube_ops
   :platform: Unix
   :synopsis: DataCube Operations.

## Module Overview
This module contains operation function for processing datacubes.

## Functions
.. autofunction:: remove_spikes
.. autofunction:: resize
"""
import os
import cv2
import copy

import random
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.transform import warp


from wizard.core import DataCube
from ..processing.spectral import calculate_modified_z_score, spec_baseline_als
from .._utils.helper import _process_slice, feature_registration, RegistrationError, auto_canny, decompose_homography, normalize_polarity











