#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2012 - 2013
# Matías Herranz <matiasherranz@gmail.com>
# Joaquín Tita <joaquintita@gmail.com>
#
# https://github.com/PyRadar/pyradar/blob/master/pyradar/classifiers/isodata.py
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import matplotlib.pyplot as plt

import wizard
from wizard._processing.cluster import isodata

path = 'hsi-open-dataset/data/coffee'

# Read datacube
dc = wizard.read(path)

# Create a figure for subplots
fig, axes = plt.subplots(1, 5, figsize=(20, 5))  # Adjust number of subplots and figure size
i=0

# Step 1: Original data
axes[i].imshow(dc.cube[0], cmap='gray')  # Display the first layer of the original datacube
axes[i].set_title('Original Data')
i+=1

# Step 2: Remove vingeting
dc.remove_vingetting(axis=1, slice_params= {"start": None, "end": 50})
axes[i].imshow(dc.cube[0], cmap='gray')  # Display the first layer after registration
axes[i].set_title('After removing vingetting')
i+=1


# Step 3: Register layers
dc.register_layers()
axes[i].imshow(dc.cube[0], cmap='gray')  # Display the first layer after registration
axes[i].set_title('After Registration')
i+=1

# Step 4: Remove background
dc.remove_background(style='bright', threshold=75)
axes[i].imshow(dc.cube[0], cmap='gray')  # Display the first layer after background removal
axes[i].set_title('After Background Removal')
i+=1

# Step 5: ISODATA clustering
res = isodata(dc, k=7)
axes[i].imshow(res, cmap='plasma')  # Display the clustering result
axes[i].set_title('ISODATA Clustering')
i+=1
# Adjust layout and show the plot
plt.tight_layout()
plt.show()
