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

path = '/Users/flx/Documents/data/Tom_MA/MIR/'

# img = plt.imread(path)
dc = wizard.read(path=path)  # from folder into datacube
dc.remove_background()
res = isodata(dc, k_=5, k=10)

plt.imshow(res, cmap='plasma')
plt.show()
