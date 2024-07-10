[![Documentation Status](https://readthedocs.org/projects/hsi-wizard/badge/?version=latest)](https://hsi-wizard.readthedocs.io/en/latest/?badge=latest)

<br/>
<center style="font-size: 18px; font-weight: bold ">
See Beyond the Visible: The Magic of Hyperspectral Imaging
</center>
<br/>
<img src="./resources/imgs/logo/linkedin_banner_image_2.png" alt="">
<br/>
<br/>

# HSI-Wizard
The goal is to set up a straightforward environment for hyperspectral analysis. The HIS-Wizard provides a range of useful tools for this purpose, spanning from basic spectral analysis to advanced methods utilizing artificial intelligence.

## Features
- DataCube Class
- Spectral plotting function
- Clustering
- Spectral Analytics
- Maldi Analytics
- Merge Spectral Data
- Hyperspectral Imaging

## Requirements
- [Python](https://www.python.org) >3.10


## Installation

### pip
When utilizing pip, hsi-wizard releases are accessible in the form of source packages and binary wheels. Before proceeding with the installation of hsi-wizard and its prerequisites, ensure that your pip, setuptools, and wheel are updated to the latest versions

```
pip install hsi-wizard
```

### Compile from source
An alternative method for installing HSIWizard involves cloning its GitHub repository and compiling it from source. This approach is commonly chosen when modifications to the code are desired. It is essential to have a development environment set up, which should include a Python distribution with header files, a compiler, as well as installations of pip and git.

```
python -m pip install -U pip setuptools wheel            # install/update build tools
git clone https://github.com/BlueSpacePotato/hsi-wizard  # clone hsi-wizard
cd hsi-wizard                                            # navigate into dir
python -m venv .env                                      # create environment in .env
source .env/bin/activate                                 # activate virtual env
pip install -e .                                         # install requirements
pip install wheel                                        # install wheel
pip install --no-build-isolation --editable .            # compile and install hsi-wizard
```


## Documentation

### [Click here for Docs!](https://hsi-wizard.readthedocs.io/en/latest/)

The Documentation is available as readthedocs project. Build with `sphinx` and the `sphinx_rtd_theme`.

## The Git Structure
* `docs/`data for the sphinx/readthedocs implementation
* `resources/` for storing images and sample data and equivalent
* `wizard/` the source code for the `hsi-wizard` python-lib that can be used as stand alone
* `.github/workflows/` .yaml-files for autobuild etc

## Code Philosophy

* keep things simple
* implement only the smallest amount of code, to solve a problem
* don't make up, no existing problems
* try to solve problems the easiest way first
* build reliability code
* write usefull comands

## Definitions
To build a clean code and communicate the ideas the right way, we need to define some basic understandings.

### DataCube
- A Datacube is a 3D array with shape `vxy`
- `x` and `y` values describe the number of pixels in each direction
- `v` values (often called λ in papers) describe the information deapth of the spectrum, commanly as measured counts.

```python3
from matplotlib import pyplot as plt
from hsi_wizard import datacube as dc

len_v = 50
len_x = 5
len_y = 6

# define empty array with given shape
data_cube = dc.DataCube()

# get the spectru for a single pixel and plot it
spectrum = data_cube[:, 3, 3]
plt.plot(spectrum)
plt.show()

# show 2d image for channel 3
img = datacube[3]
plt.imshow(img_2d)
plt.show()
```

### Difference read and load
As `loading` function is used to import already processed data. For example if you want to load in an already existing numpy array. A `read` function on the other hand, reads dedicate files, like a `*.csv` or `*.fsm` file.

### Pre-Processing Level
Based on an Idea from [DOI](www.doi.org/10.1007/s40010-017-0433-y)
* Level 0: Data is captured directly from sensor
* Level 1: Data is processed in a easy way
* Level 2: Data is hardly processed

---
## To Dos
- [ ] better hyperparameter tuning with evol [source](https://github.com/godatadriven/evol)
- [ ] R-support with patsy [source](https://github.com/pydata/patsy)
- [ ] better template-creator
- [ ] merge function for multiple specs
- [ ] spec appending function
- [ ] save file as nrrd

---

## Changelog
The changelog will be added if the beta version is fine and runs stable

---
# Acknolagement

Thanks to [shopify](https://www.shopify.com/de) providing a free logo build with the free [hatchful](https://www.shopify.com/de/tools/logo-maker) logo-generator.

Icons made by <a href="https://www.flaticon.com/authors/good-ware" title="Good Ware">Good Ware</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>

