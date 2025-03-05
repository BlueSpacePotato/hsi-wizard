[![Documentation Status](https://readthedocs.org/projects/hsi-wizard/badge/?version=latest)](https://hsi-wizard.readthedocs.io)
[![codecov](https://codecov.io/gh/BlueSpacePotato/hsi-wizard/graph/badge.svg?token=85ASSSF2ZN)](https://codecov.io/gh/BlueSpacePotato/hsi-wizard)
![PyPI - Downloads](https://img.shields.io/pypi/dm/hsi-wizard)


# HSI Wizard

See Beyond the Visible: The Magic of Hyperspectral Imaging

<img src="./resources/imgs/hsi_wizard_logo.svg" alt="hsi_wizard_logo" style="width: 100%">


## Introduction

Welcome to the `hsi-wizard` package! This Python package provides a straightforward environment for hyperspectral imaging (HSI) analysis, supporting everything from basic spectral analysis to advanced machine learning and AI methods. Whether you're working with raw sensor data or pre-processed datasets, `hsi-wizard` offers a suite of tools to simplify and enhance your analysis workflow.

If you're new here, the best place to start is the [documentation](https://hsi-wizard.readthedocs.io), where you'll find detailed instructions on how to begin.


## Features
- DataCube Class for managing and processing HSI data.
- Spectral plotting and visualization.
- Clustering and spectral analytics.
- Tools for merging and processing HSI data.
- Data loaders for various file formats (e.g., NRRD, Pickle, TDMS, and XLSX).
- Decorators for method tracking, input validation, and execution time logging.

## Requirements
- [Python](https://www.python.org) >3.10

---

## Installation

### Via pip

You can install the package via pip:

```bash
pip install hsi-wizard
```

### Compile from Source

Alternatively, you can compile HSI Wizard from source:

```bash
python -m pip install -U pip setuptools wheel            # Install/update build tools
git clone https://github.com/BlueSpacePotato/hsi-wizard   # Clone the repository
cd hsi-wizard                                             # Navigate into the directory
python -m venv .env                                       # Create a virtual environment
source .env/bin/activate                                  # Activate the environment
pip install -e .                                          # Install in editable mode
pip install wheel                                         # Install wheel
pip install --no-build-isolation --editable .             # Compile and install hsi-wizard
```

---

# Usage

After installing the package, you can import the DataCube, read function, and plotter for quick HSI data analysis:
```python
from wizard import DataCube, read, plotter

# Load an HSI datacube from a file
datacube = read('path_to_file')

# Visualize the datacube
plotter(datacube)
```

---

## Contributing

If you would like to contribute to this project, feel free to fork the repository, make changes, and submit a pull request. Please ensure that you adhere to the code style and include tests for any new features.


