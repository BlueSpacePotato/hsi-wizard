# wizard/__init__.py


# Import necessary submodules and classes/functions from them
from ._core.datacube import DataCube
from ._core.eda import plotter

from ._utils.data_loader import read
# from ._utils.utils_module1 import UtilityClass1, util_function1
# from ._utils.utils_module2 import UtilityClass2, util_function2

# Define what should be accessible when using 'from wizard import *'
__all__ = [
    'DataCube',
    'read'
]

# Example of setting package metadata
__version__ = "0.0.1"
__author__ = 'flx'


# Optionally, define a function for package initialization or configuration
def initialize():
    print("Initializing the wizard package")


# Initialize the package
initialize()
