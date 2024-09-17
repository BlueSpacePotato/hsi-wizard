"""
_utils/_loader/__init__.py
===========

.. module:: __ini__.py
   :platform: Unix
   :synopsis: __init__ for _file folder

Module Overview
---------------

Inits loader and writer functions.

Functions
---------
.. autofunction:: register_loader
.. autofunction:: read
.. autofunction:: load_all_loaders

"""

import pathlib
import importlib

# Dictionary to register loaders based on file extensions
LOADER_REGISTRY = {}

# Populate __all__ to control the public API
__all__ = ['read']

def register_loader(extension, function_name):
    """
    Register a new loader for a specific file extension.
    :param extension: file extension (e.g., '.csv')
    :param function_name: loader function (e.g., read_csv)
    """
    LOADER_REGISTRY[extension] = function_name

def read(path: str, datatype: str = 'auto', **kwargs):
    """
    Read functions for importing data from different file types.

    :param path: data path to file
    :param datatype: data type of the file (e.g., '.csv', '.xlsx')
    :param kwargs: additional keyword arguments
    :return: DataCube object
    """
    if datatype == 'auto':
        suffix = pathlib.Path(path).suffix
    else:
        suffix = datatype

    # Get the loader function based on the file extension
    loader_function = LOADER_REGISTRY.get(suffix)

    if loader_function:
        return loader_function(path, **kwargs)
    else:
        raise NotImplementedError(f'No loader for {suffix}, '
                                  f'please parse your data manually.')

# Dynamic import of loaders
def load_all_loaders():
    """
    Automatically discover and import loaders from the wizard._utils._loader package.
    """
    loader_modules = [
        "csv",
        "xlsx",
        "tdms",
        "fsm",
    ]

    for module_name in loader_modules:
        module = importlib.import_module(f'wizard._utils._loader.{module_name}')
        for attr_name in dir(module):
            if attr_name.startswith('read_'):
                # Assuming the function name is read_csv, read_xlsx, etc.
                extension = '.' + attr_name.split('_')[1]  # e.g., 'read_csv' -> '.csv'
                loader_function = getattr(module, attr_name)
                register_loader(extension, loader_function)

# Load all loaders dynamically
load_all_loaders()


