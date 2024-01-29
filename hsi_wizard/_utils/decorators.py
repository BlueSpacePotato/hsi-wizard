"""Decorators to handle behavier."""
import os
import time

from functools import wraps

import numpy as np


def check_load_dc(func) -> np.array:
    """Check if the loading function is correctly defined.

    :param func:
    :return: func
    :rtype: method
    """
    def wrapper(*args, **kwargs):
        # process function
        cube = func(*args, **kwargs)

        if cube != 'no implementation':
            # check if np array
            if cube is not np.array:
                raise ValueError('Loading function should return a np array')

            if 2 < len(cube.shape) <= 4:
                raise ValueError('loading function is not valid. The return'
                                 'shape should be (z|x|y) or (z|x|y|v)')
        else:
            cube = None
        return cube
    return wrapper


def check_path(func):
    """Check if data path is valid, if not throw exception.

    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):

        path = args[0] if kwargs.get('path') is None else kwargs.get('path')

        if path is None:
            raise ValueError('no path given')

        if not os.path.exists(path):
            raise FileNotFoundError(f'Given path is not valid. {path}')

        # process function
        return func(*args, **kwargs)

    return wrapper


def add_method(cls):
    """Add method to class.

    :param cls:
    :return:

    source: [Michael Garod @ Medium](https://mgarod.medium.com/
    dynamically-add-a-method-to-a-class-in-python-c49204b85bd6)
    """
    def decorator(func):
        @wraps(func)
        # def wrapper(self, *args, **kwargs):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but
        # does exactly the same as func
        return func  # returning func means func can still be used normally
    return decorator


def check_time(func):
    """Check execution time of an function.

    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        x = stop = time.time()
        print(stop-start)
        return x
    return wrapper


def add_to_workflow(func):
    """Add a function and the parameter to an template.

    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):
        print(*args, **kwargs)

        return func(*args, **kwargs)
    return wrapper()


def check_limits(func) -> np.array:
    """Force clipping limits to an image or array.

    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):
        # store data type
        dtype = args[0].dtype

        # process function
        image = func(*args, **kwargs)

        # check limits
        if dtype == 'uint8':
            image = np.clip(image, 0, 255).astype(dtype)
        if dtype in ['float32', 'float64']:
            image = np.clip(image, 0, 1).astype(dtype)

        return image
    return wrapper
