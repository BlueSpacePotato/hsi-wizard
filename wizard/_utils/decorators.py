"""Decorators to handle behavier."""
import os
import time

from functools import wraps

import numpy as np


def check_load_dc(func) -> np.array:
    """
    Check if the loading function is correctly defined.

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

            if 2 < len(cube.shape) <= 3:
                raise ValueError('loading function is not valid. The return'
                                 'shape should be (v|x|y)')
        else:
            cube = None
        return cube
    return wrapper


def check_path(func):
    """
    Check if data path is valid, if not throw exception.

    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):

        path = args[0] if kwargs.get('path') is None else kwargs.get('path')

        if path is None or path == '':
            raise ValueError('no path given')

        if not os.path.exists(path):
            raise FileNotFoundError(f'Given path is not valid. {path}')

        # process function
        return func(*args, **kwargs)

    return wrapper


def add_method(cls):
    """
    Add method to class.

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


def track_execution_time(func):
    """
    Decorator function to track the execution time of a function in milliseconds.

    Parameters
    ----------
    func : callable
        The function to be decorated.

    Returns
    -------
    callable
        A wrapped function that executes `func` and prints the execution time in milliseconds.

    Notes
    -----
    This decorator uses `time.time()` to measure the start and end times of `func` execution,
    computes the elapsed time, and prints it formatted to four decimal places.

    Example
    -------
    >>> @track_execution_time
    ... def example_function():
    ...     return sum(range(1000000))
    ...
    >>> example_function()
    Function 'example_function' executed in 2.1200 ms
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} ms")
        return result
    return wrapper


def add_to_workflow(func):
    """
    Add a function and the parameter to an template.

    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):
        print(*args, **kwargs)

        return func(*args, **kwargs)
    return wrapper()


def check_limits(func) -> np.array:
    """
    Force clipping limits to an image or array.

    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):
        # store data type
        dtype = args[0].dtype

        # process function
        image = func(*args, **kwargs)

        if dtype in ['float32', 'float64']:
            image = np.clip(image, 0, 1).astype(dtype)

        return image
    return wrapper
