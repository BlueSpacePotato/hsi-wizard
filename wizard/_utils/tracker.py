"""
_utils/tracker.py
=================

.. module:: tracker
   :platform: Unix
   :synopsis: Tracker functions for monitoring DataCube changes.

Module Overview
---------------

This module contains functions to track changes made to DataCube instances.

Classes
-------

.. autoclass:: TrackExecutionMeta
   :members:
   :undoc-members:
   :show-inheritance:

"""

exculted = ['stop_recording', 'save_template', '_clean_data', '_map_args_to_kwargs', 'execute_template']


class TrackExecutionMeta(type):
    """
    A metaclass for tracking method executions in classes that use it (e.g., DataCube).

    This metaclass automatically wraps methods to log their calls when recording is enabled.
    Only methods marked as dynamic (with `__is_dynamic__ = True`) are recorded.
    Useful for debugging, auditing, or dynamically analyzing method usage patterns.

    Attributes:
        recording (bool): Flag indicating whether to track method calls.
        recorded_methods (list): Stores tuples of (method_name, args, kwargs) for recorded calls.

    Methods:
        start_recording(): Enables method tracking and resets previous records.
        stop_recording(): Disables method tracking.
        record_method(func): Decorator to wrap and conditionally log method calls.
    """

    recording = False
    recorded_methods = []

    def __new__(cls, name, bases, dct):
        """
        Overrides the class creation process to wrap methods for tracking.

        Args:
            name (str): Name of the class being created.
            bases (tuple): Base classes of the new class.
            dct (dict): Dictionary of attributes/methods in the class.

        Returns:
            type: A new class with wrapped methods for execution tracking.
        """
        for key, value in dct.items():
            # Wrap only dynamic methods or those that are not in the excluded list
            if callable(value) and key != 'execute_template':
                dct[key] = cls.record_method(value)
        return super().__new__(cls, name, bases, dct)

    @staticmethod
    def record_method(func):
        """
        Decorator to wrap a method and record its execution if it's marked as dynamic.

        Args:
            func (callable): The function to be wrapped.

        Returns:
            callable: A wrapped version of the original function.
        """
        def wrapper(*args, **kwargs):
            if TrackExecutionMeta.recording:
                if getattr(func, '__is_dynamic__', False):
                    print(f"Tracking dynamic method: {func.__name__}")
                    TrackExecutionMeta.recorded_methods.append(
                        (func.__name__, args, kwargs))
            return func(*args, **kwargs)
        return wrapper

    @staticmethod
    def start_recording():
        """
        Activates the tracking of dynamic method executions and clears previous records.
        """
        TrackExecutionMeta.recording = True
        TrackExecutionMeta.recorded_methods = []

    @staticmethod
    def stop_recording():
        """
        Deactivates the tracking of method executions.
        """
        TrackExecutionMeta.recording = False
