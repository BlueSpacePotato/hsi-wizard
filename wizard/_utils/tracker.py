"""DataCube Tracker to keep record of used methods and functions."""
exculted = ['stop_recording', 'save_template']


class TrackExecutionMeta(type):
    """
    DataCube tracker.

    Keep track of used functions and methods.
    """

    recording = False
    recorded_methods = []

    def __new__(cls, name, bases, dct):
        """
        Magic Method new.

        :return:
        """
        for key, value in dct.items():
            if callable(value) and key != 'execute_template':
                dct[key] = cls.record_method(value)
        return super().__new__(cls, name, bases, dct)

    @staticmethod
    def record_method(func):
        """
        Record Decorator.

        :return: func
        """
        def wrapper(*args, **kwargs):
            if TrackExecutionMeta.recording:
                print(func.__name__)
                if func.__name__ not in exculted:
                    print('execute!')
                    TrackExecutionMeta.recorded_methods.append(
                        (func.__name__, args, kwargs))
            return func(*args, **kwargs)
        return wrapper

    @staticmethod
    def start_recording():
        """
        Start tracker.

        :return: None
        """
        TrackExecutionMeta.recording = True
        TrackExecutionMeta.recorded_methods = []

    @staticmethod
    def stop_recording() -> None:
        """
        Stop tracker.

        :return: None
        """
        TrackExecutionMeta.recording = False
