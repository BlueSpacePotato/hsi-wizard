exculted = ['stop_recording', 'save_template']


class TrackExecutionMeta(type):
    recording = False
    recorded_methods = []
    def __new__(cls, name, bases, dct):
        for key, value in dct.items():
            if callable(value) and key != 'execute_template':
                dct[key] = cls.record_method(value)
        return super().__new__(cls, name, bases, dct)

    @staticmethod
    def record_method(func):
        def wrapper(*args, **kwargs):
            if TrackExecutionMeta.recording:
                print(func.__name__)
                if func.__name__ not in exculted:
                    print('execute!')
                    TrackExecutionMeta.recorded_methods.append((func.__name__, args, kwargs))
            return func(*args, **kwargs)
        return wrapper

    @staticmethod
    def start_recording():
        TrackExecutionMeta.recording = True
        TrackExecutionMeta.recorded_methods = []

    @staticmethod
    def stop_recording():
        TrackExecutionMeta.recording = False