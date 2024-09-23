import pytest
import numpy as np
from unittest import mock
from matplotlib import pyplot as plt
from wizard._exploration.plotter import normalize_layer, plotter

# Mock DataCube class and find_nex_smaller_wave function
class MockDataCube:
    def __init__(self, wavelengths, cube, name=None):
        self.wavelengths = wavelengths
        self.cube = cube
        self.name = name

@pytest.fixture
def mock_datacube():
    wavelengths = np.array([450, 500, 550, 600])
    cube = np.random.rand(4, 10, 10)
    name = "TestCube"
    return MockDataCube(wavelengths, cube, name)

@mock.patch("your_module.plotter.find_nex_smaller_wave", return_value=500)
def test_plotter_basic(mock_find_wave, mock_datacube):
    """Test plotter function with a basic DataCube input."""
    with mock.patch.object(plt, 'show'), mock.patch("your_module.plotter.normalize_layer") as mock_normalize:
        plotter(mock_datacube)
        # Check if normalize_layer is called
        mock_normalize.assert_called()

def test_normalize_layer_basic():
    """Test normalization of a simple numpy array."""
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    normalized = normalize_layer(arr)
    assert normalized.min() == 0
    assert normalized.max() == 1

def test_normalize_layer_warning(capfd):
    """Test if warning is printed when max is > 10 * mean."""
    arr = np.array([[1, 1, 1], [100, 1, 1]])
    normalize_layer(arr)
    captured = capfd.readouterr()
    assert 'The layer max value is more than 10 times greater' in captured.out

def test_normalize_layer_non_zero_min():
    """Test normalization when min value is non-zero."""
    arr = np.array([[2, 3, 4], [5, 6, 7]])
    normalized = normalize_layer(arr)
    assert normalized.min() == 0
    assert normalized.max() == 1

def test_normalize_layer_invalid_input():
    """Test if ValueError is raised for non-numpy input."""
    with pytest.raises(ValueError):
        normalize_layer("not an array")
