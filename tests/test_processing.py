import pytest
import numpy as np

import wizard

# Importing spectral processing functions
from wizard.processing.spectral import (
    smooth_savgol,
    smooth_moving_average,
    smooth_butter_lowpass,
    calculate_modified_z_score,
    get_ratio_two_specs,
    get_sub_tow_specs,
    signal_to_noise,
    del_leading_zeros,
    del_last_zeros,
)


# Fixtures for sample data
@pytest.fixture
def sample_spectrum():
    """Fixture: Generate a sample spectrum (range of 100 values)."""
    return np.array(range(100))


@pytest.fixture
def sample_baseline_spectrum():
    """Fixture: Generate a sample baseline spectrum (range of 100 values)."""
    return np.array(range(100))


@pytest.fixture
def sample_wavelengths():
    """Fixture: Generate a sample of wavelengths (10 values between 400 and 800)."""
    return np.linspace(400, 800, 10)


# --------------- Spectral Processing Tests ---------------
class TestSpectralProcessing:
    """Test suite for spectral processing functions."""

    def test_smooth_savgol(self, sample_spectrum):
        """Test Savitzky-Golay smoothing."""
        smoothed = smooth_savgol(sample_spectrum, window_length=5, polyorder=2)
        assert len(smoothed) == len(sample_spectrum)

    def test_smooth_moving_average(self, sample_spectrum):
        """Test moving average smoothing."""
        smoothed = smooth_moving_average(sample_spectrum, window_size=3)
        assert len(smoothed) == len(sample_spectrum) - 2

    def test_smooth_butter_lowpass(self, sample_spectrum):
        """Test Butterworth low-pass filter."""
        filtered = smooth_butter_lowpass(sample_spectrum, cutoff=0.1, fs=1, order=3)
        assert len(filtered) == len(sample_spectrum)

    def test_calculate_modified_z_score(self, sample_spectrum):
        """Test calculation of the modified Z-score."""
        modified_z = calculate_modified_z_score(sample_spectrum)
        assert modified_z.shape[0] == sample_spectrum.shape[0]

    def test_get_ratio_two_specs(self, sample_spectrum, sample_wavelengths):
        """Test calculation of the ratio between two specified wavelengths."""
        ratio = get_ratio_two_specs(sample_spectrum, sample_wavelengths, wave_1=450, wave_2=650)
        assert ratio != -1
        assert ratio >= 0

    def test_get_sub_tow_specs(self, sample_spectrum, sample_wavelengths):
        """Test calculation of the difference between two specified wavelengths."""
        diff = get_sub_tow_specs(sample_spectrum, sample_wavelengths, wave_1=450, wave_2=650)
        assert diff != -1

    def test_signal_to_noise(self, sample_spectrum):
        """Test signal-to-noise ratio calculation."""
        snr = signal_to_noise(sample_spectrum)
        assert snr >= 0

    def test_del_leading_zeros(self):
        """Test removal of leading zeros from a spectrum."""
        spectrum = np.array([0, 0, 0, 5, 6, 7])
        result = del_leading_zeros(spectrum, auto_offset=0)
        assert result[0] == 5

    def test_del_last_zeros(self):
        """Test removal of trailing zeros from a spectrum."""
        spectrum = np.array([5, 6, 7, 0, 0, 0])
        result = del_last_zeros(spectrum, auto_offset=0)
        assert result[-1] == 7


# --------------- Edge Case Tests ---------------
class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_smooth_savgol_invalid_input(self):
        """Test Savitzky-Golay smoothing with invalid window length."""
        with pytest.raises(ValueError):
            smooth_savgol(np.array([1, 2, 3]), window_length=4, polyorder=2)

    def test_signal_to_noise_zero_std(self):
        """Test signal-to-noise ratio calculation when standard deviation is zero."""
        spectrum = np.array([1, 1, 1, 1])
        snr = signal_to_noise(spectrum)
        assert snr == 0

