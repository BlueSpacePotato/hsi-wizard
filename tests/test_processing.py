import pytest
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt

import wizard

# Importing the functions to be tested
from wizard._processing.spectral import (
    smooth_savgol,
    smooth_moving_average,
    smooth_butter_lowpass,
    spec_baseline_als,
    calculate_modified_z_score,
    get_ratio_two_specs,
    get_sub_tow_specs,
    signal_to_noise,
    del_leading_zeros,
    del_last_zeros
)

from wizard._processing.cluster import (
    quit_low_change_in_clusters,
    discard_clusters,
    update_clusters,
    initial_clusters,
    sort_arrays_by_first,
    split_clusters,
    compute_avg_distance,
    compute_overall_distance,
    merge_clusters,
    compute_pairwise_distances,
    isodata,
)

# Create sample data for testing
@pytest.fixture
def sample_spectrum():
    return np.array(range(100))

@pytest.fixture
def sample_baseline_spectrum():
    return np.array(range(100))

@pytest.fixture
def sample_wavelengths():
    return np.linspace(400, 800, 10)



def test_smooth_savgol(sample_spectrum):
    smoothed = smooth_savgol(sample_spectrum, window_length=5, polyorder=2)
    assert len(smoothed) == len(sample_spectrum)


def test_smooth_moving_average(sample_spectrum):
    smoothed = smooth_moving_average(sample_spectrum, window_size=3)
    assert len(smoothed) == len(sample_spectrum) - 2, "Length of moving average should be adjusted based on window size."


def test_smooth_butter_lowpass(sample_spectrum):
    filtered = smooth_butter_lowpass(sample_spectrum, cutoff=0.1, fs=1, order=3)
    assert len(filtered) == len(sample_spectrum), "Length of filtered spectrum should match input length."


def test_calculate_modified_z_score(sample_spectrum):
    modified_z = calculate_modified_z_score(sample_spectrum)
    assert modified_z.shape[0] == sample_spectrum.shape[0]


def test_get_ratio_two_specs(sample_spectrum, sample_wavelengths):
    # Generate test data with random values and fixed wavelengths
    ratio = get_ratio_two_specs(sample_spectrum, sample_wavelengths, wave_1=450, wave_2=650)
    assert ratio != -1, "Ratio should be valid for given wavelengths within the range."
    assert ratio >= 0, "Ratio should be non-negative."


def test_get_sub_tow_specs(sample_spectrum, sample_wavelengths):
    # Generate test data with random values and fixed wavelengths
    diff = get_sub_tow_specs(sample_spectrum, sample_wavelengths, wave_1=450, wave_2=650)
    assert diff != -1, "Difference should be valid for given wavelengths within the range."


def test_signal_to_noise(sample_spectrum):
    snr = signal_to_noise(sample_spectrum)
    assert snr >= 0, "Signal-to-noise ratio should be non-negative."


def test_del_leading_zeros():
    spectrum = np.array([0, 0, 0, 5, 6, 7])
    result = del_leading_zeros(spectrum, auto_offset=0)
    assert result[0] == 5, "Leading zeros should be removed."


def test_del_last_zeros():
    spectrum = np.array([5, 6, 7, 0, 0, 0])
    result = del_last_zeros(spectrum, auto_offset=0)
    print(result)
    assert result[-1] == 7, "Trailing zeros should be removed."


# Additional test case for edge cases and boundary conditions
def test_smooth_savgol_invalid_input():
    with pytest.raises(ValueError):
        smooth_savgol(np.array([1, 2, 3]), window_length=4, polyorder=2)  # Window length must be odd


def test_signal_to_noise_zero_std():
    spectrum = np.array([1, 1, 1, 1])
    snr = signal_to_noise(spectrum)
    assert snr == 0, "SNR should be zero when the standard deviation is zero."


class TestIsodata:
    def setup_method(self):
        """Initialize test data."""
        self.centers = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.last_centers = np.array([[1.1, 2.1], [2.9, 3.9], [5.1, 6.1]])
        self.img_flat = np.random.rand(100, 2)
        self.img_class_flat = np.random.randint(0, 3, 100)
        self.clusters_list = np.array([0, 1, 2])
        self.theta_o = 0.05
        self.theta_m = 10

    def test_discard_clusters(self):
        new_centers, new_clusters_list, k_ = discard_clusters(self.img_class_flat, self.centers, self.clusters_list, self.theta_m)
        assert isinstance(new_centers, np.ndarray)
        assert isinstance(new_clusters_list, np.ndarray)
        assert isinstance(k_, int)

    def test_update_clusters(self):
        new_centers, new_clusters_list, k_ = update_clusters(self.img_flat, self.img_class_flat, self.centers, self.clusters_list)
        assert new_centers.shape[1] == self.img_flat.shape[1]

    def test_initial_clusters(self):
        centers = initial_clusters(self.img_flat, 3, method="linspace")
        assert centers.shape == (3, self.img_flat.shape[1])

    def test_sort_arrays_by_first(self):
        sorted_centers, sorted_clusters_list = sort_arrays_by_first(self.centers, self.clusters_list)
        assert sorted_centers[0, 0] <= sorted_centers[1, 0] <= sorted_centers[2, 0]

    def test_split_clusters(self):
        new_centers, new_clusters_list, k_ = split_clusters(self.img_flat, self.img_class_flat, self.centers, self.clusters_list, 0.5, self.theta_m)
        assert isinstance(new_centers, np.ndarray)

    def test_compute_avg_distance(self):
        avg_dists, k_ = compute_avg_distance(self.img_flat, self.img_class_flat, self.centers, self.clusters_list)
        assert avg_dists.shape == (self.centers.shape[0],)

    def test_compute_overall_distance(self):
        avg_dists, _ = compute_avg_distance(self.img_flat, self.img_class_flat, self.centers, self.clusters_list)
        d, k_ = compute_overall_distance(self.img_class_flat, avg_dists, self.clusters_list)
        assert isinstance(d, float)

    def test_merge_clusters(self):
        new_centers, new_clusters_list, k_ = merge_clusters(self.img_class_flat, self.centers, self.clusters_list, 2, 2, 3)
        assert isinstance(new_centers, np.ndarray)

    def test_compute_pairwise_distances(self):
        pair_dists = compute_pairwise_distances(self.centers)
        assert isinstance(pair_dists, list)
        assert len(pair_dists) > 0

    def test_isodata(self):
        dc = wizard.DataCube(cube=np.random.rand(20, 8, 9), wavelengths=np.random.randint(0, 200, size=20))
        result = isodata(dc, k=3, it=10)
        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 9)