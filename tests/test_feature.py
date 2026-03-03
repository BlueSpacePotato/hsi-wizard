import pytest
import numpy as np

import wizard

# Importing clustering functions
from wizard.features.cluster import _quit_low_change_in_clusters, _discard_clusters, _update_clusters, _initial_clusters, _sort_arrays_by_first, _split_clusters, _compute_avg_distance, _compute_overall_distance, _merge_clusters, _compute_pairwise_distances, isodata



class TestClustering:
    """Test suite for clustering-related functions."""

    def setup_method(self):
        """Initialize test data for clustering tests."""
        self.centers = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.last_centers = np.array([[1.1, 2.1], [2.9, 3.9], [5.1, 6.1]])
        self.img_flat = np.random.rand(100, 2)
        self.img_class_flat = np.random.randint(0, 3, 100)
        self.clusters_list = np.array([0, 1, 2])
        self.theta_o = 0.05
        self.theta_m = 10

    def test_discard_clusters(self):
        """Test discarding clusters based on a threshold."""
        img_class_flat = np.array([0, 1, 1, 2, 2, 2])
        centers = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        clusters_list = np.array([0, 1, 2])
        theta_m = 2

        new_centers, new_clusters_list, k_ = _discard_clusters(img_class_flat, centers, clusters_list, theta_m)
        assert new_centers.shape == (1, 2)
        assert new_clusters_list.size == 1
        assert k_ == 3

    def test_update_clusters(self):
        """Test updating cluster centers."""
        new_centers, new_clusters_list, k_ = _update_clusters(
            self.img_flat, self.img_class_flat, self.centers, self.clusters_list
        )
        assert new_centers.shape[1] == self.img_flat.shape[1]

    def test_initial_clusters(self):
        """Test initialization of cluster centers."""
        img_flat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        k_ = 2

        centers = _initial_clusters(img_flat, k_, method="linspace")
        assert centers.shape == (2, 2)

        centers = _initial_clusters(img_flat, k_, method="random")
        assert centers.shape == (2, 2)

    def test_sort_arrays_by_first(self):
        """Test sorting of arrays by the first column."""
        sorted_centers, sorted_clusters_list = _sort_arrays_by_first(self.centers, self.clusters_list)
        assert sorted_centers[0, 0] <= sorted_centers[1, 0] <= sorted_centers[2, 0]

    def test_split_clusters(self):
        """Test splitting of clusters based on thresholds."""
        img_flat = np.random.rand(10, 2)
        img_class_flat = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        centers = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        clusters_list = np.array([0, 1, 2])
        theta_s = 1.0
        theta_m = 2

        new_centers, new_clusters_list, k_ = _split_clusters(
            img_flat, img_class_flat, centers, clusters_list, theta_s, theta_m
        )
        assert new_centers.shape[0] >= centers.shape[0]
        assert new_clusters_list.size >= clusters_list.size

    def test_compute_avg_distance(self):
        """Test computation of average distance within clusters."""
        avg_dists, k_ = _compute_avg_distance(self.img_flat, self.img_class_flat, self.centers, self.clusters_list)
        assert avg_dists.shape == (self.centers.shape[0],)

    def test_compute_overall_distance(self):
        """Test computation of overall distance within clusters."""
        avg_dists, _ = _compute_avg_distance(self.img_flat, self.img_class_flat, self.centers, self.clusters_list)
        d, k_ = _compute_overall_distance(self.img_class_flat, avg_dists, self.clusters_list)
        assert isinstance(d, float)

    def test_merge_clusters(self):
        """Test merging of clusters."""
        new_centers, new_clusters_list, k_ = _merge_clusters(
            self.img_class_flat, self.centers, self.clusters_list, 2, 2, 3
        )
        assert isinstance(new_centers, np.ndarray)

    def test_compute_pairwise_distances(self):
        """Test computation of pairwise distances between cluster centers."""
        pair_dists = _compute_pairwise_distances(self.centers)
        assert isinstance(pair_dists, list)
        assert len(pair_dists) > 0

    def test_isodata(self):
        """Test ISODATA clustering."""
        dc = wizard.DataCube(cube=np.random.rand(20, 8, 9), wavelengths=np.random.randint(0, 200, size=20))
        result = isodata(dc, k=3, it=10)
        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 9)

    def test_quit_low_change_in_clusters(self):
        """Test termination of clustering based on low change in cluster centers."""
        centers = np.array([[1.0, 2.0], [3.0, 4.0]])
        last_centers = np.array([[1.01, 2.01], [3.01, 4.01]])
        theta_o = 0.02
        assert _quit_low_change_in_clusters(centers, last_centers, theta_o) is True

        theta_o = 0.001
        assert _quit_low_change_in_clusters(centers, last_centers, theta_o) is False

    def test_update_clusters_mismatch_raises(self):
        centers = np.array([[1.0, 2.0]])
        clusters_list = np.array([0, 1])  # mismatch
        with pytest.raises(ValueError):
            _update_clusters(self.img_flat, self.img_class_flat, centers, clusters_list)

    def test_initial_clusters_invalid_method(self):
        img_flat = np.array([[0.0, 1.0], [1.0, 2.0]])
        with pytest.raises(AssertionError):
            _initial_clusters(img_flat, 2, method="not_a_method")

    def test_generate_gaussian_kernel_properties(self):
        size, sigma = 5, 1.0
        kernel = wizard.features.cluster._generate_gaussian_kernel(size=size, sigma=sigma)
        assert kernel.shape == (size, size)
        assert pytest.approx(kernel.sum(), rel=1e-6) == 1.0

    def test_optimal_clusters_threshold_behavior(self):
        # simple two-point data
        pixels = np.array([[0.0], [10.0], [20.0], [30.0]])
        # threshold 0 => always >0, so best_k == max_clusters
        assert wizard.features.cluster._optimal_clusters(pixels, max_clusters=3, threshold=0.0) == 3
        # very large threshold => break on first iteration, best_k stays 2
        assert wizard.features.cluster._optimal_clusters(pixels, max_clusters=5, threshold=1e6) == 2

    def test_smooth_kmeans_constant_cube(self):
        cube = np.zeros((2, 3, 3))
        dc = wizard.DataCube(cube=cube, wavelengths=np.array([0, 1]))
        labels = wizard.features.cluster.smooth_kmeans(dc, n_clusters=2, mrf_iterations=2, kernel_size=3, sigma=0.5)
        assert labels.shape == (3, 3)
        assert np.all(labels == 0)

    def test_pca_reduction_and_errors(self):
        # valid reduction
        cube = np.arange(12).reshape(3, 2, 2).astype(float)
        dc = wizard.DataCube(cube=cube, wavelengths=np.arange(3))
        dc2 = wizard.features.cluster.pca(dc, n_components=2)
        assert dc2.cube.shape == (2, 2, 2)
        assert np.array_equal(dc2.wavelengths, np.array([0, 1]))
        # too many components
        dc3 = wizard.DataCube(cube=np.zeros((3, 2, 2)), wavelengths=np.arange(3))
        with pytest.raises(ValueError):
            wizard.features.cluster.pca(dc3, n_components=5)

    def test_spectral_spatial_kmeans(self):
        cube = np.array([[[0, 0], [100, 100]]], dtype=float)  # v=1, x=2, y=2
        dc = wizard.DataCube(cube=cube, wavelengths=np.array([0]))
        labels = wizard.features.cluster.spectral_spatial_kmeans(dc, n_clusters=2, spatial_radius=0)
        assert labels.shape == (2, 2)
        assert set(np.unique(labels)) == {0, 1}
        # invalid parameters
        with pytest.raises(ValueError):
            wizard.features.cluster.spectral_spatial_kmeans(dc, n_clusters=0, spatial_radius=1)
        with pytest.raises(ValueError):
            wizard.features.cluster.spectral_spatial_kmeans(dc, n_clusters=1, spatial_radius=-1)

    def test_spatial_agglomerative_clustering(self):
        cube = np.array([[[0, 1], [1, 0]]], dtype=float)
        dc = wizard.DataCube(cube=cube, wavelengths=np.array([0]))
        labels = wizard.features.cluster.spatial_agglomerative_clustering(dc, n_clusters=2)
        assert labels.shape == (2, 2)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_smooth_cluster_errors_and_identity(self):
        # bad type
        with pytest.raises(TypeError):
            wizard.features.cluster.smooth_cluster([1, 2, 3], sigma=1.0)
        # empty array
        with pytest.raises(ValueError):
            wizard.features.cluster.smooth_cluster(np.array([]), sigma=1.0)
        # identity on constant image
        img = np.ones((5, 5), dtype=int)
        result = wizard.features.cluster.smooth_cluster(img, sigma=2.0)
        assert result.dtype == img.dtype
        assert np.array_equal(result, img)
