import random

import numpy as np
import matplotlib

import wizard
from wizard._exploration import plotter

import pytest

matplotlib.use("Agg")

class TestPlotter:

    def test_plot_function_runs(self):
        dc = wizard.DataCube(cube=np.random.rand(20, 8, 9), wavelengths=np.random.randint(0, 200, size=20))
        try:
            wizard.plotter(dc)
            assert True
        except:
            assert False

    def test_normalize_small_range(self):
        spec = np.array([0.0001, 0.0002, 0.0003])
        normalized = plotter.normalize_spec(spec)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(normalized, expected, atol=1e-6)

class TestSurcefacePlot:

    def test_surface_function_runs(self):
        dc = wizard.DataCube(cube=np.random.rand(20, 8, 9))
        wizard.plot_surface(dc)

    def test_dc_cut_by_value(self):
        # Create a random DataCube
        dc = wizard.DataCube(cube=np.random.rand(20, 8, 9))

        # Extract a random slice
        z = dc.cube[5, :, :]
        val = 0.5  # Example cut value

        # Apply function
        modified_z = wizard._exploration.surface.dc_cut_by_value(z, val, type="")

        # Check modifications
        assert modified_z.shape == z.shape, "Shape should remain unchanged"
        assert np.all(modified_z[modified_z <= val] == modified_z.min()), "Values below threshold should be set to min"

    def test_get_z_surface(self):
        # Create a random DataCube
        dc = wizard.DataCube(cube=np.random.rand(20, 8, 9))

        # Select a slice index
        v = 5

        # Compute surface
        z_surface = wizard._exploration.surface.get_z_surface(dc.cube, v)

        # Check shape
        assert z_surface.shape == (dc.shape[1], dc.shape[2]), "Surface shape mismatch"

    def test_plot_surface(self):
        # Create a random DataCube
        dc = wizard.DataCube(cube=np.random.rand(20, 8, 9))

        # Run plot function (not testing output, only checking for errors)
        try:
            wizard._exploration.surface.plot_surface(dc, index=0)
        except Exception as e:
            pytest.fail(f"plot_surface raised an exception: {e}")

