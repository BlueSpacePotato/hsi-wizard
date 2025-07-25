import numpy as np
from wizard import DataCube, plotter
from wizard._utils.example import generate_pattern_stack
from matplotlib.pyplot import subplots, show

# Generate datasets
data_a = generate_pattern_stack(20, 600, 400)
data_b = data_a[:, ::2, ::2]

# Create DataCubes
wls_a = np.linspace(0, 20, 20, dtype=int)
wls_b = np.linspace(60, 100, 20, dtype=int)
dc_a = DataCube(data_a, wavelengths=wls_a, name="dc_a")
dc_b = DataCube(data_b, wavelengths=wls_b, name="dc_b")

# Display cube info
print(dc_a, dc_b, sep='\n')

# Helper for side-by-side comparison
def compare_cubes(c1, c2, idx=1):
    fig, axes = subplots(1, 2, figsize=(10, 5))
    for ax, dc in zip(axes, (c1, c2)):
        ax.imshow(dc[idx])
        ax.set_title(dc.name)
    fig.tight_layout()
    show()

# Initial comparison
compare_cubes(dc_a, dc_b)

# Resize and compare again
dc_b.resize(x_new=dc_a.shape[1], y_new=dc_a.shape[2])
print(dc_b)
compare_cubes(dc_a, dc_b)

# Merge and inspect
dc_a.merge_cubes(dc_b)
print(dc_a)
plotter(dc_a)