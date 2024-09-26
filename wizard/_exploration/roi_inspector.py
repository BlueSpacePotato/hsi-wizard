import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.pyplot import ylabel
from matplotlib.widgets import RectangleSelector, Slider

from wizard import DataCube

def roi(dc: DataCube):
    # Initialize ROI dimensions
    roi_x_start, roi_x_end = 0, dc.cube.shape[2]
    roi_y_start, roi_y_end = 0, dc.cube.shape[1]

    def update(val):
        """Update image and line plot based on the slider value."""
        idx = int(slider.val)
        ax_img.imshow(dc.cube[idx], cmap='gray')
        w = dc.wavelengths[idx]
        ax_img.set_title(f'Wavelengths {w}')
        update_line_plot(idx)
        fig.canvas.draw_idle()

    def update_line_plot(idx):
        """Update line plot based on the selected ROI and frame."""
        roi_data = dc.cube[:, roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        mean = np.mean(roi_data, axis=(1,2))
        line_ax.clear()
        line_ax.set_xlabel("Wavelengths [cm⁻¹]")
        line_ax.set_ylabel("Counts [-]")
        line_ax.plot(dc.wavelengths, mean)  # Example: plot the mean across y-axis
        line_ax.set_ylim(0, mean.max() * 1.05)
        line_ax.set_title(f"Mean of ROI")

    def on_change(eclick, erelease):
        """Handle rectangle selector change to update ROI and plots."""
        nonlocal roi_x_start, roi_x_end, roi_y_start, roi_y_end
        roi_x_start, roi_y_start = int(eclick.xdata), int(eclick.ydata)
        roi_x_end, roi_y_end = int(erelease.xdata), int(erelease.ydata)

        # Update the rectangle patch position and size
        rect.set_x(roi_x_start)
        rect.set_y(roi_y_start)
        rect.set_width(roi_x_end - roi_x_start)
        rect.set_height(roi_y_end - roi_y_start)

        update(int(slider.val))  # Update the image and line plot based on the new ROI
        fig.canvas.draw_idle()

    # Create a new figure with two subplots (one for imshow, one for line plot)
    fig, (ax_img, line_ax) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1]})

    # Display the first frame of the cube
    ax_img.imshow(dc.cube[0], cmap='gray')
    w = dc.wavelengths[0]
    ax_img.set_title(f'Wavelengths {w}')
    
    # Add rectangle patch for ROI selection
    rect = Rectangle((0, 0), 10, 10, linewidth=2, edgecolor='r', facecolor='#ff00ff33')
    ax_img.add_patch(rect)

    # Create a rectangle selector and connect it to the on_change function
    rs = RectangleSelector(ax_img, on_change, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels')

    # Slider axis and slider widget
    ax_slider = plt.axes([0.15, 0.02, 0.7, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Frame', 0, dc.cube.shape[0] - 1, valinit=0, valfmt='%d')

    # Connect the slider update function
    slider.on_changed(update)

    # Initial line plot
    update_line_plot(0)

    # Show the plot
    plt.tight_layout()
    plt.show()
