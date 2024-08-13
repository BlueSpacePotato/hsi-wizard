import numpy as np
# from wizard import DataCube
from joblib import Parallel, delayed


def calculate_modified_z_score(cube):
    """
    Calculate the modified z-score of a data cube by computing the difference in intensity
    along the first axis.
    
    Parameters
    ----------
    cube : numpy.ndarray
        The input data cube of shape (Z, Y, X), where Z is the number of slices,
        Y is the height, and X is the width.
    
    Returns
    -------
    numpy.ndarray
        The modified z-score, which is the difference in intensity along the first axis.
    
    Example
    --------
    >>> cube = np.random.rand(10, 20, 30)  # Example data cube of shape (10, 20, 30)
    >>> modified_z_score = calculate_modified_z_score(cube)
    >>> print(modified_z_score.shape)  # Output: (20, 30)
    """
    delta_intensity = np.diff(cube, axis=0)
    return delta_intensity


def process_slice(spec_out_flat, spikes_flat, idx, window):
    """
    Process a single slice of the data cube to remove spikes by replacing them with the mean
    of the neighboring values within a given window.
    
    Parameters
    ----------
    spec_out_flat : numpy.ndarray
        Flattened output spectrum.
    spikes_flat : numpy.ndarray
        Flattened array indicating the presence of spikes.
    idx : int
        Index of the current slice to process.
    window : int
        Size of the window used to calculate the mean of neighboring values.
    
    Returns
    -------
    tuple
        A tuple containing the index of the processed slice and the modified slice.
    """
    
    w_h = int(window / 2)
    spike = spikes_flat[idx]
    tmp = np.copy(spec_out_flat[idx])

    for spk_idx in np.where(spike)[0]:
        window_min = max(0, spk_idx - w_h)
        window_max = min(len(tmp), spk_idx + w_h + 1)

        if window_min == spk_idx:
            window_data = tmp[spk_idx + 1:window_max]
        elif window_max == spk_idx + 1:
            window_data = tmp[window_min:spk_idx]
        else:
            window_data = np.concatenate((tmp[window_min:spk_idx], tmp[spk_idx + 1:window_max]))

        if len(window_data) > 0:
            tmp[spk_idx] = np.mean(window_data)
        else:
            tmp[spk_idx] = tmp[spk_idx]

    return idx, tmp


# @track_execution_time
def remove_spikes(dc, threshold: int = 6500, window: int = 3):
    """
    Remove cosmic spikes from a data cube using a specified threshold and window size.

    Parameters
    ----------
    dc : DataCube
        The input data cube from which spikes are to be removed.
    threshold : int, optional
        The threshold value for detecting spikes based on the modified z-score.
    window : int, optional
        The size of the window used to calculate the mean of neighboring values
        when replacing spikes.
    
    Returns
    -------
    numpy.ndarray
        The data cube with spikes removed.
    """
        
    z_spectrum = calculate_modified_z_score(dc.cube)
    spikes = abs(z_spectrum) > threshold
    cube_out = dc.cube.copy()

    spikes_flat = spikes.reshape(dc.cube.shape[0] - 1, -1)
    spec_out_flat = cube_out.reshape(cube_out.shape[0], -1)

    results = Parallel(n_jobs=-1)(
        delayed(process_slice)(spec_out_flat, spikes_flat, idx, window) for idx in range(spikes_flat.shape[0]))

    for idx, tmp in results:
        spec_out_flat[idx] = tmp

    spec_out_flat.reshape(cube_out.shape)
    return cube_out
