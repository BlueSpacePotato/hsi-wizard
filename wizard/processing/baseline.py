from ..core import DataCube
from .spectral import spec_baseline_als

def baseline_als(dc: DataCube, lam: float = 1000000, p: float = 0.01, niter: int = 10) -> DataCube:
    """
    Apply Adaptive Smoothness (ALS) baseline correction.

    Iterates through each pixel (spectrum) in the DataCube and subtracts
    the baseline calculated by the `spec_baseline_als` function.

    Parameters
    ----------
    dc : DataCube
    The input DataCube.
    lam : float, optional
    The smoothness parameter for ALS, defaults to 1000000.
    Larger lambda makes the baseline smoother.
    p : float, optional
    The asymmetry parameter for ALS, defaults to 0.01.
    Value between 0 and 1. Controls how much the baseline is pushed
    towards the data (0 for minimal, 1 for maximal).
    niter : int, optional
    The number of iterations for the ALS algorithm, defaults to 10.

    Returns
    -------
    DataCube
    The DataCube with baseline correction applied.

    Examples
    --------
    >>> import wizard
    >>> dc = wizard.read("example.fsm")
    >>> dc.baseline_als(lam=1e6, p=.001, niter=10)
    """
    for x in range(dc.shape[1]):
        for y in range(dc.shape[2]):
            dc.cube[:, x, y] -= spec_baseline_als(
                spectrum=dc.cube[:, x, y],
                lam=lam,
                p=p,
                niter=niter
            )
    return dc


