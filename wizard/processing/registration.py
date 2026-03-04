"""
wizard.processing.registration
==============================

.. module:: wizard.processing.registration
:platform: Unix
:synopsis: Spatial registration utilities for DataCube layers.

Module Overview
---------------

This module provides functions for spatial alignment (registration) of
spectral layers within a :class:`~wizard.core.DataCube`.

The implemented methods use feature-based homography estimation to align
each layer to a reference layer. A simple registration method and a more
robust fallback-based approach are provided.

All functions operate **in-place** on the given :class:`DataCube` and return
the modified object to allow method chaining.

Functions
---------
.. autofunction:: register_layers_simple
.. autofunction:: register_layers_best
"""


import cv2

from .._utils.helper import feature_registration, RegistrationError, decompose_homography, normalize_polarity, \
    auto_canny
from ..core import DataCube
import copy


def register_layers_simple(dc: DataCube, max_features: int = 5000, match_percent: float = 0.1) -> DataCube:
    """
    Align images within a DataCube using simple feature-based registration.

    Each layer in the DataCube is aligned to the first layer (index 0)
    using ORB feature detection and homography estimation via `_feature_registration`.

    Parameters
    ----------
    dc : DataCube
        The DataCube whose layers are to be registered.
    max_features : int, optional
        Maximum number of keypoint regions to detect, defaults to 5000.
    match_percent : float, optional
        Percentage of keypoint matches to consider for homography,
        defaults to 0.1 (10%).

    Returns
    -------
    DataCube
    The DataCube with layers registered.

    Examples
    --------
    >>> import wizard
    >>> dc = wizard.read('example.fsm')
    >>> dc.register_layers_simple()
    """
    o_img = dc.cube[0, :, :]
    for i in range(dc.cube.shape[0]):
        if i > 0:
            a_img = copy.deepcopy(dc.cube[i, :, :])
            try:
                _, h = feature_registration(
                    o_img=o_img, a_img=a_img,
                    max_features=max_features, match_percent=match_percent
                )
                height, width = o_img.shape
                aligned_img = cv2.warpPerspective(a_img, h, (width, height))
                dc.cube[i, :, :] = aligned_img
            except RegistrationError as e:
                print(f"Warning: Could not register layer {i} in simple registration: {e}")
                pass
    dc.registered = True
    return dc


def register_layers_best(dc: DataCube, ref_layer: int = 0, max_features: int = 5000, match_percent: float = 0.1, rot_thresh: float = 20.0, scale_thresh: float = 1.1) -> DataCube:
    """
    Align DataCube layers with robust registration.

    Aligns each slice of `dc.cube` to a reference layer. It uses
    feature-based registration primarily. If feature-based registration
    yields a degenerate homography (based on rotation and scale thresholds)
    or fails, it falls back to Canny-based edge registration.
    Failed alignments are retried once.

    Parameters
    ----------
    dc : DataCube
        The DataCube to process.
    ref_layer : int, optional
        Index of the reference layer, defaults to 0.
    max_features : int, optional
        Maximum features for ORB, defaults to 5000.
    match_percent : float, optional
        Match percentage for ORB, defaults to 0.1.
    rot_thresh : float, optional
        Rotation threshold (degrees) for homography validation,
        defaults to 20.0.
    scale_thresh : float, optional
        Scale threshold for homography validation, defaults to 1.1.
        Checks if max_scale <= scale_thresh and min_scale >= 1/scale_thresh.

    Returns
    -------
    DataCube
    The DataCube with aligned layers.

    Raises
    ------
    RuntimeError
    If alignment fails for a layer after retry.

    Examples
    --------
    >>> import wizard
    >>> dc = wizard.read('example.fsm')
    >>> dc.register_layers_best()
    """
    aligned_indices = {ref_layer}
    waitlist = set()
    n_layers, H_dim, W_dim = dc.cube.shape

    def try_align(layer_idx: int, current_aligned_indices: set) -> bool:
        # nonlocal dc
        a_img = dc.cube[layer_idx]
        best_alignment_img = None

        for ref_idx in current_aligned_indices:
            try:
                o_img = dc.cube[ref_idx]
                aligned_img_feat, H_ij = feature_registration(
                    o_img, a_img, max_features, match_percent
                )
                angle, S = decompose_homography(H_ij)
                s_ok = (S.max() <= scale_thresh and S.min() >= 1 / scale_thresh)
                if abs(angle) <= rot_thresh and s_ok:
                    print(f"[Layer {layer_idx}] aligned to {ref_idx}: θ={angle:.1f}°, S={S.round(3)}")
                    best_alignment_img = aligned_img_feat
                    break
                else:
                    print(f"[Layer {layer_idx}] reject vs {ref_idx}: θ={angle:.1f}°, S={S.round(3)}")
            except RegistrationError as e:
                print(f"[Layer {layer_idx}] registration to {ref_idx} failed: {e}")

        if best_alignment_img is None:
            print(f"[Layer {layer_idx}] edge-map fallback to reference layer {ref_layer}")
            try:
                ref_img_for_edge = normalize_polarity(dc.cube[ref_layer])
                tgt_img_for_edge = normalize_polarity(a_img)
                edges_ref = auto_canny(ref_img_for_edge)
                edges_tgt = auto_canny(tgt_img_for_edge)
                aligned_img_edge, H_e = feature_registration(
                    edges_ref.astype(float), edges_tgt.astype(float),
                    max_features, match_percent
                )
                h_orig, w_orig = a_img.shape
                best_alignment_img = cv2.warpPerspective(a_img, H_e, (w_orig, h_orig), flags=cv2.INTER_LINEAR)
                angle_e, S_e = decompose_homography(H_e)
                print(f"[Layer {layer_idx}] edges (vs layer {ref_layer}): θ={angle_e:.1f}°, S={S_e.round(3)}")
            except RegistrationError as e:
                print(f"[Layer {layer_idx}] edge registration failed: {e}")
            except Exception as e:
                print(f"[Layer {layer_idx}] unexpected error in edge registration: {e}")

        if best_alignment_img is not None:
            dc.cube[layer_idx] = best_alignment_img
            return True
        return False

    for i in range(n_layers):
        if i == ref_layer:
            continue
        if try_align(i, aligned_indices.copy()):
            aligned_indices.add(i)
        else:
            waitlist.add(i)

    if waitlist:
        print(f"\nRetrying layers: {list(waitlist)}\n")
        for i in list(waitlist):
            if try_align(i, aligned_indices.copy()):
                aligned_indices.add(i)
                waitlist.remove(i)
            else:
                raise RuntimeError(f"Layer {i}: alignment failed after retry.")
    dc.registered = True
    return dc
