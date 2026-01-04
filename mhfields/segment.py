"""
Magnetic field calculations for finite current segments.

Copyright (C) 2008-2026 Indrek Mandre <indrek(at)mare.ee>
Licensed under the MIT License.

This module implements the magnetic field of a finite straight wire segment
using the Biot-Savart law.

Based on: http://www.mare.ee/indrek/octave/
"""

import numpy as np


def line_magnetic_field(p0, p1, I, points):
    """
    Compute the magnetic field from a finite current-carrying wire segment.

    Uses the Biot-Savart law for a finite straight wire.

    Parameters
    ----------
    p0 : array_like, shape (3,)
        Start point of the segment (m)
    p1 : array_like, shape (3,)
        End point of the segment (m)
    I : float
        Current (A), positive = current flows from p0 to p1
    points : array_like, shape (N, 3) or (3,)
        Evaluation points (m)

    Returns
    -------
    B : ndarray, shape (N, 3) or (3,)
        Magnetic field vectors (T)

    Notes
    -----
    The formula used is derived from the Biot-Savart law:

        B = (μ₀I/4π) × [(r̂₁ - t̂)/(b + |r₁|) - (r̂₀ - t̂)/(|r₀| - a)] × t̂

    where:
    - t̂ = unit vector along segment from p0 to p1
    - r₀ = point - p0, r₁ = point - p1
    - a = t̂ · r₀, b = -t̂ · r₁
    - r̂₀, r̂₁ = unit vectors in direction of r₀, r₁

    Singularity handling:
    - Points on the line return B = 0 (field direction undefined)
    - Endpoint singularities handled by swapping r0/r1 when needed

    Examples
    --------
    >>> # Field from a wire segment along Z-axis
    >>> B = line_magnetic_field([0, 0, -1], [0, 0, 1], 1.0, [1, 0, 0])
    >>> print(f"B = {B}")  # Should be approximately [0, 1.414e-7, 0]
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    points = np.asarray(points, dtype=float)

    # Handle single point input
    single_point = points.ndim == 1
    if single_point:
        points = points.reshape(1, 3)

    N = points.shape[0]

    # Segment direction
    seg = p1 - p0
    seg_len = np.linalg.norm(seg)

    if seg_len < 1e-30:
        # Zero-length segment produces no field
        return np.zeros(3) if single_point else np.zeros((N, 3))

    t = seg / seg_len  # Unit tangent vector

    # Vectors from endpoints to evaluation points
    r0 = points - p0  # (N, 3)
    r1 = points - p1  # (N, 3)

    # Projections onto line direction
    a = np.sum(r0 * t, axis=1)  # (N,)
    b = -np.sum(r1 * t, axis=1)  # (N,)

    # Distances to endpoints
    r0_len = np.linalg.norm(r0, axis=1)  # (N,)
    r1_len = np.linalg.norm(r1, axis=1)  # (N,)

    # Swap r0/r1 and a/b to avoid singularity when b + r1_len is small
    # This happens when point is near p1 endpoint
    swap_mask = (b + r1_len) < 1e-10 * seg_len
    if np.any(swap_mask):
        r0[swap_mask], r1[swap_mask] = r1[swap_mask].copy(), r0[swap_mask].copy()
        a[swap_mask], b[swap_mask] = b[swap_mask].copy(), a[swap_mask].copy()
        r0_len[swap_mask], r1_len[swap_mask] = r1_len[swap_mask].copy(), r0_len[swap_mask].copy()

    # Normalized direction vectors
    # Handle r0_len = 0 (point at p0)
    r0_len_safe = np.maximum(r0_len, 1e-30)
    r1_len_safe = np.maximum(r1_len, 1e-30)

    r0_hat = r0 / r0_len_safe[:, np.newaxis]  # (N, 3)
    r1_hat = r1 / r1_len_safe[:, np.newaxis]  # (N, 3)

    # Denominators for the two terms
    denom1 = b + r1_len  # (N,)
    denom2 = r0_len - a  # (N,)

    # Avoid division by zero
    denom1_safe = np.where(np.abs(denom1) > 1e-30, denom1, 1.0)
    denom2_safe = np.where(np.abs(denom2) > 1e-30, denom2, 1.0)

    # Terms in brackets: (r1_hat - t)/denom1 - (r0_hat - t)/denom2
    term1 = (r1_hat - t) / denom1_safe[:, np.newaxis]  # (N, 3)
    term2 = (r0_hat - t) / denom2_safe[:, np.newaxis]  # (N, 3)
    bracket = term1 - term2  # (N, 3)

    # Cross product with t gives B direction
    # B = (μ₀I/4π) × bracket × t
    B = I * 1e-7 * np.cross(bracket, t)  # (N, 3)

    # Zero out invalid points (on the line or at endpoints)
    # Points on line: perpendicular distance is zero
    # r_perp = r0 - a*t, perp_dist = |r_perp|
    r_perp = r0 - a[:, np.newaxis] * t
    perp_dist = np.linalg.norm(r_perp, axis=1)

    on_line = perp_dist < 1e-14 * seg_len
    B[on_line] = 0.0

    # Also zero out if denominators were too small
    invalid = (np.abs(denom1) < 1e-30) | (np.abs(denom2) < 1e-30)
    B[invalid] = 0.0

    return B[0] if single_point else B
