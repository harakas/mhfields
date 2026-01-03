"""
Electromagnetic field calculations for rings of charge and current.

Copyright (C) 2007-2026 Indrek Mandre <indrek(at)mare.ee>
Licensed under the MIT License.

This module implements the off-axis electric field of a ring of charge
and the magnetic field of a ring of current using complete elliptic integrals.

The formulae are derived in: "Off-axis electric field of a ring of charge"
by Indrek Mandre, July 2007.

Coordinate system:
- Ring lies in the XY plane, centered at origin
- Ring normal points along +Z axis
- r = radial distance from the Z axis
- a = axial distance along Z axis from the ring plane
"""

import numpy as np
from scipy.special import ellipk, ellipe

# Physical constants
MU_0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)
EPSILON_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
C = 299792458  # Speed of light (m/s)

# Numerical cutoff for singularity handling
CUTOFF = 5e-7


def _compute_elliptic_params(r, a, R):
    """
    Compute the elliptic integral parameter and related quantities.

    Parameters
    ----------
    r : array_like
        Radial distance from axis
    a : array_like
        Axial distance from ring plane
    R : float
        Ring radius

    Returns
    -------
    q : ndarray
        q = r² + R² + a² + 2rR
    mu : ndarray
        Elliptic parameter μ = 4rR/q
    sqrt_q : ndarray
        √q
    """
    r = np.asarray(r)
    a = np.asarray(a)

    q = r**2 + R**2 + a**2 + 2*r*R
    mu = np.where(q > 0, 4*r*R / q, 0.0)
    sqrt_q = np.sqrt(q)

    return q, mu, sqrt_q


def _cutoff_radius(a, R, cutoff=CUTOFF):
    """
    Find the cutoff radius where μ = cutoff.

    Solves: cutoff × r² + 2R(cutoff-2) × r + cutoff(a²+R²) = 0

    This is derived from μ = 4rR/q = cutoff, rearranged as a quadratic in r.

    Parameters
    ----------
    a : array_like
        Axial distance
    R : float
        Ring radius
    cutoff : float
        Cutoff value for μ

    Returns
    -------
    r_c : ndarray
        Cutoff radius
    """
    a = np.asarray(a)

    # Quadratic coefficients: A*r² + B*r + C = 0
    A = cutoff
    B = 2 * R * (cutoff - 2)
    C = cutoff * (a**2 + R**2)

    discriminant = B**2 - 4*A*C
    # Use the negative root (smaller r)
    r_c = (-B - np.sqrt(np.maximum(discriminant, 0))) / (2*A)

    return np.maximum(r_c, 1e-32)


def ring_magnetic_field(r, a, R, I):
    """
    Compute the off-axis magnetic field of a ring of current.

    Parameters
    ----------
    r : array_like
        Radial distance from the ring axis (must be >= 0)
    a : array_like
        Axial distance from the ring plane (can be negative)
    R : float
        Ring radius (m)
    I : float
        Current in the ring (A)

    Returns
    -------
    B_r : ndarray
        Radial component of magnetic field (T)
    B_a : ndarray
        Axial component of magnetic field (T)

    Notes
    -----
    The magnetic field components are:

    B_r = (μ₀/4π) × I × (2/√q) × (a/r) × [E(√μ) × (q-2rR)/(q-4rR) - K(√μ)]
    B_a = (μ₀/4π) × I × (2/√q) × [E(√μ) × (R²-r²-a²)/(q-4rR) + K(√μ)]

    where q = r² + R² + a² + 2rR and μ = 4rR/q

    Singularity handling:
    - On axis (r→0): B_r = 0 by symmetry
    - For small μ: Linear interpolation from cutoff radius
    """
    r = np.asarray(r, dtype=float)
    a = np.asarray(a, dtype=float)

    # Ensure arrays are broadcastable
    r, a = np.broadcast_arrays(r, a)
    original_shape = r.shape
    r = r.ravel()
    a = a.ravel()

    # Initialize output
    B_r = np.zeros_like(r)
    B_a = np.zeros_like(r)

    # Compute elliptic parameters
    q, mu, sqrt_q = _compute_elliptic_params(r, a, R)

    # Prefactor: (μ₀/4π) × I × 2 = μ₀I/(2π)
    # But Octave uses f1 = 2e-7 * I which is μ₀I/(4π) × 2
    prefactor = 2e-7 * I  # = μ₀I/(2π)

    # Mask for valid (non-singular) calculations
    # Exclude points on the ring itself where denom = (r-R)² + a² ≈ 0
    denom_full = (r - R)**2 + a**2
    valid = (mu >= CUTOFF) & (r > 1e-32) & (denom_full > 1e-20)

    if np.any(valid):
        mu_v = mu[valid]
        r_v = r[valid]
        a_v = a[valid]
        q_v = q[valid]
        sqrt_q_v = sqrt_q[valid]

        # Elliptic integrals (scipy uses parameter m, not modulus k)
        # K(k) and E(k) where k = √μ, so we pass m = μ
        K = ellipk(mu_v)
        E = ellipe(mu_v)

        # Common denominator
        denom = q_v - 4*r_v*R  # = (r-R)² + a²

        # Radial component: B_r = prefactor × (a/r) × [E×(q-2rR)/denom - K] / √q
        B_r[valid] = prefactor * a_v * (E * (q_v - 2*r_v*R) / denom - K) / (sqrt_q_v * r_v)

        # Axial component: B_a = prefactor × [E×(R²-r²-a²)/denom + K] / √q
        B_a[valid] = prefactor * (E * (R**2 - r_v**2 - a_v**2) / denom + K) / sqrt_q_v

    # Handle small μ region (near axis) with cutoff interpolation
    small_mu = (mu < CUTOFF) & (mu > 0) & (r > 1e-32)

    if np.any(small_mu):
        a_sm = a[small_mu]
        r_sm = r[small_mu]

        # Find cutoff radius
        r_c = _cutoff_radius(a_sm, R, CUTOFF)

        # Compute field at cutoff radius
        q_c = r_c**2 + R**2 + a_sm**2 + 2*r_c*R
        mu_c = 4*r_c*R / q_c
        sqrt_q_c = np.sqrt(q_c)

        K_c = ellipk(mu_c)
        E_c = ellipe(mu_c)
        denom_c = q_c - 4*r_c*R

        B_r_c = prefactor * a_sm * (E_c * (q_c - 2*r_c*R) / denom_c - K_c) / (sqrt_q_c * r_c)

        # Linear interpolation: B_r(r) = (r/r_c) × B_r(r_c)
        B_r[small_mu] = (r_sm / r_c) * B_r_c

        # Axial component is well-behaved, compute directly
        q_sm, mu_sm_actual, sqrt_q_sm = _compute_elliptic_params(r_sm, a_sm, R)
        # For very small mu, use limiting value
        mu_sm_safe = np.maximum(mu_sm_actual, 1e-15)
        K_sm = ellipk(mu_sm_safe)
        E_sm = ellipe(mu_sm_safe)
        denom_sm = q_sm - 4*r_sm*R
        B_a[small_mu] = prefactor * (E_sm * (R**2 - r_sm**2 - a_sm**2) / denom_sm + K_sm) / sqrt_q_sm

    # On-axis (r = 0): Use analytical formula
    on_axis = r <= 1e-32

    if np.any(on_axis):
        a_ax = a[on_axis]
        # B_a = (μ₀/2) × I × R² / (a² + R²)^(3/2)
        B_a[on_axis] = (MU_0 / 2) * I * R**2 / (a_ax**2 + R**2)**1.5
        # B_r = 0 by symmetry
        B_r[on_axis] = 0.0

    return B_r.reshape(original_shape), B_a.reshape(original_shape)


def ring_electric_field(r, a, R, Q):
    """
    Compute the off-axis electric field of a ring of charge.

    Parameters
    ----------
    r : array_like
        Radial distance from the ring axis (must be >= 0)
    a : array_like
        Axial distance from the ring plane (can be negative)
    R : float
        Ring radius (m)
    Q : float
        Total charge on the ring (C)

    Returns
    -------
    E_r : ndarray
        Radial component of electric field (V/m)
    E_a : ndarray
        Axial component of electric field (V/m)

    Notes
    -----
    The electric field components are:

    E_r = (Q/4πε₀) × (2/πq^(3/2)(1-μ)) × (1/μ) × [2RK(√μ)(1-μ) - E(√μ)(2R - μ(r+R))]
    E_a = (Q/4πε₀) × (2/πq^(3/2)(1-μ)) × a×E(√μ)

    where q = r² + R² + a² + 2rR and μ = 4rR/q
    """
    r = np.asarray(r, dtype=float)
    a = np.asarray(a, dtype=float)

    # Ensure arrays are broadcastable
    r, a = np.broadcast_arrays(r, a)
    original_shape = r.shape
    r = r.ravel()
    a = a.ravel()

    # Initialize output
    E_r = np.zeros_like(r)
    E_a = np.zeros_like(r)

    # Compute elliptic parameters
    q, mu, sqrt_q = _compute_elliptic_params(r, a, R)

    # Prefactor: Q/(4πε₀) × 2/π = Q × c² × μ₀ × 2/π (using c² = 1/(μ₀ε₀))
    # Simplified: Q/(4πε₀) = Q × c² × μ₀ / (4π)
    k_e = 1 / (4 * np.pi * EPSILON_0)  # Coulomb constant
    prefactor = k_e * Q * 2 / np.pi

    # Mask for valid calculations
    # Exclude points on the ring itself where (1-μ) → 0
    one_minus_mu = 1 - mu
    valid = (mu >= CUTOFF) & (r > 1e-32) & (one_minus_mu > 1e-10)

    if np.any(valid):
        mu_v = mu[valid]
        r_v = r[valid]
        a_v = a[valid]
        q_v = q[valid]
        sqrt_q_v = sqrt_q[valid]

        K = ellipk(mu_v)
        E = ellipe(mu_v)

        one_minus_mu = 1 - mu_v
        q32 = q_v * sqrt_q_v  # q^(3/2)

        # Common factor for both components
        common = prefactor / (q32 * one_minus_mu)

        # Radial component
        # E_r = common × (1/μ) × [2RK(1-μ) - E(2R - μ(r+R))]
        E_r[valid] = common * (1/mu_v) * (2*R*K*one_minus_mu - E*(2*R - mu_v*(r_v + R)))

        # Axial component
        # E_a = common × a × E
        E_a[valid] = common * a_v * E

    # Handle small μ region with cutoff interpolation
    small_mu = (mu < CUTOFF) & (mu > 0) & (r > 1e-32)

    if np.any(small_mu):
        a_sm = a[small_mu]
        r_sm = r[small_mu]

        # Find cutoff radius
        r_c = _cutoff_radius(a_sm, R, CUTOFF)

        # Compute field at cutoff radius
        q_c = r_c**2 + R**2 + a_sm**2 + 2*r_c*R
        mu_c = 4*r_c*R / q_c
        sqrt_q_c = np.sqrt(q_c)

        K_c = ellipk(mu_c)
        E_c = ellipe(mu_c)

        one_minus_mu_c = 1 - mu_c
        q32_c = q_c * sqrt_q_c

        common_c = prefactor / (q32_c * one_minus_mu_c)
        E_r_c = common_c * (1/mu_c) * (2*R*K_c*one_minus_mu_c - E_c*(2*R - mu_c*(r_c + R)))

        # Linear interpolation
        E_r[small_mu] = (r_sm / r_c) * E_r_c

        # Axial component - compute at actual position
        q_sm = r_sm**2 + R**2 + a_sm**2 + 2*r_sm*R
        mu_sm = np.maximum(4*r_sm*R / q_sm, 1e-15)
        sqrt_q_sm = np.sqrt(q_sm)
        E_sm = ellipe(mu_sm)
        one_minus_mu_sm = 1 - mu_sm
        q32_sm = q_sm * sqrt_q_sm
        common_sm = prefactor / (q32_sm * one_minus_mu_sm)
        E_a[small_mu] = common_sm * a_sm * E_sm

    # On-axis (r = 0): Use analytical formula
    on_axis = r <= 1e-32

    if np.any(on_axis):
        a_ax = a[on_axis]
        # E_a = Q × a / (4πε₀ × (a² + R²)^(3/2))
        E_a[on_axis] = k_e * Q * a_ax / (a_ax**2 + R**2)**1.5
        # E_r = 0 by symmetry
        E_r[on_axis] = 0.0

    return E_r.reshape(original_shape), E_a.reshape(original_shape)


def ring_magnetic_field_xyz(x, y, z, R, I):
    """
    Compute the magnetic field of a ring of current in Cartesian coordinates.

    The ring lies in the XY plane, centered at the origin, with current
    flowing counterclockwise when viewed from +Z.

    Parameters
    ----------
    x, y, z : array_like
        Cartesian coordinates (m)
    R : float
        Ring radius (m)
    I : float
        Current in the ring (A)

    Returns
    -------
    Bx, By, Bz : ndarray
        Cartesian components of magnetic field (T)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    # Convert to cylindrical coordinates
    r = np.sqrt(x**2 + y**2)
    a = z

    # Compute field in cylindrical coordinates
    B_r, B_a = ring_magnetic_field(r, a, R, I)

    # Convert radial component to Cartesian
    # B_r points radially outward from axis
    # Handle r = 0 case
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_phi = np.where(r > 1e-32, x / r, 0.0)
        sin_phi = np.where(r > 1e-32, y / r, 0.0)

    Bx = B_r * cos_phi
    By = B_r * sin_phi
    Bz = B_a

    return Bx, By, Bz


def ring_electric_field_xyz(x, y, z, R, Q):
    """
    Compute the electric field of a ring of charge in Cartesian coordinates.

    The ring lies in the XY plane, centered at the origin.

    Parameters
    ----------
    x, y, z : array_like
        Cartesian coordinates (m)
    R : float
        Ring radius (m)
    Q : float
        Total charge on the ring (C)

    Returns
    -------
    Ex, Ey, Ez : ndarray
        Cartesian components of electric field (V/m)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    # Convert to cylindrical coordinates
    r = np.sqrt(x**2 + y**2)
    a = z

    # Compute field in cylindrical coordinates
    E_r, E_a = ring_electric_field(r, a, R, Q)

    # Convert radial component to Cartesian
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_phi = np.where(r > 1e-32, x / r, 0.0)
        sin_phi = np.where(r > 1e-32, y / r, 0.0)

    Ex = E_r * cos_phi
    Ey = E_r * sin_phi
    Ez = E_a

    return Ex, Ey, Ez
