"""
Numerical validation tests for ring field calculations.

Copyright (C) 2007-2026 Indrek Mandre <indrek(at)mare.ee>
Licensed under the MIT License.

These tests compare the analytical elliptic integral formulas against
numerical approximations using discrete point charges arranged in a circle.
This validates that the analytical derivation is correct.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from mhfields.fields import ring_electric_field, EPSILON_0


# Physical constants
K_E = 1 / (4 * np.pi * EPSILON_0)  # Coulomb constant

# Test parameters
R = 1.0  # Ring radius (m)
Q = 1e-9  # Total charge (C)


def numerical_efield_ring(r, a, R, Q, N=1000):
    """
    Compute electric field numerically using N point charges on a ring.

    Uses extended precision (float128/longdouble) for summation to reduce
    numerical errors.

    Parameters
    ----------
    r : float
        Radial distance from axis
    a : float
        Axial distance from ring plane
    R : float
        Ring radius
    Q : float
        Total charge on ring
    N : int
        Number of point charges to use

    Returns
    -------
    E_r, E_a : float
        Radial and axial components of electric field
    """
    # Use extended precision for accumulation
    dtype = np.longdouble

    # Charge per point
    q = Q / N

    # Angles for point charges (uniformly distributed)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False, dtype=dtype)

    # Positions of point charges on ring (in XY plane)
    # Ring centered at origin, lying in z=0 plane
    x_charges = R * np.cos(angles)
    y_charges = R * np.sin(angles)
    z_charges = np.zeros(N, dtype=dtype)

    # Field point position (in XZ plane due to symmetry, y=0)
    x_field = dtype(r)
    y_field = dtype(0.0)
    z_field = dtype(a)

    # Displacement vectors from each charge to field point
    dx = x_field - x_charges
    dy = y_field - y_charges
    dz = z_field - z_charges

    # Distance cubed
    dist_sq = dx**2 + dy**2 + dz**2
    dist_cubed = dist_sq * np.sqrt(dist_sq)

    # Avoid division by zero (shouldn't happen for valid test points)
    dist_cubed = np.maximum(dist_cubed, 1e-100)

    # Electric field contribution from each charge: E = k*q*(r-r')/|r-r'|^3
    # Sum contributions (using extended precision)
    Ex = np.sum(K_E * q * dx / dist_cubed, dtype=dtype)
    Ey = np.sum(K_E * q * dy / dist_cubed, dtype=dtype)
    Ez = np.sum(K_E * q * dz / dist_cubed, dtype=dtype)

    # Convert to cylindrical coordinates
    # E_r is the radial component (in direction away from z-axis)
    # At our field point (r, 0, a), the radial direction is +x
    E_r = float(Ex)
    E_a = float(Ez)

    # Ey should be zero by symmetry
    assert abs(float(Ey)) < 1e-10 * (abs(E_r) + abs(E_a) + 1e-20), \
        f"Ey should be zero by symmetry, got {Ey}"

    return E_r, E_a


class TestNumericalComparison:
    """Compare analytical results with numerical point charge summation."""

    # Test points: (r, a, description)
    TEST_POINTS = [
        # On axis (but not at origin where both are zero)
        (0.0, 0.5, "on_axis_near"),
        (0.0, 1.0, "on_axis_at_R"),
        (0.0, 2.0, "on_axis_2R"),
        (0.0, 5.0, "on_axis_far"),
        (0.0, -1.0, "on_axis_negative"),

        # In ring plane
        (0.5, 0.0, "ring_plane_inside"),
        (2.0, 0.0, "ring_plane_outside"),

        # General off-axis points
        (0.3, 0.7, "off_axis_1"),
        (0.8, 0.4, "off_axis_2"),
        (1.5, 1.2, "off_axis_3"),
        (0.5, 0.5, "off_axis_4"),
        (0.2, 1.5, "off_axis_5"),

        # Far field
        (5.0, 5.0, "far_field_1"),
        (0.0, 10.0, "far_field_axis"),
        (10.0, 0.0, "far_field_radial"),
    ]

    def test_numerical_vs_analytical_N1000(self):
        """Compare with N=1000 point charges."""
        N = 1000
        max_rel_error = 0.0

        for r, a, name in self.TEST_POINTS:
            E_r_analytical, E_a_analytical = ring_electric_field(r, a, R, Q)
            E_r_numerical, E_a_numerical = numerical_efield_ring(r, a, R, Q, N=N)

            # Compute relative errors (use magnitude for reference)
            E_mag_analytical = np.sqrt(E_r_analytical**2 + E_a_analytical**2)

            if E_mag_analytical > 1e-20:
                rel_err_r = abs(E_r_numerical - E_r_analytical) / (E_mag_analytical + 1e-30)
                rel_err_a = abs(E_a_numerical - E_a_analytical) / (E_mag_analytical + 1e-30)
                rel_err = max(rel_err_r, rel_err_a)
                max_rel_error = max(max_rel_error, rel_err)

                # With N=1000, error should be < 1% for most points
                assert rel_err < 0.01, \
                    f"Large error at {name}: analytical=({E_r_analytical}, {E_a_analytical}), " \
                    f"numerical=({E_r_numerical}, {E_a_numerical}), rel_err={rel_err:.2e}"

    def test_numerical_vs_analytical_N10000(self):
        """Compare with N=10000 point charges for higher accuracy."""
        N = 10000

        for r, a, name in self.TEST_POINTS:
            E_r_analytical, E_a_analytical = ring_electric_field(r, a, R, Q)
            E_r_numerical, E_a_numerical = numerical_efield_ring(r, a, R, Q, N=N)

            E_mag_analytical = np.sqrt(E_r_analytical**2 + E_a_analytical**2)

            if E_mag_analytical > 1e-20:
                rel_err_r = abs(E_r_numerical - E_r_analytical) / (E_mag_analytical + 1e-30)
                rel_err_a = abs(E_a_numerical - E_a_analytical) / (E_mag_analytical + 1e-30)
                rel_err = max(rel_err_r, rel_err_a)

                # With N=10000, error should be < 0.1%
                assert rel_err < 0.001, \
                    f"Large error at {name}: analytical=({E_r_analytical}, {E_a_analytical}), " \
                    f"numerical=({E_r_numerical}, {E_a_numerical}), rel_err={rel_err:.2e}"

    def test_convergence(self):
        """Test that error decreases as N increases (convergence)."""
        # Use a point close to the ring where discretization error is significant
        test_point = (0.95, 0.05)  # Close to ring
        r, a = test_point

        E_r_analytical, E_a_analytical = ring_electric_field(r, a, R, Q)
        E_mag = np.sqrt(E_r_analytical**2 + E_a_analytical**2)

        N_values = [100, 500, 1000, 5000, 10000]
        errors = []

        for N in N_values:
            E_r_num, E_a_num = numerical_efield_ring(r, a, R, Q, N=N)
            err_r = abs(E_r_num - E_r_analytical) / E_mag
            err_a = abs(E_a_num - E_a_analytical) / E_mag
            errors.append(max(err_r, err_a))

        # Error should generally decrease with N
        # Check that error at N=10000 is less than at N=100
        # (at least factor of 2 improvement expected)
        assert errors[-1] < errors[0] / 2, \
            f"Convergence not observed: errors = {errors}"

        # If errors are already at machine precision, skip monotonicity check
        if errors[0] > 1e-12:
            # Check general downward trend (first vs last)
            assert errors[-1] < errors[0], \
                f"Error did not decrease: first={errors[0]}, last={errors[-1]}"


class TestNumericalEdgeCases:
    """Test numerical comparison at edge cases."""

    def test_on_axis_radial_zero(self):
        """On axis, E_r should be zero for both analytical and numerical."""
        a_values = [0.5, 1.0, 2.0]
        N = 5000

        for a in a_values:
            E_r_analytical, E_a_analytical = ring_electric_field(0.0, a, R, Q)
            E_r_numerical, E_a_numerical = numerical_efield_ring(0.0, a, R, Q, N=N)

            # Both should be essentially zero
            assert abs(E_r_analytical) < 1e-15, f"Analytical E_r not zero on axis"
            assert abs(E_r_numerical) < 1e-10, f"Numerical E_r not zero on axis: {E_r_numerical}"

    def test_ring_plane_axial_zero(self):
        """In ring plane, E_a should be zero for both analytical and numerical."""
        r_values = [0.5, 1.5, 2.0]
        N = 5000

        for r in r_values:
            E_r_analytical, E_a_analytical = ring_electric_field(r, 0.0, R, Q)
            E_r_numerical, E_a_numerical = numerical_efield_ring(r, 0.0, R, Q, N=N)

            # Both should be essentially zero
            assert abs(E_a_analytical) < 1e-15, f"Analytical E_a not zero in ring plane"
            assert abs(E_a_numerical) < 1e-10, f"Numerical E_a not zero in ring plane: {E_a_numerical}"

    def test_symmetry_above_below(self):
        """E_a should be antisymmetric in a."""
        r, a = 0.5, 0.7
        N = 5000

        E_r_pos, E_a_pos = numerical_efield_ring(r, a, R, Q, N=N)
        E_r_neg, E_a_neg = numerical_efield_ring(r, -a, R, Q, N=N)

        # E_a should flip sign
        np.testing.assert_allclose(E_a_pos, -E_a_neg, rtol=1e-6)
        # E_r should be the same
        np.testing.assert_allclose(E_r_pos, E_r_neg, rtol=1e-6)


class TestNumericalReport:
    """Generate a report of numerical vs analytical comparison."""

    def test_detailed_comparison_report(self):
        """Print detailed comparison (for debugging/verification)."""
        N = 10000

        test_points = [
            (0.0, 1.0, "on_axis"),
            (0.5, 0.0, "in_plane"),
            (0.5, 0.5, "off_axis"),
            (0.9, 0.1, "near_ring"),
            (5.0, 5.0, "far_field"),
        ]

        print("\n" + "="*80)
        print(f"Numerical vs Analytical Comparison (N={N} point charges)")
        print("="*80)
        print(f"{'Point':<12} {'r':>6} {'a':>6} | {'E_r anal':>12} {'E_r num':>12} {'err':>10} | {'E_a anal':>12} {'E_a num':>12} {'err':>10}")
        print("-"*80)

        max_err = 0.0
        for r, a, name in test_points:
            E_r_a, E_a_a = ring_electric_field(r, a, R, Q)
            E_r_n, E_a_n = numerical_efield_ring(r, a, R, Q, N=N)

            E_mag = np.sqrt(E_r_a**2 + E_a_a**2)
            if E_mag > 1e-20:
                err_r = abs(E_r_n - E_r_a) / E_mag
                err_a = abs(E_a_n - E_a_a) / E_mag
            else:
                err_r = err_a = 0.0

            max_err = max(max_err, err_r, err_a)

            print(f"{name:<12} {r:>6.2f} {a:>6.2f} | {E_r_a:>12.4e} {E_r_n:>12.4e} {err_r:>10.2e} | {E_a_a:>12.4e} {E_a_n:>12.4e} {err_a:>10.2e}")

        print("-"*80)
        print(f"Maximum relative error: {max_err:.2e}")
        print("="*80 + "\n")

        # This is also a test - max error should be small
        assert max_err < 0.001, f"Maximum error too large: {max_err}"


def run_tests_without_pytest():
    """Run tests without pytest."""
    passed = 0
    failed = 0

    test_classes = [
        TestNumericalComparison,
        TestNumericalEdgeCases,
        TestNumericalReport,
    ]

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()
        for name in dir(instance):
            if name.startswith('test_'):
                try:
                    getattr(instance, name)()
                    print(f"  PASS: {name}")
                    passed += 1
                except Exception as e:
                    print(f"  FAIL: {name}: {e}")
                    failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == '__main__':
    if HAS_PYTEST:
        pytest.main([__file__, '-v', '-s'])  # -s to show print output
    else:
        print("Running tests without pytest...")
        success = run_tests_without_pytest()
        sys.exit(0 if success else 1)
