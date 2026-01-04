"""
Tests for on-axis field values against known analytical formulae.

Copyright (C) 2007-2026 Indrek Mandre <indrek(at)mare.ee>
Licensed under the MIT License.

On-axis formulae (r=0):

Electric field:
    E_a = Q × a / (4πε₀ × (a² + R²)^(3/2))
    E_r = 0

Magnetic field:
    B_a = (μ₀/2) × I × R² / (a² + R²)^(3/2)
    B_r = 0
"""

import numpy as np
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from mhfields.fields import (
    ring_electric_field,
    ring_magnetic_field,
    ring_electric_field_xyz,
    ring_magnetic_field_xyz,
    MU_0,
    EPSILON_0,
)


# Test parameters
R = 1.0  # Ring radius (m)
I = 1.0  # Current (A)
Q = 1e-9  # Charge (C)
K_E = 1 / (4 * np.pi * EPSILON_0)  # Coulomb constant


class TestMagneticFieldOnAxis:
    """Test magnetic field on the ring axis (r=0)."""

    def test_on_axis_axial_component(self):
        """B_a on axis should match analytical formula."""
        a_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        r_values = np.zeros_like(a_values)

        B_r, B_a = ring_magnetic_field(r_values, a_values, R, I)

        # Analytical: B_a = (μ₀/2) × I × R² / (a² + R²)^(3/2)
        B_a_analytical = (MU_0 / 2) * I * R**2 / (a_values**2 + R**2)**1.5

        np.testing.assert_allclose(B_a, B_a_analytical, rtol=1e-10)

    def test_on_axis_radial_is_zero(self):
        """B_r on axis should be zero by symmetry."""
        a_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        r_values = np.zeros_like(a_values)

        B_r, B_a = ring_magnetic_field(r_values, a_values, R, I)

        np.testing.assert_allclose(B_r, 0.0, atol=1e-20)

    def test_symmetry_axial(self):
        """B_a should be symmetric: B_a(a) = B_a(-a)."""
        a_values = np.array([0.1, 0.5, 1.0, 2.0])

        B_r_pos, B_a_pos = ring_magnetic_field(np.zeros_like(a_values), a_values, R, I)
        B_r_neg, B_a_neg = ring_magnetic_field(np.zeros_like(a_values), -a_values, R, I)

        np.testing.assert_allclose(B_a_pos, B_a_neg, rtol=1e-10)

    def test_at_ring_center(self):
        """B_a at the center of the ring (r=0, a=0)."""
        B_r, B_a = ring_magnetic_field(0.0, 0.0, R, I)

        # B_a = (μ₀/2) × I × R² / R³ = μ₀I/(2R)
        B_a_expected = MU_0 * I / (2 * R)

        np.testing.assert_allclose(B_a, B_a_expected, rtol=1e-10)


class TestElectricFieldOnAxis:
    """Test electric field on the ring axis (r=0)."""

    def test_on_axis_axial_component(self):
        """E_a on axis should match analytical formula."""
        a_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        r_values = np.zeros_like(a_values)

        E_r, E_a = ring_electric_field(r_values, a_values, R, Q)

        # Analytical: E_a = Q × a / (4πε₀ × (a² + R²)^(3/2))
        E_a_analytical = K_E * Q * a_values / (a_values**2 + R**2)**1.5

        np.testing.assert_allclose(E_a, E_a_analytical, rtol=1e-10)

    def test_on_axis_radial_is_zero(self):
        """E_r on axis should be zero by symmetry."""
        a_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        r_values = np.zeros_like(a_values)

        E_r, E_a = ring_electric_field(r_values, a_values, R, Q)

        np.testing.assert_allclose(E_r, 0.0, atol=1e-20)

    def test_antisymmetry_axial(self):
        """E_a should be antisymmetric: E_a(-a) = -E_a(a)."""
        a_values = np.array([0.1, 0.5, 1.0, 2.0])

        E_r_pos, E_a_pos = ring_electric_field(np.zeros_like(a_values), a_values, R, Q)
        E_r_neg, E_a_neg = ring_electric_field(np.zeros_like(a_values), -a_values, R, Q)

        np.testing.assert_allclose(E_a_pos, -E_a_neg, rtol=1e-10)

    def test_at_ring_center(self):
        """E_a at the center of the ring (r=0, a=0) should be zero."""
        E_r, E_a = ring_electric_field(0.0, 0.0, R, Q)

        # At center, E = 0 by symmetry
        np.testing.assert_allclose(E_a, 0.0, atol=1e-20)
        np.testing.assert_allclose(E_r, 0.0, atol=1e-20)


class TestCartesianAPI:
    """Test Cartesian coordinate API."""

    def test_xyz_on_z_axis(self):
        """Field on Z axis should have only Bz/Ez component."""
        z_values = np.array([0.5, 1.0, 2.0])
        x_values = np.zeros_like(z_values)
        y_values = np.zeros_like(z_values)

        Bx, By, Bz = ring_magnetic_field_xyz(x_values, y_values, z_values, R, I)

        # On Z axis, only Bz should be non-zero
        np.testing.assert_allclose(Bx, 0.0, atol=1e-20)
        np.testing.assert_allclose(By, 0.0, atol=1e-20)

        # Compare with cylindrical
        B_r, B_a = ring_magnetic_field(np.zeros_like(z_values), z_values, R, I)
        np.testing.assert_allclose(Bz, B_a, rtol=1e-10)

    def test_xyz_in_xz_plane(self):
        """Field in XZ plane should have Bx and Bz components, By=0."""
        x = 0.5
        z = 0.5

        Bx, By, Bz = ring_magnetic_field_xyz(x, 0.0, z, R, I)

        # By should be zero in XZ plane
        np.testing.assert_allclose(By, 0.0, atol=1e-20)

        # Compare with cylindrical
        B_r, B_a = ring_magnetic_field(x, z, R, I)
        np.testing.assert_allclose(Bx, B_r, rtol=1e-10)
        np.testing.assert_allclose(Bz, B_a, rtol=1e-10)


class TestFarField:
    """Test far-field behavior (dipole approximation)."""

    def test_magnetic_dipole_far_field(self):
        """Far from the ring, B should approach magnetic dipole field."""
        # Magnetic dipole moment: m = I × π × R²
        m = I * np.pi * R**2

        # Far field on axis: B_a ≈ (μ₀/4π) × 2m/z³
        z_far = 100.0  # Very far from ring

        B_r, B_a = ring_magnetic_field(0.0, z_far, R, I)

        # Dipole approximation
        B_dipole = (MU_0 / (4 * np.pi)) * 2 * m / z_far**3

        # Should match to within a few percent at this distance
        np.testing.assert_allclose(B_a, B_dipole, rtol=0.01)

    def test_electric_far_field(self):
        """Far from the ring, E should approach point charge field."""
        z_far = 100.0  # Very far from ring

        E_r, E_a = ring_electric_field(0.0, z_far, R, Q)

        # Point charge approximation: E = Q/(4πε₀z²)
        E_point = K_E * Q / z_far**2

        # Should match to within a few percent at this distance
        np.testing.assert_allclose(E_a, E_point, rtol=0.01)


class TestNearAxisBehavior:
    """Test behavior near the axis (small r)."""

    def test_smooth_transition_bfield(self):
        """B_r should vary smoothly as r → 0."""
        a = 0.5
        r_values = np.array([0.001, 0.01, 0.1])

        B_r, B_a = ring_magnetic_field(r_values, a, R, I)

        # B_r should be small and vary roughly linearly with r near axis
        # Just check it doesn't blow up
        assert np.all(np.isfinite(B_r))
        assert np.all(np.isfinite(B_a))

    def test_smooth_transition_efield(self):
        """E_r should vary smoothly as r → 0."""
        a = 0.5
        r_values = np.array([0.001, 0.01, 0.1])

        E_r, E_a = ring_electric_field(r_values, a, R, Q)

        # E_r should be small and vary roughly linearly with r near axis
        assert np.all(np.isfinite(E_r))
        assert np.all(np.isfinite(E_a))


def run_tests_without_pytest():
    """Run tests without pytest."""
    passed = 0
    failed = 0

    test_classes = [
        TestMagneticFieldOnAxis,
        TestElectricFieldOnAxis,
        TestCartesianAPI,
        TestFarField,
        TestNearAxisBehavior,
    ]

    for test_class in test_classes:
        instance = test_class()
        for name in dir(instance):
            if name.startswith('test_'):
                try:
                    getattr(instance, name)()
                    print(f"  PASS: {test_class.__name__}.{name}")
                    passed += 1
                except Exception as e:
                    print(f"  FAIL: {test_class.__name__}.{name}: {e}")
                    failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == '__main__':
    if HAS_PYTEST:
        pytest.main([__file__, '-v'])
    else:
        print("Running tests without pytest...")
        print()
        success = run_tests_without_pytest()
        sys.exit(0 if success else 1)
