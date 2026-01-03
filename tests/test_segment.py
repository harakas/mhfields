"""
Tests for line_magnetic_field function.

Copyright (C) 2008-2026 Indrek Mandre <indrek(at)mare.ee>
Licensed under the MIT License.
"""

import numpy as np
import pytest
from mhfields import line_magnetic_field


class TestLineMagneticField:
    """Tests for line_magnetic_field Biot-Savart implementation."""

    def test_wire_along_z_axis(self):
        """Wire along Z-axis, field at point on X-axis."""
        B = line_magnetic_field([0, 0, -1], [0, 0, 1], 1.0, [1, 0, 0])

        # Field should be in Y direction (right-hand rule)
        assert B[0] == pytest.approx(0, abs=1e-20)
        assert B[2] == pytest.approx(0, abs=1e-20)

        # For a wire from -1 to +1 on Z-axis, point at (1,0,0):
        # cos(theta) for both endpoints = 1/sqrt(2)
        # B = mu0*I/(4*pi*d) * 2*cos(theta) = 1e-7 * 2/sqrt(2) = sqrt(2)*1e-7
        expected_By = np.sqrt(2) * 1e-7
        assert B[1] == pytest.approx(expected_By, rel=1e-10)

    def test_wire_along_x_axis(self):
        """Wire along X-axis, field at point on Z-axis."""
        B = line_magnetic_field([-1, 0, 0], [1, 0, 0], 1.0, [0, 0, 1])

        # Field should be in -Y direction
        assert B[0] == pytest.approx(0, abs=1e-20)
        assert B[2] == pytest.approx(0, abs=1e-20)

        expected_By = -np.sqrt(2) * 1e-7
        assert B[1] == pytest.approx(expected_By, rel=1e-10)

    def test_current_direction_reversal(self):
        """Reversing current reverses field."""
        B_pos = line_magnetic_field([0, 0, -1], [0, 0, 1], 1.0, [1, 0, 0])
        B_neg = line_magnetic_field([0, 0, -1], [0, 0, 1], -1.0, [1, 0, 0])

        np.testing.assert_allclose(B_neg, -B_pos)

    def test_segment_direction_reversal(self):
        """Swapping p0 and p1 reverses field (current flows opposite)."""
        B1 = line_magnetic_field([0, 0, -1], [0, 0, 1], 1.0, [1, 0, 0])
        B2 = line_magnetic_field([0, 0, 1], [0, 0, -1], 1.0, [1, 0, 0])

        np.testing.assert_allclose(B2, -B1)

    def test_point_on_line_returns_zero(self):
        """Points exactly on the wire should return zero field."""
        B = line_magnetic_field([0, 0, -1], [0, 0, 1], 1.0, [0, 0, 0.5])
        np.testing.assert_allclose(B, [0, 0, 0], atol=1e-20)

    def test_point_at_endpoint(self):
        """Points at endpoints should return zero (singularity handled)."""
        B = line_magnetic_field([0, 0, 0], [0, 0, 1], 1.0, [0, 0, 0])
        np.testing.assert_allclose(B, [0, 0, 0], atol=1e-20)

    def test_zero_length_segment(self):
        """Zero-length segment produces no field."""
        B = line_magnetic_field([0, 0, 0], [0, 0, 0], 1.0, [1, 0, 0])
        np.testing.assert_allclose(B, [0, 0, 0], atol=1e-20)

    def test_multiple_points(self):
        """Test vectorized computation with multiple points."""
        points = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [2, 0, 0]
        ])
        B = line_magnetic_field([0, 0, -1], [0, 0, 1], 1.0, points)

        assert B.shape == (3, 3)

        # All fields should be perpendicular to wire (no Z component)
        np.testing.assert_allclose(B[:, 2], 0, atol=1e-20)

        # Point at distance 2 should have half the field of distance 1
        # (approximately, for finite wire)
        assert np.linalg.norm(B[2]) < np.linalg.norm(B[0])

    def test_single_point_output_shape(self):
        """Single point input should return 1D output."""
        B = line_magnetic_field([0, 0, -1], [0, 0, 1], 1.0, [1, 0, 0])
        assert B.shape == (3,)

    def test_symmetry(self):
        """Field should be symmetric about the wire."""
        B1 = line_magnetic_field([0, 0, -1], [0, 0, 1], 1.0, [1, 0, 0])
        B2 = line_magnetic_field([0, 0, -1], [0, 0, 1], 1.0, [-1, 0, 0])

        # Magnitudes should be equal
        assert np.linalg.norm(B1) == pytest.approx(np.linalg.norm(B2))

        # Directions should be opposite in the radial direction
        assert B1[1] == pytest.approx(-B2[1])

    def test_long_wire_approaches_infinite(self):
        """Long wire should approach infinite wire formula."""
        # For infinite wire: B = mu0 * I / (2 * pi * d)
        d = 0.1  # distance from wire
        I = 1.0
        B_infinite = 4 * np.pi * 1e-7 * I / (2 * np.pi * d)  # = 2e-6

        # Very long wire (-1000 to +1000)
        B_long = line_magnetic_field([0, 0, -1000], [0, 0, 1000], I, [d, 0, 0])

        # Should be within 0.01% of infinite wire
        assert abs(B_long[1]) == pytest.approx(B_infinite, rel=1e-4)
