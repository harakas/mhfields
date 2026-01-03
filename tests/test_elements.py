"""
Tests for CurrentLoop and CurrentSegment classes.

Copyright (C) 2007-2026 Indrek Mandre <indrek(at)mare.ee>
Licensed under the MIT License.
"""

import numpy as np
import pytest
from mhfields import CurrentLoop, CurrentSegment, ring_magnetic_field_xyz


class TestCurrentLoop:
    """Tests for CurrentLoop class."""

    def test_matches_ring_magnetic_field_xyz(self):
        """Loop at origin with Z-normal should match ring_magnetic_field_xyz."""
        R, I = 0.1, 1.0
        loop = CurrentLoop([0, 0, 0], [0, 0, 1], R, I)

        points = np.array([
            [0.05, 0, 0.05],
            [0, 0, 0.1],
            [0.08, 0.06, 0.02]
        ])

        B_loop = loop.bfield(points)
        Bx, By, Bz = ring_magnetic_field_xyz(
            points[:, 0], points[:, 1], points[:, 2], R, I
        )
        B_direct = np.column_stack([Bx, By, Bz])

        np.testing.assert_allclose(B_loop, B_direct, rtol=1e-14, atol=1e-20)

    def test_on_axis_field(self):
        """On-axis field should follow analytical formula."""
        R, I = 0.1, 1.0
        loop = CurrentLoop([0, 0, 0], [0, 0, 1], R, I)

        z = 0.1
        B = loop.bfield([0, 0, z])

        # B_z = mu0 * I * R^2 / (2 * (z^2 + R^2)^1.5)
        mu0 = 4 * np.pi * 1e-7
        B_theory = mu0 * I * R**2 / (2 * (z**2 + R**2)**1.5)

        assert B[0] == pytest.approx(0, abs=1e-20)
        assert B[1] == pytest.approx(0, abs=1e-20)
        assert B[2] == pytest.approx(B_theory, rel=1e-10)

    def test_field_at_center(self):
        """Field at center follows B = mu0 * I / (2 * R)."""
        R, I = 0.1, 1.0
        loop = CurrentLoop([0, 0, 0], [0, 0, 1], R, I)

        B = loop.bfield([0, 0, 0])

        mu0 = 4 * np.pi * 1e-7
        B_theory = mu0 * I / (2 * R)

        assert B[2] == pytest.approx(B_theory, rel=1e-10)

    def test_tilted_loop(self):
        """Tilted loop field should be along normal at center."""
        R, I = 0.1, 1.0
        normal = np.array([1, 0, 1]) / np.sqrt(2)
        loop = CurrentLoop([0, 0, 0], normal, R, I)

        B = loop.bfield([0, 0, 0])

        # Field at center should be along normal
        B_normalized = B / np.linalg.norm(B)
        np.testing.assert_allclose(B_normalized, normal, rtol=1e-10)

    def test_translated_loop(self):
        """Translated loop should work correctly."""
        R, I = 0.1, 1.0
        center = np.array([1, 2, 3])
        loop = CurrentLoop(center, [0, 0, 1], R, I)

        # Field at center should be same as origin-centered loop
        B = loop.bfield(center)
        loop_origin = CurrentLoop([0, 0, 0], [0, 0, 1], R, I)
        B_origin = loop_origin.bfield([0, 0, 0])

        np.testing.assert_allclose(B, B_origin, rtol=1e-12)

    def test_get_geometry(self):
        """get_geometry should return a closed circle."""
        loop = CurrentLoop([1, 2, 3], [0, 0, 1], 0.1, 1.0)
        pts = loop.get_geometry()

        # Should be a circle in XY plane
        assert pts.shape == (100, 3)

        # All z values should be 3
        np.testing.assert_allclose(pts[:, 2], 3.0)

        # Radius from center should be 0.1
        dists = np.sqrt((pts[:, 0] - 1)**2 + (pts[:, 1] - 2)**2)
        np.testing.assert_allclose(dists, 0.1, rtol=1e-10)

    def test_get_bounding_box(self):
        """Bounding box should contain the loop."""
        loop = CurrentLoop([0, 0, 0], [0, 0, 1], 0.1, 1.0)
        bb_min, bb_max = loop.get_bounding_box()

        assert bb_min[0] == pytest.approx(-0.1, rel=1e-3)
        assert bb_max[0] == pytest.approx(0.1, rel=1e-3)
        assert bb_min[2] == pytest.approx(0, abs=1e-10)
        assert bb_max[2] == pytest.approx(0, abs=1e-10)


class TestCurrentSegment:
    """Tests for CurrentSegment class."""

    def test_endpoint_constructor(self):
        """Test construction with p0 and p1."""
        seg = CurrentSegment([0, 0, -1], [0, 0, 1], current=1.0)

        assert seg.length == pytest.approx(2.0)
        np.testing.assert_allclose(seg.direction, [0, 0, 1])
        np.testing.assert_allclose(seg.p0, [0, 0, -1])
        np.testing.assert_allclose(seg.p1, [0, 0, 1])

    def test_direction_length_constructor(self):
        """Test construction with direction and length."""
        seg = CurrentSegment(
            [1, 2, 3], direction=[1, 0, 0], length=2.0, current=1.0
        )

        assert seg.length == pytest.approx(2.0)
        np.testing.assert_allclose(seg.direction, [1, 0, 0])
        np.testing.assert_allclose(seg.p0, [1, 2, 3])
        np.testing.assert_allclose(seg.p1, [3, 2, 3])

    def test_constructors_equivalent(self):
        """Both constructors should produce same field."""
        seg1 = CurrentSegment([0, 0, -1], [0, 0, 1], current=1.0)
        seg2 = CurrentSegment(
            [0, 0, -1], direction=[0, 0, 1], length=2.0, current=1.0
        )

        B1 = seg1.bfield([1, 0, 0])
        B2 = seg2.bfield([1, 0, 0])

        np.testing.assert_allclose(B1, B2)

    def test_bfield(self):
        """Test that bfield returns expected values."""
        seg = CurrentSegment([0, 0, -1], [0, 0, 1], current=1.0)
        B = seg.bfield([1, 0, 0])

        # Should match line_magnetic_field directly
        from mhfields import line_magnetic_field
        B_direct = line_magnetic_field([0, 0, -1], [0, 0, 1], 1.0, [1, 0, 0])

        np.testing.assert_allclose(B, B_direct)

    def test_get_geometry(self):
        """get_geometry should return endpoints."""
        seg = CurrentSegment([1, 2, 3], [4, 5, 6], current=1.0)
        pts = seg.get_geometry()

        assert pts.shape == (2, 3)
        np.testing.assert_allclose(pts[0], [1, 2, 3])
        np.testing.assert_allclose(pts[1], [4, 5, 6])

    def test_get_bounding_box(self):
        """Bounding box should contain the segment."""
        seg = CurrentSegment([1, 2, 3], [4, 5, 6], current=1.0)
        bb_min, bb_max = seg.get_bounding_box()

        np.testing.assert_allclose(bb_min, [1, 2, 3])
        np.testing.assert_allclose(bb_max, [4, 5, 6])

    def test_invalid_constructor(self):
        """Should raise error for invalid constructor arguments."""
        with pytest.raises(ValueError):
            CurrentSegment([0, 0, 0])  # Missing p1 or direction/length

        with pytest.raises(ValueError):
            # Both p1 and direction given
            CurrentSegment([0, 0, 0], [1, 1, 1], direction=[1, 0, 0], length=1)
