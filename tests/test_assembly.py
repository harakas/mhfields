"""
Tests for Assembly class.

Copyright (C) 2007-2026 Indrek Mandre <indrek(at)mare.ee>
Licensed under the MIT License.
"""

import numpy as np
import pytest
from mhfields import Assembly, CurrentLoop, CurrentSegment


class TestAssemblySuperposition:
    """Tests for field superposition in Assembly."""

    def test_empty_assembly(self):
        """Empty assembly should return zero field."""
        asm = Assembly()
        B = asm.bfield([0, 0, 0])
        np.testing.assert_allclose(B, [0, 0, 0])

    def test_single_element(self):
        """Assembly with one element should match that element."""
        loop = CurrentLoop([0, 0, 0], [0, 0, 1], 0.1, 1.0)
        asm = Assembly()
        asm.add(loop)

        B_asm = asm.bfield([0.05, 0, 0.05])
        B_loop = loop.bfield([0.05, 0, 0.05])

        np.testing.assert_allclose(B_asm, B_loop)

    def test_superposition(self):
        """Total field should be sum of individual fields."""
        loop1 = CurrentLoop([0, 0, -0.05], [0, 0, 1], 0.1, 1.0)
        loop2 = CurrentLoop([0, 0, 0.05], [0, 0, 1], 0.1, 1.0)

        asm = Assembly()
        asm.add(loop1)
        asm.add(loop2)

        point = [0.02, 0.03, 0]
        B_asm = asm.bfield(point)
        B_sum = loop1.bfield(point) + loop2.bfield(point)

        np.testing.assert_allclose(B_asm, B_sum)

    def test_helmholtz_coil(self):
        """Helmholtz coil field at center matches theory."""
        R = 0.1
        I = 1.0
        d = R  # Helmholtz spacing

        asm = Assembly()
        asm.add(CurrentLoop([0, 0, -d/2], [0, 0, 1], R, I))
        asm.add(CurrentLoop([0, 0, d/2], [0, 0, 1], R, I))

        B = asm.bfield([0, 0, 0])

        # Helmholtz formula: B = 8 * mu0 * N * I / (5^1.5 * R)
        mu0 = 4 * np.pi * 1e-7
        B_theory = 8 * mu0 * I / (5**1.5 * R)

        assert B[2] == pytest.approx(B_theory, rel=1e-10)

    def test_rectangular_coil(self):
        """Rectangular coil from 4 segments produces sensible field."""
        L = 0.1
        corners = [
            [-L/2, -L/2, 0],
            [L/2, -L/2, 0],
            [L/2, L/2, 0],
            [-L/2, L/2, 0]
        ]

        asm = Assembly()
        for i in range(4):
            asm.add(CurrentSegment(
                corners[i], corners[(i+1) % 4], current=1.0
            ))

        # Field at center above the coil
        B = asm.bfield([0, 0, 0.05])

        # Should be primarily in Z direction (normal to coil)
        assert abs(B[2]) > abs(B[0])
        assert abs(B[2]) > abs(B[1])

    def test_method_chaining(self):
        """add() should return self for chaining."""
        asm = Assembly()
        result = asm.add(CurrentLoop([0, 0, 0], [0, 0, 1], 0.1, 1.0))

        assert result is asm


class TestAssemblyBoundingBox:
    """Tests for bounding box computation."""

    def test_empty_bounding_box(self):
        """Empty assembly should return zero bounding box."""
        asm = Assembly()
        bb_min, bb_max = asm.get_bounding_box()

        np.testing.assert_allclose(bb_min, [0, 0, 0])
        np.testing.assert_allclose(bb_max, [0, 0, 0])

    def test_single_element_bounding_box(self):
        """Single element bounding box."""
        loop = CurrentLoop([1, 2, 3], [0, 0, 1], 0.1, 1.0)
        asm = Assembly()
        asm.add(loop)

        bb_min, bb_max = asm.get_bounding_box()

        assert bb_min[0] == pytest.approx(0.9, rel=0.01)
        assert bb_max[0] == pytest.approx(1.1, rel=0.01)
        assert bb_min[2] == pytest.approx(3.0, abs=0.01)
        assert bb_max[2] == pytest.approx(3.0, abs=0.01)

    def test_multiple_elements_bounding_box(self):
        """Bounding box should contain all elements."""
        asm = Assembly()
        asm.add(CurrentLoop([0, 0, 0], [0, 0, 1], 0.1, 1.0))
        asm.add(CurrentLoop([0, 0, 1], [0, 0, 1], 0.1, 1.0))
        asm.add(CurrentSegment([0.5, 0, 0], [0.5, 0, 1], current=1.0))

        bb_min, bb_max = asm.get_bounding_box()

        assert bb_min[0] < -0.09
        assert bb_max[0] >= 0.5
        assert bb_min[2] < 0.01
        assert bb_max[2] > 0.99


class TestAssemblyIteration:
    """Tests for iterating over assembly elements."""

    def test_len(self):
        """len() should return number of elements."""
        asm = Assembly()
        assert len(asm) == 0

        asm.add(CurrentLoop([0, 0, 0], [0, 0, 1], 0.1, 1.0))
        assert len(asm) == 1

        asm.add(CurrentSegment([0, 0, 0], [1, 0, 0], current=1.0))
        assert len(asm) == 2

    def test_iter(self):
        """Should be able to iterate over elements."""
        loop = CurrentLoop([0, 0, 0], [0, 0, 1], 0.1, 1.0)
        seg = CurrentSegment([0, 0, 0], [1, 0, 0], current=1.0)

        asm = Assembly()
        asm.add(loop)
        asm.add(seg)

        elements = list(asm)
        assert len(elements) == 2
        assert elements[0] is loop
        assert elements[1] is seg


class TestAssemblyVisualization:
    """Smoke tests for visualization methods."""

    @pytest.fixture
    def helmholtz_assembly(self):
        """Create a simple Helmholtz coil for testing."""
        asm = Assembly()
        asm.add(CurrentLoop([0, 0, -0.05], [0, 0, 1], 0.1, 1.0))
        asm.add(CurrentLoop([0, 0, 0.05], [0, 0, 1], 0.1, 1.0))
        return asm

    def test_slice_xz(self, helmholtz_assembly):
        """slice_xz should run without error."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        fig, ax = helmholtz_assembly.slice_xz(y=0, show=False)
        assert fig is not None
        assert ax is not None

    def test_slice_xy(self, helmholtz_assembly):
        """slice_xy should run without error."""
        import matplotlib
        matplotlib.use('Agg')

        fig, ax = helmholtz_assembly.slice_xy(z=0, show=False)
        assert fig is not None

    def test_slice_yz(self, helmholtz_assembly):
        """slice_yz should run without error."""
        import matplotlib
        matplotlib.use('Agg')

        fig, ax = helmholtz_assembly.slice_yz(x=0, show=False)
        assert fig is not None

    def test_slice_generic(self, helmholtz_assembly):
        """Generic slice() method should work."""
        import matplotlib
        matplotlib.use('Agg')

        fig, ax = helmholtz_assembly.slice('xz', offset=0, show=False)
        assert fig is not None

    def test_slice_invalid_plane(self, helmholtz_assembly):
        """Invalid plane should raise error."""
        with pytest.raises(ValueError):
            helmholtz_assembly.slice('invalid', offset=0)

    def test_slice_with_custom_extent(self, helmholtz_assembly):
        """Custom extent should be respected."""
        import matplotlib
        matplotlib.use('Agg')

        extent = ((-0.5, 0.5), (-0.3, 0.3))
        fig, ax = helmholtz_assembly.slice_xz(
            y=0, extent=extent, show=False
        )
        assert fig is not None

    def test_draw_geometry(self, helmholtz_assembly):
        """draw_geometry should run without error."""
        import matplotlib
        matplotlib.use('Agg')

        ax = helmholtz_assembly.draw_geometry()
        assert ax is not None
