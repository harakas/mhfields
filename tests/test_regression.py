"""
Regression tests for ring field calculations.

Copyright (C) 2007-2026 Indrek Mandre <indrek(at)mare.ee>
Licensed under the MIT License.

These tests capture the current output values at various important points
to detect any unintended changes in the calculation results.
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
K_E = 1 / (4 * np.pi * EPSILON_0)


# Regression baselines: (r, a, E_r, E_a)
EFIELD_REGRESSION = {
    'origin': (0.0, 0.0, 0.000000000000000e+00, 0.000000000000000e+00),
    'on_axis_near': (0.0, 0.5, 0.000000000000000e+00, 3.215484279882098e+00),
    'on_axis_at_R': (0.0, 1.0, 0.000000000000000e+00, 3.177579407779302e+00),
    'on_axis_2R': (0.0, 2.0, 0.000000000000000e+00, 1.607742139941049e+00),
    'on_axis_far': (0.0, 10.0, 0.000000000000000e+00, 8.854404235639809e-02),
    'on_axis_negative': (0.0, -0.5, 0.000000000000000e+00, -3.215484279882098e+00),
    'near_axis_tiny': (1e-06, 0.5, -1.285767216978188e-06, 3.215484279885955e+00),
    'near_axis_small': (0.001, 0.5, -1.286192168419210e-03, 3.215488138464931e+00),
    'near_axis_medium': (0.01, 0.5, -1.286039343391498e-02, 3.215870154972605e+00),
    'near_axis_large': (0.1, 0.5, -1.270504826788884e-01, 3.254239014241950e+00),
    'ring_plane_inside': (0.5, 0.0, -3.099601750759273e+00, 0.000000000000000e+00),
    'ring_plane_near_ring': (0.9, 0.0, -2.470659081569049e+01, 0.000000000000000e+00),
    'ring_plane_outside_near': (1.1, 0.0, 3.161327664375262e+01, 0.000000000000000e+00),
    'ring_plane_outside_far': (2.0, 0.0, 2.798769935645523e+00, 0.000000000000000e+00),
    'near_ring_inside': (0.99, 0.01, -1.360354786090963e+02, 1.437794896521408e+02),
    'near_ring_outside': (1.01, 0.01, 1.498777345294097e+02, 1.423485036636764e+02),
    'at_R_small_a': (1.0, 0.1, 4.829538094021229e+00, 2.871103063253615e+01),
    'at_R_tiny_a': (1.0, 0.01, 8.131200663538518e+00, 2.861011984858003e+02),
    'random_1': (0.3, 0.7, 4.398978250813239e-02, 3.666998158642041e+00),
    'random_2': (0.8, 0.4, 9.916907395881329e-02, 6.633522911304143e+00),
    'random_3': (1.5, 1.2, 1.430908270501918e+00, 1.745796284957221e+00),
    'random_4': (0.123, 0.456, -1.990526979290686e-01, 3.149607586261375e+00),
    'random_5': (0.789, 1.234, 8.390178974368213e-01, 2.612449539230944e+00),
    'far_diagonal': (10.0, 10.0, 3.159664046697546e-02, 3.183458085846699e-02),
    'very_far_axis': (0.0, 100.0, 0.000000000000000e+00, 8.986203823726650e-04),
    'very_far_diagonal': (50.0, 50.0, 1.270745754164796e-03, 1.271127039847288e-03),
    'very_far_radial': (100.0, 0.0, 8.988225917581881e-04, 0.000000000000000e+00),
}

# Regression baselines: (r, a, B_r, B_a)
BFIELD_REGRESSION = {
    'origin': (0.0, 0.0, 0.000000000000000e+00, 6.283185307179586e-07),
    'on_axis_near': (0.0, 0.5, 0.000000000000000e+00, 4.495881427866064e-07),
    'on_axis_at_R': (0.0, 1.0, 0.000000000000000e+00, 2.221441469079183e-07),
    'on_axis_2R': (0.0, 2.0, 0.000000000000000e+00, 5.619851784832580e-08),
    'on_axis_far': (0.0, 10.0, 0.000000000000000e+00, 6.190102033291746e-10),
    'on_axis_negative': (0.0, -0.5, 0.000000000000000e+00, 4.495881427866064e-07),
    'near_axis_tiny': (1e-06, 0.5, 2.697221549042642e-13, 4.495881427866064e-07),
    'near_axis_small': (0.001, 0.5, 2.697531014635786e-10, 4.495881427864123e-07),
    'near_axis_medium': (0.01, 0.5, 2.697744662047416e-09, 4.495881408442084e-07),
    'near_axis_large': (0.1, 0.5, 2.719138199922393e-08, 4.495685428685740e-07),
    'ring_plane_inside': (0.5, 0.0, 0.000000000000000e+00, 7.826465116476944e-07),
    'ring_plane_near_ring': (0.9, 0.0, 0.000000000000000e+00, 2.466730637434929e-06),
    'ring_plane_outside_near': (1.1, 0.0, 0.000000000000000e+00, -1.586751873581711e-06),
    'ring_plane_outside_far': (2.0, 0.0, 0.000000000000000e+00, -5.417318486132805e-08),
    'near_ring_inside': (0.99, 0.01, 1.004619086031016e-05, 1.058757362562868e-05),
    'near_ring_outside': (1.01, 0.01, 9.946300975809179e-06, -9.419933463655852e-06),
    'at_R_small_a': (1.0, 0.1, 1.973421035324711e-06, 3.376323553800382e-07),
    'at_R_tiny_a': (1.0, 0.01, 1.999561160454572e-05, 5.684511393534241e-07),
    'random_1': (0.3, 0.7, 7.475507986432592e-08, 3.341897452433114e-07),
    'random_2': (0.8, 0.4, 3.682257934773871e-07, 4.229201383576976e-07),
    'random_3': (1.5, 1.2, 6.303104892987239e-08, 2.291817410013726e-08),
    'random_4': (0.123, 0.456, 3.342878352622918e-08, 4.738528514750401e-07),
    'random_5': (0.789, 1.234, 7.171875469715030e-08, 1.021474041540359e-07),
    'far_diagonal': (10.0, 10.0, 1.663438060852812e-10, 5.621131841175050e-11),
    'very_far_axis': (0.0, 100.0, 0.000000000000000e+00, 6.282242947179491e-13),
    'very_far_diagonal': (50.0, 50.0, 1.332781526372117e-12, 4.445048645649156e-13),
    'very_far_radial': (100.0, 0.0, 0.000000000000000e+00, -3.141946119581704e-13),
}


class TestElectricFieldRegression:
    """Regression tests for electric field values."""

    def test_on_axis_points(self):
        """Test on-axis points (r=0)."""
        on_axis_cases = ['origin', 'on_axis_near', 'on_axis_at_R', 'on_axis_2R',
                         'on_axis_far', 'on_axis_negative']
        for name in on_axis_cases:
            r, a, expected_Er, expected_Ea = EFIELD_REGRESSION[name]
            E_r, E_a = ring_electric_field(r, a, R, Q)
            np.testing.assert_allclose(E_r, expected_Er, atol=1e-15,
                err_msg=f"E_r mismatch at {name}")
            np.testing.assert_allclose(E_a, expected_Ea, rtol=1e-10, atol=1e-15,
                err_msg=f"E_a mismatch at {name}")

    def test_near_axis_points(self):
        """Test points close to axis (small r)."""
        near_axis_cases = ['near_axis_tiny', 'near_axis_small',
                          'near_axis_medium', 'near_axis_large']
        for name in near_axis_cases:
            r, a, expected_Er, expected_Ea = EFIELD_REGRESSION[name]
            E_r, E_a = ring_electric_field(r, a, R, Q)
            np.testing.assert_allclose(E_r, expected_Er, rtol=1e-6,
                err_msg=f"E_r mismatch at {name}")
            np.testing.assert_allclose(E_a, expected_Ea, rtol=1e-10,
                err_msg=f"E_a mismatch at {name}")

    def test_ring_plane_points(self):
        """Test points in the ring plane (a=0)."""
        plane_cases = ['ring_plane_inside', 'ring_plane_near_ring',
                      'ring_plane_outside_near', 'ring_plane_outside_far']
        for name in plane_cases:
            r, a, expected_Er, expected_Ea = EFIELD_REGRESSION[name]
            E_r, E_a = ring_electric_field(r, a, R, Q)
            np.testing.assert_allclose(E_r, expected_Er, rtol=1e-10,
                err_msg=f"E_r mismatch at {name}")
            np.testing.assert_allclose(E_a, expected_Ea, atol=1e-15,
                err_msg=f"E_a mismatch at {name}")

    def test_near_ring_points(self):
        """Test points close to the ring (r≈R, a≈0)."""
        near_ring_cases = ['near_ring_inside', 'near_ring_outside',
                          'at_R_small_a', 'at_R_tiny_a']
        for name in near_ring_cases:
            r, a, expected_Er, expected_Ea = EFIELD_REGRESSION[name]
            E_r, E_a = ring_electric_field(r, a, R, Q)
            np.testing.assert_allclose(E_r, expected_Er, rtol=1e-10,
                err_msg=f"E_r mismatch at {name}")
            np.testing.assert_allclose(E_a, expected_Ea, rtol=1e-10,
                err_msg=f"E_a mismatch at {name}")

    def test_random_points(self):
        """Test random off-axis points."""
        random_cases = ['random_1', 'random_2', 'random_3', 'random_4', 'random_5']
        for name in random_cases:
            r, a, expected_Er, expected_Ea = EFIELD_REGRESSION[name]
            E_r, E_a = ring_electric_field(r, a, R, Q)
            np.testing.assert_allclose(E_r, expected_Er, rtol=1e-10,
                err_msg=f"E_r mismatch at {name}")
            np.testing.assert_allclose(E_a, expected_Ea, rtol=1e-10,
                err_msg=f"E_a mismatch at {name}")

    def test_far_field_points(self):
        """Test far field points."""
        far_cases = ['far_diagonal', 'very_far_axis', 'very_far_diagonal', 'very_far_radial']
        for name in far_cases:
            r, a, expected_Er, expected_Ea = EFIELD_REGRESSION[name]
            E_r, E_a = ring_electric_field(r, a, R, Q)
            np.testing.assert_allclose(E_r, expected_Er, rtol=1e-10, atol=1e-15,
                err_msg=f"E_r mismatch at {name}")
            np.testing.assert_allclose(E_a, expected_Ea, rtol=1e-10, atol=1e-15,
                err_msg=f"E_a mismatch at {name}")


class TestMagneticFieldRegression:
    """Regression tests for magnetic field values."""

    def test_on_axis_points(self):
        """Test on-axis points (r=0)."""
        on_axis_cases = ['origin', 'on_axis_near', 'on_axis_at_R', 'on_axis_2R',
                         'on_axis_far', 'on_axis_negative']
        for name in on_axis_cases:
            r, a, expected_Br, expected_Ba = BFIELD_REGRESSION[name]
            B_r, B_a = ring_magnetic_field(r, a, R, I)
            np.testing.assert_allclose(B_r, expected_Br, atol=1e-20,
                err_msg=f"B_r mismatch at {name}")
            np.testing.assert_allclose(B_a, expected_Ba, rtol=1e-10,
                err_msg=f"B_a mismatch at {name}")

    def test_near_axis_points(self):
        """Test points close to axis (small r)."""
        near_axis_cases = ['near_axis_tiny', 'near_axis_small',
                          'near_axis_medium', 'near_axis_large']
        for name in near_axis_cases:
            r, a, expected_Br, expected_Ba = BFIELD_REGRESSION[name]
            B_r, B_a = ring_magnetic_field(r, a, R, I)
            np.testing.assert_allclose(B_r, expected_Br, rtol=1e-6,
                err_msg=f"B_r mismatch at {name}")
            np.testing.assert_allclose(B_a, expected_Ba, rtol=1e-10,
                err_msg=f"B_a mismatch at {name}")

    def test_ring_plane_points(self):
        """Test points in the ring plane (a=0)."""
        plane_cases = ['ring_plane_inside', 'ring_plane_near_ring',
                      'ring_plane_outside_near', 'ring_plane_outside_far']
        for name in plane_cases:
            r, a, expected_Br, expected_Ba = BFIELD_REGRESSION[name]
            B_r, B_a = ring_magnetic_field(r, a, R, I)
            np.testing.assert_allclose(B_r, expected_Br, atol=1e-20,
                err_msg=f"B_r mismatch at {name}")
            np.testing.assert_allclose(B_a, expected_Ba, rtol=1e-10,
                err_msg=f"B_a mismatch at {name}")

    def test_near_ring_points(self):
        """Test points close to the ring (r≈R, a≈0)."""
        near_ring_cases = ['near_ring_inside', 'near_ring_outside',
                          'at_R_small_a', 'at_R_tiny_a']
        for name in near_ring_cases:
            r, a, expected_Br, expected_Ba = BFIELD_REGRESSION[name]
            B_r, B_a = ring_magnetic_field(r, a, R, I)
            np.testing.assert_allclose(B_r, expected_Br, rtol=1e-10,
                err_msg=f"B_r mismatch at {name}")
            np.testing.assert_allclose(B_a, expected_Ba, rtol=1e-10,
                err_msg=f"B_a mismatch at {name}")

    def test_random_points(self):
        """Test random off-axis points."""
        random_cases = ['random_1', 'random_2', 'random_3', 'random_4', 'random_5']
        for name in random_cases:
            r, a, expected_Br, expected_Ba = BFIELD_REGRESSION[name]
            B_r, B_a = ring_magnetic_field(r, a, R, I)
            np.testing.assert_allclose(B_r, expected_Br, rtol=1e-10,
                err_msg=f"B_r mismatch at {name}")
            np.testing.assert_allclose(B_a, expected_Ba, rtol=1e-10,
                err_msg=f"B_a mismatch at {name}")

    def test_far_field_points(self):
        """Test far field points."""
        far_cases = ['far_diagonal', 'very_far_axis', 'very_far_diagonal', 'very_far_radial']
        for name in far_cases:
            r, a, expected_Br, expected_Ba = BFIELD_REGRESSION[name]
            B_r, B_a = ring_magnetic_field(r, a, R, I)
            np.testing.assert_allclose(B_r, expected_Br, rtol=1e-10, atol=1e-20,
                err_msg=f"B_r mismatch at {name}")
            np.testing.assert_allclose(B_a, expected_Ba, rtol=1e-10, atol=1e-20,
                err_msg=f"B_a mismatch at {name}")


class TestElectricFieldPolarity:
    """Test polarity (sign) of electric field components."""

    def test_positive_charge_above_ring_points_up(self):
        """E_a should be positive when a > 0 for positive charge."""
        a_values = [0.1, 0.5, 1.0, 2.0, 10.0]
        for a in a_values:
            E_r, E_a = ring_electric_field(0.0, a, R, Q)
            assert E_a > 0, f"E_a should be positive at a={a}"

    def test_positive_charge_below_ring_points_down(self):
        """E_a should be negative when a < 0 for positive charge."""
        a_values = [-0.1, -0.5, -1.0, -2.0]
        for a in a_values:
            E_r, E_a = ring_electric_field(0.0, a, R, Q)
            assert E_a < 0, f"E_a should be negative at a={a}"

    def test_negative_charge_reverses_polarity(self):
        """Negative charge should reverse field direction."""
        test_points = [(0.5, 0.5), (0.0, 1.0), (1.5, 0.3)]
        for r, a in test_points:
            E_r_pos, E_a_pos = ring_electric_field(r, a, R, Q)
            E_r_neg, E_a_neg = ring_electric_field(r, a, R, -Q)
            np.testing.assert_allclose(E_r_neg, -E_r_pos, rtol=1e-10)
            np.testing.assert_allclose(E_a_neg, -E_a_pos, rtol=1e-10)

    def test_radial_field_inside_ring_plane_points_inward(self):
        """E_r inside ring (r < R, a=0) should be negative (toward axis)."""
        r_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        for r in r_values:
            E_r, E_a = ring_electric_field(r, 0.0, R, Q)
            assert E_r < 0, f"E_r should be negative (inward) at r={r}, a=0"

    def test_radial_field_outside_ring_plane_points_outward(self):
        """E_r outside ring (r > R, a=0) should be positive (away from axis)."""
        r_values = [1.1, 1.5, 2.0, 5.0]
        for r in r_values:
            E_r, E_a = ring_electric_field(r, 0.0, R, Q)
            assert E_r > 0, f"E_r should be positive (outward) at r={r}, a=0"

    def test_axial_field_zero_in_ring_plane(self):
        """E_a should be zero in the ring plane (a=0)."""
        r_values = [0.0, 0.5, 0.9, 1.1, 2.0]
        for r in r_values:
            E_r, E_a = ring_electric_field(r, 0.0, R, Q)
            np.testing.assert_allclose(E_a, 0.0, atol=1e-15,
                err_msg=f"E_a should be zero at r={r}, a=0")


class TestMagneticFieldPolarity:
    """Test polarity (sign) of magnetic field components."""

    def test_positive_current_at_center_points_up(self):
        """B_a at ring center should be positive for positive current."""
        B_r, B_a = ring_magnetic_field(0.0, 0.0, R, I)
        assert B_a > 0, "B_a should be positive at center for positive current"

    def test_negative_current_reverses_polarity(self):
        """Negative current should reverse field direction."""
        test_points = [(0.5, 0.5), (0.0, 1.0), (1.5, 0.3)]
        for r, a in test_points:
            B_r_pos, B_a_pos = ring_magnetic_field(r, a, R, I)
            B_r_neg, B_a_neg = ring_magnetic_field(r, a, R, -I)
            np.testing.assert_allclose(B_r_neg, -B_r_pos, rtol=1e-10)
            np.testing.assert_allclose(B_a_neg, -B_a_pos, rtol=1e-10)

    def test_radial_field_zero_on_axis(self):
        """B_r should be zero on axis (r=0)."""
        a_values = [0.0, 0.5, 1.0, 2.0, -0.5]
        for a in a_values:
            B_r, B_a = ring_magnetic_field(0.0, a, R, I)
            np.testing.assert_allclose(B_r, 0.0, atol=1e-20,
                err_msg=f"B_r should be zero on axis at a={a}")

    def test_radial_field_zero_in_ring_plane(self):
        """B_r should be zero in the ring plane (a=0)."""
        r_values = [0.1, 0.5, 0.9, 1.1, 2.0]
        for r in r_values:
            B_r, B_a = ring_magnetic_field(r, 0.0, R, I)
            np.testing.assert_allclose(B_r, 0.0, atol=1e-15,
                err_msg=f"B_r should be zero at r={r}, a=0")

    def test_axial_field_changes_sign_across_ring(self):
        """B_a changes sign from inside to outside ring in plane."""
        B_r_in, B_a_in = ring_magnetic_field(0.9, 0.0, R, I)
        B_r_out, B_a_out = ring_magnetic_field(1.1, 0.0, R, I)
        assert B_a_in > 0, "B_a should be positive inside ring"
        assert B_a_out < 0, "B_a should be negative outside ring"

    def test_radial_field_symmetric_above_below(self):
        """B_r(r, a) = -B_r(r, -a) (antisymmetric in a)."""
        test_points = [(0.5, 0.5), (0.8, 0.3), (1.5, 1.0)]
        for r, a in test_points:
            B_r_pos, B_a_pos = ring_magnetic_field(r, a, R, I)
            B_r_neg, B_a_neg = ring_magnetic_field(r, -a, R, I)
            np.testing.assert_allclose(B_r_pos, -B_r_neg, rtol=1e-10,
                err_msg=f"B_r antisymmetry failed at r={r}, a={a}")

    def test_axial_field_symmetric_above_below(self):
        """B_a(r, a) = B_a(r, -a) (symmetric in a)."""
        test_points = [(0.0, 0.5), (0.5, 0.5), (0.8, 0.3), (1.5, 1.0)]
        for r, a in test_points:
            B_r_pos, B_a_pos = ring_magnetic_field(r, a, R, I)
            B_r_neg, B_a_neg = ring_magnetic_field(r, -a, R, I)
            np.testing.assert_allclose(B_a_pos, B_a_neg, rtol=1e-10,
                err_msg=f"B_a symmetry failed at r={r}, a={a}")


class TestExactlyOnAxis:
    """Test behavior exactly on the axis (r=0)."""

    def test_efield_exactly_r_zero(self):
        """Electric field with r exactly zero."""
        a_values = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 10.0, -0.5, -1.0])
        E_r, E_a = ring_electric_field(np.zeros_like(a_values), a_values, R, Q)

        # E_r must be exactly zero
        np.testing.assert_array_equal(E_r, 0.0)

        # E_a at a=0 must be zero
        np.testing.assert_equal(E_a[0], 0.0)

        # All values must be finite
        assert np.all(np.isfinite(E_a))

    def test_bfield_exactly_r_zero(self):
        """Magnetic field with r exactly zero."""
        a_values = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 10.0, -0.5, -1.0])
        B_r, B_a = ring_magnetic_field(np.zeros_like(a_values), a_values, R, I)

        # B_r must be exactly zero
        np.testing.assert_array_equal(B_r, 0.0)

        # All values must be finite
        assert np.all(np.isfinite(B_a))

        # B_a must be positive everywhere on axis
        assert np.all(B_a > 0)


class TestCloseToRing:
    """Test behavior close to the ring itself."""

    def test_field_diverges_approaching_ring(self):
        """Field magnitude should increase approaching the ring."""
        # Approach the ring from inside
        distances = [0.1, 0.05, 0.02, 0.01]
        E_mags_inside = []
        B_mags_inside = []
        for d in distances:
            E_r, E_a = ring_electric_field(R - d, 0.0, R, Q)
            B_r, B_a = ring_magnetic_field(R - d, 0.0, R, I)
            E_mags_inside.append(np.sqrt(E_r**2 + E_a**2))
            B_mags_inside.append(np.sqrt(B_r**2 + B_a**2))

        # Field should increase as we get closer
        for i in range(len(E_mags_inside) - 1):
            assert E_mags_inside[i+1] > E_mags_inside[i], \
                "E magnitude should increase approaching ring from inside"
            assert B_mags_inside[i+1] > B_mags_inside[i], \
                "B magnitude should increase approaching ring from inside"

    def test_finite_at_reasonable_distance(self):
        """Field should be finite at reasonable distance from ring."""
        # Points near but not on the ring
        test_points = [
            (0.95, 0.05), (1.05, 0.05), (1.0, 0.1),
            (0.9, 0.1), (1.1, 0.1), (1.0, 0.05),
        ]
        for r, a in test_points:
            E_r, E_a = ring_electric_field(r, a, R, Q)
            B_r, B_a = ring_magnetic_field(r, a, R, I)
            assert np.isfinite(E_r), f"E_r not finite at ({r}, {a})"
            assert np.isfinite(E_a), f"E_a not finite at ({r}, {a})"
            assert np.isfinite(B_r), f"B_r not finite at ({r}, {a})"
            assert np.isfinite(B_a), f"B_a not finite at ({r}, {a})"


class TestVeryFarField:
    """Test far-field behavior."""

    def test_efield_approaches_point_charge(self):
        """Far from ring, E should approach point charge field."""
        # At large distance, ring looks like point charge
        distances = [50.0, 100.0, 200.0]
        for d in distances:
            # On axis
            E_r, E_a = ring_electric_field(0.0, d, R, Q)
            E_point = K_E * Q / d**2
            np.testing.assert_allclose(E_a, E_point, rtol=1e-3,
                err_msg=f"E should approach point charge at d={d}")

            # Off axis (at 45 degrees)
            r = d / np.sqrt(2)
            a = d / np.sqrt(2)
            E_r, E_a = ring_electric_field(r, a, R, Q)
            E_total = np.sqrt(E_r**2 + E_a**2)
            E_point = K_E * Q / d**2
            np.testing.assert_allclose(E_total, E_point, rtol=1e-2,
                err_msg=f"E magnitude should approach point charge at d={d}")

    def test_bfield_approaches_dipole(self):
        """Far from ring, B on axis should approach dipole field."""
        # Magnetic dipole moment: m = I * pi * R^2
        m = I * np.pi * R**2

        distances = [50.0, 100.0]
        for d in distances:
            B_r, B_a = ring_magnetic_field(0.0, d, R, I)
            # Dipole on axis: B = (mu0/4pi) * 2m/z^3
            B_dipole = (MU_0 / (4 * np.pi)) * 2 * m / d**3
            np.testing.assert_allclose(B_a, B_dipole, rtol=1e-3,
                err_msg=f"B should approach dipole at d={d}")

    def test_field_decreases_with_distance(self):
        """Field magnitude should decrease with distance."""
        d1 = 10.0
        d2 = 20.0

        # Electric field
        E_r1, E_a1 = ring_electric_field(0.0, d1, R, Q)
        E_r2, E_a2 = ring_electric_field(0.0, d2, R, Q)
        assert abs(E_a2) < abs(E_a1), "E should decrease with distance"

        # Magnetic field
        B_r1, B_a1 = ring_magnetic_field(0.0, d1, R, I)
        B_r2, B_a2 = ring_magnetic_field(0.0, d2, R, I)
        assert abs(B_a2) < abs(B_a1), "B should decrease with distance"


class TestCartesianRegression:
    """Regression tests for Cartesian coordinate API."""

    def test_xyz_matches_cylindrical(self):
        """XYZ results should match cylindrical for points in XZ plane."""
        test_points = [
            (0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (1.5, 0.8), (0.3, -0.7),
        ]
        for r, a in test_points:
            # Cylindrical
            E_r, E_a = ring_electric_field(r, a, R, Q)
            B_r, B_a = ring_magnetic_field(r, a, R, I)

            # Cartesian (point on X axis, so x=r, y=0, z=a)
            Ex, Ey, Ez = ring_electric_field_xyz(r, 0.0, a, R, Q)
            Bx, By, Bz = ring_magnetic_field_xyz(r, 0.0, a, R, I)

            np.testing.assert_allclose(Ex, E_r, rtol=1e-10)
            np.testing.assert_allclose(Ey, 0.0, atol=1e-15)
            np.testing.assert_allclose(Ez, E_a, rtol=1e-10)

            np.testing.assert_allclose(Bx, B_r, rtol=1e-10)
            np.testing.assert_allclose(By, 0.0, atol=1e-15)
            np.testing.assert_allclose(Bz, B_a, rtol=1e-10)

    def test_rotational_symmetry(self):
        """Field at same r but different phi should have same magnitude."""
        r = 0.7
        a = 0.5
        angles = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]

        E_mags = []
        B_mags = []
        for phi in angles:
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            Ex, Ey, Ez = ring_electric_field_xyz(x, y, a, R, Q)
            Bx, By, Bz = ring_magnetic_field_xyz(x, y, a, R, I)
            E_mags.append(np.sqrt(Ex**2 + Ey**2 + Ez**2))
            B_mags.append(np.sqrt(Bx**2 + By**2 + Bz**2))

        # All magnitudes should be equal
        np.testing.assert_allclose(E_mags, E_mags[0], rtol=1e-10)
        np.testing.assert_allclose(B_mags, B_mags[0], rtol=1e-10)


def run_tests_without_pytest():
    """Run tests without pytest."""
    passed = 0
    failed = 0

    test_classes = [
        TestElectricFieldRegression,
        TestMagneticFieldRegression,
        TestElectricFieldPolarity,
        TestMagneticFieldPolarity,
        TestExactlyOnAxis,
        TestCloseToRing,
        TestVeryFarField,
        TestCartesianRegression,
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
        pytest.main([__file__, '-v'])
    else:
        print("Running tests without pytest...")
        success = run_tests_without_pytest()
        sys.exit(0 if success else 1)
