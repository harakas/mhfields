#!/usr/bin/env python3
"""
Generate all visualization outputs for the mhfields library.

Copyright (C) 2007-2026 Indrek Mandre <indrek(at)mare.ee>
Licensed under the MIT License.

This script generates all PNG variations for both electric and magnetic fields.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from mhfields import ring_electric_field, ring_magnetic_field, plot_field_2d

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ring parameters
R = 1.0    # Ring radius (m)
Q = 1e-9   # Total charge (C)
I = 1.0    # Current (A)

# Plot extent
EXTENT = 2.0


def generate_field_plots(field_func, prefix, title_base, colorbar_label, cap_value=None):
    """Generate all visualization variants for a field."""

    # 1. Linear scale + streamlines (capped)
    print(f"Generating {prefix}_linear.png...")
    plot_field_2d(
        field_func,
        r_range=(-EXTENT, EXTENT),
        a_range=(-EXTENT, EXTENT),
        mode='contour_streamlines',
        colormap='jet',
        cap_value=cap_value,
        title=f'{title_base} - Linear Scale',
        colorbar_label=colorbar_label,
        save_path=os.path.join(OUTPUT_DIR, f'{prefix}_linear.png'),
        show=False
    )

    # 2. Log scale + streamlines
    print(f"Generating {prefix}_log.png...")
    plot_field_2d(
        field_func,
        r_range=(-EXTENT, EXTENT),
        a_range=(-EXTENT, EXTENT),
        mode='log_contour_streamlines',
        colormap='jet',
        title=f'{title_base} - Log Scale',
        colorbar_label=colorbar_label,
        save_path=os.path.join(OUTPUT_DIR, f'{prefix}_log.png'),
        show=False
    )

    # 3. Log scale + contour lines + streamlines
    print(f"Generating {prefix}_contours.png...")
    plot_field_2d(
        field_func,
        r_range=(-EXTENT, EXTENT),
        a_range=(-EXTENT, EXTENT),
        mode='log_contour_lines_streamlines',
        colormap='jet',
        title=f'{title_base} - Contours',
        colorbar_label=colorbar_label,
        save_path=os.path.join(OUTPUT_DIR, f'{prefix}_contours.png'),
        show=False
    )


def efield_func(r, a):
    """Electric field with negative r handling for XZ cross-section."""
    E_r, E_a = ring_electric_field(np.abs(r), a, R, Q)
    E_r = np.where(r < 0, -E_r, E_r)
    return E_r, E_a


def bfield_func(r, a):
    """Magnetic field with negative r handling for XZ cross-section."""
    B_r, B_a = ring_magnetic_field(np.abs(r), a, R, I)
    B_r = np.where(r < 0, -B_r, B_r)
    return B_r, B_a


if __name__ == '__main__':
    # Cap at field value 5cm from ring (towards origin)
    CAP_DIST = 0.10

    print("=" * 50)
    print("Generating Electric Field Visualizations")
    print("=" * 50)

    Er_cap, Ea_cap = ring_electric_field(R - CAP_DIST, 0.0, R, Q)
    efield_cap = float(np.sqrt(Er_cap**2 + Ea_cap**2))
    print(f"E-field cap value: {efield_cap:.6e}")

    generate_field_plots(
        efield_func,
        'efield',
        f'Electric Field of Ring (R={R} m, Q={Q:.0e} C)',
        colorbar_label='|E| (V/m)',
        cap_value=efield_cap
    )

    print()
    print("=" * 50)
    print("Generating Magnetic Field Visualizations")
    print("=" * 50)

    Br_cap, Ba_cap = ring_magnetic_field(R - CAP_DIST, 0.0, R, I)
    bfield_cap = float(np.sqrt(Br_cap**2 + Ba_cap**2))
    print(f"B-field cap value: {bfield_cap:.6e}")

    generate_field_plots(
        bfield_func,
        'bfield',
        f'Magnetic Field of Ring (R={R} m, I={I} A)',
        colorbar_label='|B| (T)',
        cap_value=bfield_cap
    )

    print()
    print("=" * 50)
    print("All visualizations generated successfully!")
    print("=" * 50)
