#!/usr/bin/env python3
"""
Magnetic field visualization of a ring of current.

Copyright (C) 2007-2026 Indrek Mandre <indrek(at)mare.ee>
Licensed under the MIT License.

This script generates various visualizations of the magnetic field
produced by a current-carrying ring.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mhfields import ring_magnetic_field, plot_field_2d

# Ring parameters
R = 1.0   # Ring radius (m)
I = 1.0   # Current (A)


import numpy as np

def field_func(r, a):
    """Magnetic field function for plotting (handles negative r for XZ cross-section)."""
    B_r, B_a = ring_magnetic_field(np.abs(r), a, R, I)
    # Flip radial component for negative r (other side of axis)
    B_r = np.where(r < 0, -B_r, B_r)
    return B_r, B_a


if __name__ == '__main__':
    # Generate different visualization styles
    extent = 2.0

    # 1. Arrows only
    print("Generating bfield_arrows.png...")
    plot_field_2d(
        field_func,
        r_range=(-extent, extent),
        a_range=(-extent, extent),
        mode='quiver',
        arrow_scale=0.4,
        title='Magnetic Field of Ring of Current (I = 1 A, R = 1 m)',
        save_path='bfield_arrows.png',
        show=False
    )

    # 2. Linear contour + quiver
    print("Generating bfield_colored.png...")
    plot_field_2d(
        field_func,
        r_range=(-extent, extent),
        a_range=(-extent, extent),
        mode='contour_quiver',
        arrow_scale=0.4,
        colormap='jet',
        cap_factor=5.0,
        title='Magnetic Field of Ring of Current',
        save_path='bfield_colored.png',
        show=False
    )

    # 3. Log contour + quiver
    print("Generating bfield_log.png...")
    plot_field_2d(
        field_func,
        r_range=(-extent, extent),
        a_range=(-extent, extent),
        mode='log_contour_quiver',
        arrow_scale=0.4,
        colormap='jet',
        title='Magnetic Field of Ring of Current (log scale)',
        save_path='bfield_log.png',
        show=False
    )

    print("Done!")
