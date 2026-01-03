#!/usr/bin/env python3
"""
Electric field visualization of a ring of charge.

This script generates various visualizations of the electric field
produced by a uniformly charged ring.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mhfields import ring_electric_field, plot_field_2d

# Ring parameters
R = 1.0    # Ring radius (m)
Q = 1e-9   # Total charge (C)


def field_func(r, a):
    """Electric field function for plotting."""
    return ring_electric_field(r, a, R, Q)


if __name__ == '__main__':
    # Generate different visualization styles
    extent = 2.0

    # 1. Arrows only
    print("Generating efield_arrows.png...")
    plot_field_2d(
        field_func,
        r_range=(-extent, extent),
        a_range=(-extent, extent),
        mode='quiver',
        arrow_scale=0.4,
        title='Electric Field of Ring of Charge (Q = 1 nC, R = 1 m)',
        save_path='efield_arrows.png',
        show=False
    )

    # 2. Linear contour + quiver
    print("Generating efield_colored.png...")
    plot_field_2d(
        field_func,
        r_range=(-extent, extent),
        a_range=(-extent, extent),
        mode='contour_quiver',
        arrow_scale=0.4,
        colormap='jet',
        cap_factor=5.0,
        title='Electric Field of Ring of Charge',
        save_path='efield_colored.png',
        show=False
    )

    # 3. Log contour + quiver
    print("Generating efield_log.png...")
    plot_field_2d(
        field_func,
        r_range=(-extent, extent),
        a_range=(-extent, extent),
        mode='log_contour_quiver',
        arrow_scale=0.4,
        colormap='jet',
        title='Electric Field of Ring of Charge (log scale)',
        save_path='efield_log.png',
        show=False
    )

    print("Done!")
