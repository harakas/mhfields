#!/usr/bin/env python3
"""
Magnetic field visualization of a solenoid (9 coaxial coils).

Copyright (C) 2007-2026 Indrek Mandre <indrek(at)mare.ee>
Licensed under the MIT License.

This script demonstrates the Assembly class by creating a solenoid-like
configuration of 9 current loops and visualizing the magnetic field.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mhfields import Assembly, CurrentLoop, plot_field_2d


if __name__ == '__main__':
    # Solenoid parameters
    R = 1.0           # Coil radius (m)
    I = 1.0           # Current (A)
    spacing = 0.4     # Spacing between coils (m)
    n_coils = 9       # Number of coils

    # Create assembly with 9 coils centered around z=0
    asm = Assembly()
    for i in range(n_coils):
        z = (i - (n_coils - 1) / 2) * spacing
        asm.add(CurrentLoop([0, 0, z], [0, 0, 1], R, I))

    print(f"Created solenoid with {len(asm)} coils")
    print(f"  Radius: {R} m")
    print(f"  Spacing: {spacing} m")
    print(f"  Z range: {-(n_coils-1)/2 * spacing} to {(n_coils-1)/2 * spacing} m")

    # Cap at field value 5cm from middle coil towards center
    cap_point = [R - 0.05, 0, 0]
    B_cap = asm.bfield(cap_point)
    cap_value = float(np.linalg.norm(B_cap))
    print(f"  Cap value at 5cm from coil: {cap_value:.6e} T")

    # Visualize in ZX plane at y=0 (z horizontal, x vertical)
    # Extend 1.2m beyond geometry edges
    margin = 1.2
    z_extent = (n_coils - 1) / 2 * spacing + margin
    z_range = (-z_extent, z_extent)
    x_range = (-(R + margin), R + margin)

    def field_func(z, x):
        """Field function with z horizontal, x vertical."""
        shape = z.shape
        pts = np.column_stack([x.ravel(), np.zeros(z.size), z.ravel()])
        B = asm.bfield(pts)
        Bz = B[:, 2].reshape(shape)
        Bx = B[:, 0].reshape(shape)
        return Bz, Bx

    print("\nGenerating solenoid_zx.png...")
    fig, ax = plot_field_2d(
        field_func,
        z_range,
        x_range,
        mode='log_contour_lines_streamlines',
        colormap='jet',
        cap_value=cap_value,
        xlabel='z (m)',
        ylabel='x (m)',
        title=f'Solenoid B-field ({n_coils} coils, R={R}m, spacing={spacing}m, I={I}A)',
        colorbar_label='|B| (T)',
        save_path='solenoid_zx.png',
        show=False
    )

    print("Done!")
