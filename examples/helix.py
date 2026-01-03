#!/usr/bin/env python3
"""
Magnetic field visualization of a helical coil.

Copyright (C) 2007-2026 Indrek Mandre <indrek(at)mare.ee>
Licensed under the MIT License.

This script demonstrates the Assembly class by creating a helical coil
from current segments and visualizing the magnetic field.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from mhfields import Assembly, CurrentSegment, plot_field_2d


if __name__ == '__main__':
    # Helix parameters
    R = 1.0              # Helix radius (m)
    I = 1.0              # Current (A)
    pitch = 0.4          # Axial distance per turn (m)
    n_turns = 9          # Number of turns
    segments_per_turn = 32

    # Total number of segments
    n_segments = n_turns * segments_per_turn

    # Create helix from segments
    asm = Assembly()

    # Helix centered around z=0
    z_start = -n_turns / 2 * pitch

    for i in range(n_segments):
        # Angle for this segment
        theta0 = 2 * np.pi * i / segments_per_turn
        theta1 = 2 * np.pi * (i + 1) / segments_per_turn

        # Z position (advances with each segment)
        z0 = z_start + pitch * i / segments_per_turn
        z1 = z_start + pitch * (i + 1) / segments_per_turn

        # Start and end points
        p0 = [R * np.cos(theta0), R * np.sin(theta0), z0]
        p1 = [R * np.cos(theta1), R * np.sin(theta1), z1]

        asm.add(CurrentSegment(p0, p1, current=I))

    print(f"Created helical coil with {len(asm)} segments")
    print(f"  Radius: {R} m")
    print(f"  Pitch: {pitch} m")
    print(f"  Turns: {n_turns}")
    print(f"  Segments per turn: {segments_per_turn}")
    print(f"  Z range: {z_start:.2f} to {-z_start:.2f} m")

    # Cap at field value 5cm from wire towards center
    # Find z where helix crosses x=R, y=0 (at theta=0, 2pi, 4pi, ...)
    # Wire crosses at z = z_start + k*pitch for k = 0, 1, 2, ...
    # Find the crossing closest to z=0
    k_closest = round(-z_start / pitch)
    z_wire = z_start + k_closest * pitch
    cap_point = [R - 0.05, 0, z_wire]
    B_cap = asm.bfield(cap_point)
    cap_value = float(np.linalg.norm(B_cap))
    print(f"  Cap point: 5cm from wire at z={z_wire:.2f}m")
    print(f"  Cap value: {cap_value:.6e} T")

    # Visualize in ZX plane at y=0 (z horizontal, x vertical)
    # Extend 1.2m beyond geometry edges
    margin = 1.2
    z_extent = n_turns / 2 * pitch + margin
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

    print("\nGenerating helix_zx.png...")
    fig, ax = plot_field_2d(
        field_func,
        z_range,
        x_range,
        mode='log_contour_lines_streamlines',
        colormap='jet',
        cap_value=cap_value,
        xlabel='z (m)',
        ylabel='x (m)',
        title=f'Helical Coil B-field ({n_turns} turns, R={R}m, pitch={pitch}m, I={I}A)',
        colorbar_label='|B| (T)',
        save_path='helix_zx.png',
        show=False
    )

    print("Done!")
