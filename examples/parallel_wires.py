#!/usr/bin/env python3
"""
Magnetic field visualization of two parallel wires with antiparallel currents.

Copyright (C) 2007-2026 Indrek Mandre <indrek(at)mare.ee>
Licensed under the MIT License.

This script demonstrates the Assembly class with CurrentSegment objects,
showing the magnetic field from two parallel wires carrying current in
opposite directions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mhfields import Assembly, CurrentSegment


if __name__ == '__main__':
    # Wire parameters
    L = 2.0           # Wire length (m)
    d = 0.2           # Separation between wires (m)
    I = 1.0           # Current magnitude (A)

    # Create assembly with two parallel wires along z-axis
    # Wire 1 at x = -d/2, current +I (upward)
    # Wire 2 at x = +d/2, current -I (downward, i.e. antiparallel)
    asm = Assembly()
    asm.add(CurrentSegment([-d/2, 0, -L/2], [-d/2, 0, L/2], current=+I))
    asm.add(CurrentSegment([+d/2, 0, -L/2], [+d/2, 0, L/2], current=-I))

    print(f"Created parallel wires configuration")
    print(f"  Wire length: {L} m")
    print(f"  Separation: {d} m")
    print(f"  Currents: +{I}A and -{I}A (antiparallel)")

    # Visualize in XY plane at z=0 (perpendicular symmetric cut)
    extent = ((-0.5, 0.5), (-0.5, 0.5))

    print("\nGenerating parallel_wires_xy.png...")
    fig, ax = asm.slice_xy(
        z=0,
        extent=extent,
        mode='log_contour_lines_streamlines',
        colormap='jet',
        title=f'Antiparallel Wires B-field (d={d}m)',
        colorbar_label='|B| (T)',
        save_path='parallel_wires_xy.png',
        show=False
    )

    print("Done!")
