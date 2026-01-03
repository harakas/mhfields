"""
mhfields - A Python library for calculating electromagnetic fields of rings.

Copyright (C) 2007-2026 Indrek Mandre <indrek(at)mare.ee>
Licensed under the MIT License.

This library provides functions for computing the off-axis electric field
of a ring of charge and the magnetic field of a ring of current, using
complete elliptic integrals.
"""

from .fields import (
    ring_electric_field,
    ring_magnetic_field,
    ring_electric_field_xyz,
    ring_magnetic_field_xyz,
)

from .visualization import plot_field_2d

__all__ = [
    'ring_electric_field',
    'ring_magnetic_field',
    'ring_electric_field_xyz',
    'ring_magnetic_field_xyz',
    'plot_field_2d',
]

__version__ = '0.1.0'
