"""
mhfields - Maxwell-Heaviside Fields library.

Copyright (C) 2007-2026 Indrek Mandre <indrek(at)mare.ee>
Licensed under the MIT License.

This library provides functions and classes for computing electromagnetic
fields from current-carrying conductors:

- Ring fields using complete elliptic integrals
- Line segment fields using Biot-Savart law
- CurrentLoop and CurrentSegment element classes
- Assembly class for combining multiple elements
"""

from .fields import (
    ring_electric_field,
    ring_magnetic_field,
    ring_electric_field_xyz,
    ring_magnetic_field_xyz,
)

from .segment import line_magnetic_field

from .elements import CurrentLoop, CurrentSegment

from .assembly import Assembly

from .visualization import plot_field_2d

__all__ = [
    # Ring field functions
    'ring_electric_field',
    'ring_magnetic_field',
    'ring_electric_field_xyz',
    'ring_magnetic_field_xyz',
    # Line segment field
    'line_magnetic_field',
    # Element classes
    'CurrentLoop',
    'CurrentSegment',
    # Assembly
    'Assembly',
    # Visualization
    'plot_field_2d',
]

__version__ = '0.1.0'
