"""
Assembly class for combining multiple current elements.

Copyright (C) 2007-2026 Indrek Mandre <indrek(at)mare.ee>
Licensed under the MIT License.

This module provides the Assembly class for combining CurrentLoop and
CurrentSegment objects into configurations and computing total magnetic fields.
"""

import numpy as np
import matplotlib.pyplot as plt
from .visualization import plot_field_2d


class Assembly:
    """
    Container for multiple current elements with field superposition.

    The total magnetic field is computed by summing contributions from all
    elements (superposition principle).

    Examples
    --------
    >>> from mhfields import Assembly, CurrentLoop

    >>> # Create a Helmholtz coil pair
    >>> asm = Assembly()
    >>> asm.add(CurrentLoop([0, 0, -0.05], [0, 0, 1], 0.1, 1.0))
    >>> asm.add(CurrentLoop([0, 0, 0.05], [0, 0, 1], 0.1, 1.0))

    >>> # Compute field at a point
    >>> B = asm.bfield([0, 0, 0])

    >>> # Visualize
    >>> fig, ax = asm.slice_xz(y=0)
    """

    def __init__(self):
        self._elements = []

    def add(self, element):
        """
        Add a current element to the assembly.

        Parameters
        ----------
        element : CurrentLoop or CurrentSegment
            Element to add

        Returns
        -------
        self : Assembly
            For method chaining
        """
        self._elements.append(element)
        return self

    def __iter__(self):
        """Iterate over elements."""
        return iter(self._elements)

    def __len__(self):
        """Number of elements."""
        return len(self._elements)

    def bfield(self, points):
        """
        Compute total magnetic field (superposition of all elements).

        Parameters
        ----------
        points : array_like, shape (N, 3) or (3,)
            Evaluation points (m)

        Returns
        -------
        B : ndarray, shape (N, 3) or (3,)
            Total magnetic field vectors (T)
        """
        points = np.asarray(points, dtype=float)
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, 3)

        B_total = np.zeros_like(points)
        for element in self._elements:
            B_total += element.bfield(points)

        return B_total[0] if single_point else B_total

    def get_bounding_box(self):
        """
        Compute axis-aligned bounding box of all elements.

        Returns
        -------
        bbox : tuple
            (min_corner, max_corner) where each is ndarray of shape (3,)
        """
        if not self._elements:
            return (np.zeros(3), np.zeros(3))

        all_mins = []
        all_maxs = []
        for elem in self._elements:
            bb_min, bb_max = elem.get_bounding_box()
            all_mins.append(bb_min)
            all_maxs.append(bb_max)

        return (np.min(all_mins, axis=0), np.max(all_maxs, axis=0))

    def _compute_extent(self, extent, axes):
        """
        Compute plot extent from bounding box or user override.

        Parameters
        ----------
        extent : tuple or None
            User-specified ((min1, max1), (min2, max2)) or None for auto
        axes : tuple of int
            Which axes (0=x, 1=y, 2=z) are the free axes

        Returns
        -------
        ranges : tuple
            ((min1, max1), (min2, max2)) for the two free axes
        """
        if extent is not None:
            return extent

        bb_min, bb_max = self.get_bounding_box()

        # Expand by 50% beyond geometry
        size = bb_max - bb_min
        margin = 0.5 * size
        # Minimum margin if geometry is very flat
        margin = np.maximum(margin, 0.1 * np.max(size))

        ranges = []
        for ax in axes:
            ranges.append((bb_min[ax] - margin[ax], bb_max[ax] + margin[ax]))

        return tuple(ranges)

    def slice_xy(self, z=0.0, extent=None, **kwargs):
        """
        Visualize field in an XY plane slice.

        Parameters
        ----------
        z : float
            Z-coordinate of the slice plane
        extent : tuple of tuples, optional
            ((xmin, xmax), (ymin, ymax)). Auto-computed if None.
        **kwargs
            Passed to plot_field_2d (mode, colormap, arrow_scale, etc.)

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        ranges = self._compute_extent(extent, axes=(0, 1))
        x_range, y_range = ranges

        def field_func(x, y):
            # x, y are 2D grids
            shape = x.shape
            pts = np.column_stack([x.ravel(), y.ravel(),
                                   np.full(x.size, z)])
            B = self.bfield(pts)  # (N, 3)
            Bx = B[:, 0].reshape(shape)
            By = B[:, 1].reshape(shape)
            return Bx, By

        # Default labels
        kwargs.setdefault('xlabel', 'x (m)')
        kwargs.setdefault('ylabel', 'y (m)')
        kwargs.setdefault('title', f'B-field in XY plane at z = {z} m')

        return plot_field_2d(field_func, x_range, y_range, **kwargs)

    def slice_xz(self, y=0.0, extent=None, **kwargs):
        """
        Visualize field in an XZ plane slice.

        Parameters
        ----------
        y : float
            Y-coordinate of the slice plane
        extent : tuple of tuples, optional
            ((xmin, xmax), (zmin, zmax)). Auto-computed if None.
        **kwargs
            Passed to plot_field_2d

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        ranges = self._compute_extent(extent, axes=(0, 2))
        x_range, z_range = ranges

        def field_func(x, z):
            shape = x.shape
            pts = np.column_stack([x.ravel(),
                                   np.full(x.size, y),
                                   z.ravel()])
            B = self.bfield(pts)
            Bx = B[:, 0].reshape(shape)
            Bz = B[:, 2].reshape(shape)
            return Bx, Bz

        kwargs.setdefault('xlabel', 'x (m)')
        kwargs.setdefault('ylabel', 'z (m)')
        kwargs.setdefault('title', f'B-field in XZ plane at y = {y} m')

        return plot_field_2d(field_func, x_range, z_range, **kwargs)

    def slice_yz(self, x=0.0, extent=None, **kwargs):
        """
        Visualize field in a YZ plane slice.

        Parameters
        ----------
        x : float
            X-coordinate of the slice plane
        extent : tuple of tuples, optional
            ((ymin, ymax), (zmin, zmax)). Auto-computed if None.
        **kwargs
            Passed to plot_field_2d

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        ranges = self._compute_extent(extent, axes=(1, 2))
        y_range, z_range = ranges

        def field_func(y, z):
            shape = y.shape
            pts = np.column_stack([np.full(y.size, x),
                                   y.ravel(),
                                   z.ravel()])
            B = self.bfield(pts)
            By = B[:, 1].reshape(shape)
            Bz = B[:, 2].reshape(shape)
            return By, Bz

        kwargs.setdefault('xlabel', 'y (m)')
        kwargs.setdefault('ylabel', 'z (m)')
        kwargs.setdefault('title', f'B-field in YZ plane at x = {x} m')

        return plot_field_2d(field_func, y_range, z_range, **kwargs)

    def slice(self, plane='xz', offset=0.0, extent=None, **kwargs):
        """
        Generic slice visualization.

        Parameters
        ----------
        plane : str
            One of 'xy', 'xz', 'yz'
        offset : float
            Coordinate value for the fixed axis
        extent : tuple of tuples, optional
            Ranges for the two free axes
        **kwargs
            Passed to the specific slice method

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        plane = plane.lower()
        if plane == 'xy':
            return self.slice_xy(z=offset, extent=extent, **kwargs)
        elif plane == 'xz':
            return self.slice_xz(y=offset, extent=extent, **kwargs)
        elif plane == 'yz':
            return self.slice_yz(x=offset, extent=extent, **kwargs)
        else:
            raise ValueError(f"Unknown plane: {plane}. Use 'xy', 'xz', or 'yz'.")

    def draw_geometry(self, ax=None, **kwargs):
        """
        Draw the geometry of all elements on a 3D axis.

        Parameters
        ----------
        ax : matplotlib Axes3D, optional
            If None, create new 3D figure
        **kwargs
            Passed to plot functions (color, linewidth, etc.)

        Returns
        -------
        ax : matplotlib Axes3D
        """
        from mpl_toolkits.mplot3d import Axes3D
        from .elements import CurrentLoop, CurrentSegment

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        kwargs.setdefault('color', 'blue')
        kwargs.setdefault('linewidth', 1.5)

        for elem in self._elements:
            pts = elem.get_geometry()
            if isinstance(elem, CurrentLoop):
                # Close the loop
                pts = np.vstack([pts, pts[0]])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], **kwargs)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        # Set equal aspect ratio
        bb_min, bb_max = self.get_bounding_box()
        max_range = np.max(bb_max - bb_min) / 2
        mid = (bb_max + bb_min) / 2
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        return ax
