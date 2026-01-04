"""
Current-carrying element classes for magnetic field calculations.

Copyright (C) 2007-2026 Indrek Mandre <indrek(at)mare.ee>
Licensed under the MIT License.

This module provides classes for current loops and segments that can be
combined into assemblies for computing total magnetic fields.
"""

import numpy as np
from .fields import ring_magnetic_field
from .segment import line_magnetic_field


class CurrentLoop:
    """
    A circular current loop at arbitrary position and orientation.

    The magnetic field is computed using the analytical elliptic integral
    formulas from the ring_magnetic_field function, with coordinate
    transformation for arbitrary orientation.

    Parameters
    ----------
    center : array_like, shape (3,)
        Center position of the loop (m)
    normal : array_like, shape (3,)
        Normal vector to the loop plane (will be normalized).
        Current flows counterclockwise when viewed from the +normal direction.
    radius : float
        Loop radius (m)
    current : float
        Current (A)
    viz_radius : float, optional
        Radius for visualization wire thickness (m). Default: radius/50

    Attributes
    ----------
    center : ndarray, shape (3,)
    normal : ndarray, shape (3,) - unit normal
    radius : float
    current : float
    viz_radius : float

    Examples
    --------
    >>> # Loop in XY plane at origin
    >>> loop = CurrentLoop([0, 0, 0], [0, 0, 1], 0.1, 1.0)
    >>> B = loop.bfield([0, 0, 0.1])  # Field on axis

    >>> # Tilted loop
    >>> loop = CurrentLoop([0, 0, 0.5], [1, 0, 1], 0.1, 1.0)
    >>> B = loop.bfield([[0, 0, 0], [0, 0, 1]])
    """

    def __init__(self, center, normal, radius, current, viz_radius=None):
        self.center = np.asarray(center, dtype=float)
        n = np.asarray(normal, dtype=float)
        self.normal = n / np.linalg.norm(n)
        self.radius = float(radius)
        self.current = float(current)
        self.viz_radius = viz_radius if viz_radius is not None else radius / 50

        # Precompute rotation matrix (from loop's local frame to world frame)
        self._rotation_matrix = self._compute_rotation_matrix()

    def _compute_rotation_matrix(self):
        """
        Compute rotation matrix from loop's local frame to world frame.

        Local frame: z-axis along normal, loop in xy-plane.
        """
        z = self.normal

        # Find a vector not parallel to z to cross with
        if abs(z[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])

        x = np.cross(z, ref)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        # Rotation matrix: columns are the new basis vectors
        R = np.column_stack([x, y, z])  # (3, 3)
        return R

    def bfield(self, points):
        """
        Compute magnetic field at given points.

        Parameters
        ----------
        points : array_like, shape (N, 3) or (3,)
            Evaluation points in world coordinates (m)

        Returns
        -------
        B : ndarray, shape (N, 3) or (3,)
            Magnetic field vectors (T)
        """
        points = np.asarray(points, dtype=float)
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, 3)

        # Transform to loop's local frame
        # 1. Translate: shift so loop center is at origin
        p_local = points - self.center  # (N, 3)

        # 2. Rotate: apply inverse rotation (R^T since R is orthogonal)
        p_local = p_local @ self._rotation_matrix  # (N, 3)
        # Now in local frame: z = axial, xy = radial plane

        # Extract cylindrical coordinates in local frame
        r = np.sqrt(p_local[:, 0]**2 + p_local[:, 1]**2)  # (N,)
        a = p_local[:, 2]  # (N,) - axial coordinate

        # Compute field in cylindrical coordinates using existing function
        B_r, B_a = ring_magnetic_field(r, a, self.radius, self.current)

        # Convert to local Cartesian
        # B_r points radially outward in xy-plane
        phi = np.arctan2(p_local[:, 1], p_local[:, 0])
        B_local_x = B_r * np.cos(phi)
        B_local_y = B_r * np.sin(phi)
        B_local_z = B_a

        B_local = np.column_stack([B_local_x, B_local_y, B_local_z])  # (N, 3)

        # Handle r=0 case (on axis): B_r = 0, direction undefined
        on_axis = r < 1e-15 * self.radius
        B_local[on_axis, 0] = 0.0
        B_local[on_axis, 1] = 0.0

        # Transform back to world frame
        B_world = B_local @ self._rotation_matrix.T  # (N, 3)

        return B_world[0] if single_point else B_world

    def get_geometry(self, n_points=100):
        """
        Return points for drawing the loop.

        Parameters
        ----------
        n_points : int
            Number of points around the circle

        Returns
        -------
        points : ndarray, shape (n_points, 3)
            Points along the loop circumference
        """
        theta = np.linspace(0, 2*np.pi, n_points)

        # Circle in local xy-plane
        local_pts = np.column_stack([
            self.radius * np.cos(theta),
            self.radius * np.sin(theta),
            np.zeros(n_points)
        ])

        # Transform to world
        world_pts = local_pts @ self._rotation_matrix.T + self.center
        return world_pts

    def get_bounding_box(self):
        """
        Return axis-aligned bounding box.

        Returns
        -------
        bbox : tuple
            (min_corner, max_corner) where each is shape (3,)
        """
        pts = self.get_geometry()
        return (pts.min(axis=0), pts.max(axis=0))


class CurrentSegment:
    """
    A finite straight current-carrying wire segment.

    Parameters
    ----------
    p0 : array_like, shape (3,)
        Start point (m)
    p1 : array_like, shape (3,), optional
        End point (m). Required if direction/length not given.
    direction : array_like, shape (3,), optional
        Direction vector (will be normalized). Used with length.
    length : float, optional
        Segment length (m). Used with direction.
    current : float
        Current (A), positive = flows from p0 toward p1
    viz_radius : float, optional
        Radius for visualization (m). Default: length/100

    Notes
    -----
    Either p1 OR (direction, length) must be provided, not both.

    Examples
    --------
    >>> # Segment using endpoints
    >>> seg = CurrentSegment([0, 0, -1], [0, 0, 1], current=1.0)

    >>> # Segment using direction and length
    >>> seg = CurrentSegment([0, 0, 0], direction=[0, 0, 1], length=2.0, current=1.0)
    """

    def __init__(self, p0, p1=None, *, direction=None, length=None,
                 current=1.0, viz_radius=None):
        self.p0 = np.asarray(p0, dtype=float)

        # Handle two construction modes
        if p1 is not None:
            if direction is not None or length is not None:
                raise ValueError("Provide either p1 OR (direction, length), not both")
            self.p1 = np.asarray(p1, dtype=float)
        elif direction is not None and length is not None:
            d = np.asarray(direction, dtype=float)
            d = d / np.linalg.norm(d)
            self.p1 = self.p0 + d * length
        else:
            raise ValueError("Must provide either p1 or (direction, length)")

        self.current = float(current)
        self._length = np.linalg.norm(self.p1 - self.p0)
        self.viz_radius = viz_radius if viz_radius is not None else self._length / 100

    @property
    def length(self):
        """Segment length (m)."""
        return self._length

    @property
    def direction(self):
        """Unit direction vector from p0 to p1."""
        if self._length > 0:
            return (self.p1 - self.p0) / self._length
        return np.zeros(3)

    def bfield(self, points):
        """
        Compute magnetic field at given points.

        Parameters
        ----------
        points : array_like, shape (N, 3) or (3,)
            Evaluation points (m)

        Returns
        -------
        B : ndarray, shape (N, 3) or (3,)
            Magnetic field vectors (T)
        """
        return line_magnetic_field(self.p0, self.p1, self.current, points)

    def get_geometry(self):
        """
        Return endpoints for drawing.

        Returns
        -------
        points : ndarray, shape (2, 3)
            Start and end points
        """
        return np.array([self.p0, self.p1])

    def get_bounding_box(self):
        """
        Return axis-aligned bounding box.

        Returns
        -------
        bbox : tuple
            (min_corner, max_corner) where each is shape (3,)
        """
        pts = self.get_geometry()
        return (pts.min(axis=0), pts.max(axis=0))
