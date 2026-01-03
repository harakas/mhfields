"""
Visualization tools for electromagnetic field plots.

This module provides functions for creating 2D visualizations of electromagnetic
fields, including:
- Quiver plots (direction arrows)
- Contour plots (magnitude colormaps)
- Combined contour + quiver plots
- Logarithmic scale contour plots
- Streamline plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_field_2d(field_func, r_range, a_range,
                  mode='contour_quiver',
                  cap_factor=5.0,
                  cap_value=None,
                  arrow_scale=0.04,
                  colormap='viridis',
                  figsize=(10, 8),
                  title=None,
                  xlabel='r (m)',
                  ylabel='a (m)',
                  colorbar_label=None,
                  n_grid_fine=200,
                  n_grid_coarse=25,
                  save_path=None,
                  show=True):
    """
    Create a 2D visualization of an electromagnetic field.

    Parameters
    ----------
    field_func : callable
        Function that takes (r, a) arrays and returns (F_r, F_a) field components.
        Should handle numpy arrays.
    r_range : tuple
        (r_min, r_max) for the radial axis. Note: if r_min < 0, we plot
        the field mirrored (XZ plane cross-section).
    a_range : tuple
        (a_min, a_max) for the axial axis
    mode : str
        Visualization mode:
        - 'quiver': Direction arrows only
        - 'contour': Magnitude colormap only
        - 'contour_quiver': Colormap with arrows overlay
        - 'log_contour_quiver': Log-scale colormap with arrows
        - 'streamlines': Field line traces
    cap_factor : float
        Cap magnitude at cap_factor × reference_value. Values above are set to 0.
        Reference is computed at (R, 0) for typical ring fields.
    arrow_scale : float
        Scale factor for quiver arrows (default 0.4)
    colormap : str
        Matplotlib colormap name
    figsize : tuple
        Figure size in inches
    title : str, optional
        Plot title
    xlabel, ylabel : str
        Axis labels
    n_grid_fine : int
        Number of grid points for fine (color) mesh
    n_grid_coarse : int
        Number of grid points for coarse (arrow) mesh
    save_path : str, optional
        If provided, save figure to this path
    show : bool
        Whether to display the figure

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """
    r_min, r_max = r_range
    a_min, a_max = a_range

    # Create fine grid for magnitude/color plot
    r_fine = np.linspace(r_min, r_max, n_grid_fine)
    a_fine = np.linspace(a_min, a_max, n_grid_fine)
    R_fine, A_fine = np.meshgrid(r_fine, a_fine)

    # Create coarse grid for arrows
    r_coarse = np.linspace(r_min, r_max, n_grid_coarse)
    a_coarse = np.linspace(a_min, a_max, n_grid_coarse)
    R_coarse, A_coarse = np.meshgrid(r_coarse, a_coarse)

    # Handle negative r values (mirror for XZ plane visualization)
    def compute_field_with_mirror(R_grid, A_grid):
        """Compute field, handling negative r by mirroring."""
        R_abs = np.abs(R_grid)
        F_r, F_a = field_func(R_abs, A_grid)
        # For negative r, radial component flips sign
        F_r = np.where(R_grid < 0, -F_r, F_r)
        return F_r, F_a

    # Compute fields on fine grid
    Fr_fine, Fa_fine = compute_field_with_mirror(R_fine, A_fine)

    # Compute magnitude
    magnitude = np.sqrt(Fr_fine**2 + Fa_fine**2)

    # Apply capping
    if cap_value is not None:
        # Absolute cap value
        magnitude = np.where(magnitude > cap_value, 0, magnitude)
    elif cap_factor is not None and cap_factor > 0:
        # Find reference magnitude (at a typical point away from singularities)
        # Try the center of the grid, offset from axis
        r_ref = max(abs(r_max), abs(r_min)) * 0.5
        a_ref = (a_max + a_min) * 0.5
        if r_ref < 0.01:
            r_ref = 0.1
        Fr_ref, Fa_ref = field_func(np.array([r_ref]), np.array([a_ref]))
        ref_mag = np.sqrt(Fr_ref**2 + Fa_ref**2)[0]
        if ref_mag > 0:
            max_val = cap_factor * ref_mag
            magnitude = np.where(magnitude > max_val, 0, magnitude)

    # Compute fields on coarse grid for arrows
    Fr_coarse, Fa_coarse = compute_field_with_mirror(R_coarse, A_coarse)

    # Normalize arrows (unit vectors for direction only)
    mag_coarse = np.sqrt(Fr_coarse**2 + Fa_coarse**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        Fr_norm = np.where(mag_coarse > 0, Fr_coarse / mag_coarse, 0)
        Fa_norm = np.where(mag_coarse > 0, Fa_coarse / mag_coarse, 0)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    if mode == 'quiver':
        # Arrows only
        ax.quiver(R_coarse, A_coarse, Fr_norm, Fa_norm, scale=1/arrow_scale)
        ax.set_aspect('equal')

    elif mode == 'contour':
        # Colormap only
        levels = 50
        cf = ax.contourf(R_fine, A_fine, magnitude, levels=levels, cmap=colormap)
        plt.colorbar(cf, ax=ax, label=colorbar_label or 'Field magnitude')
        ax.set_aspect('equal')

    elif mode == 'contour_quiver':
        # Colormap with arrows overlay
        cf = ax.pcolormesh(R_fine, A_fine, magnitude, cmap=colormap, shading='auto')
        plt.colorbar(cf, ax=ax, label=colorbar_label or 'Field magnitude')
        ax.quiver(R_coarse, A_coarse, Fr_norm, Fa_norm, scale=1/arrow_scale, color='white', alpha=0.8)
        ax.set_aspect('equal')

    elif mode == 'log_contour_quiver':
        # Log-scale colormap with arrows
        # Handle zeros for log scale
        mag_log = np.where(magnitude > 0, magnitude, np.nan)
        vmin = np.nanpercentile(mag_log, 5)
        vmax = np.nanpercentile(mag_log, 95)
        if vmin <= 0:
            vmin = np.nanmin(mag_log[mag_log > 0]) if np.any(mag_log > 0) else 1e-10

        cf = ax.pcolormesh(R_fine, A_fine, magnitude, cmap=colormap,
                          norm=LogNorm(vmin=vmin, vmax=vmax), shading='auto')
        plt.colorbar(cf, ax=ax, label=colorbar_label or 'Field magnitude')
        ax.quiver(R_coarse, A_coarse, Fr_norm, Fa_norm, scale=1/arrow_scale, color='white', alpha=0.8)
        ax.set_aspect('equal')

    elif mode == 'streamlines':
        # Field line traces
        speed = np.sqrt(Fr_fine**2 + Fa_fine**2)
        lw = 2 * speed / speed.max() if speed.max() > 0 else 1
        ax.streamplot(r_fine, a_fine, Fr_fine, Fa_fine,
                     color=magnitude, cmap=colormap, linewidth=lw, density=1.5)
        ax.set_aspect('equal')

    elif mode == 'contour_streamlines':
        # Colormap with streamlines overlay
        cf = ax.pcolormesh(R_fine, A_fine, magnitude, cmap=colormap, shading='auto')
        plt.colorbar(cf, ax=ax, label=colorbar_label or 'Field magnitude')
        ax.streamplot(r_fine, a_fine, Fr_fine, Fa_fine,
                     color='white', linewidth=0.8, density=1.5, arrowsize=0.8)
        ax.set_aspect('equal')

    elif mode == 'log_contour_streamlines':
        # Log-scale colormap with streamlines
        mag_log = np.where(magnitude > 0, magnitude, np.nan)
        vmin = np.nanpercentile(mag_log, 5)
        vmax = np.nanpercentile(mag_log, 95)
        if vmin <= 0:
            vmin = np.nanmin(mag_log[mag_log > 0]) if np.any(mag_log > 0) else 1e-10

        cf = ax.pcolormesh(R_fine, A_fine, magnitude, cmap=colormap,
                          norm=LogNorm(vmin=vmin, vmax=vmax), shading='auto')
        plt.colorbar(cf, ax=ax, label=colorbar_label or 'Field magnitude')
        ax.streamplot(r_fine, a_fine, Fr_fine, Fa_fine,
                     color='white', linewidth=0.8, density=1.5, arrowsize=0.8)
        ax.set_aspect('equal')

    elif mode == 'log_contour_lines_streamlines':
        # Log-scale colormap with contour lines and streamlines
        mag_log = np.where(magnitude > 0, magnitude, np.nan)
        vmin = np.nanpercentile(mag_log, 5)
        vmax = np.nanpercentile(mag_log, 95)
        if vmin <= 0:
            vmin = np.nanmin(mag_log[mag_log > 0]) if np.any(mag_log > 0) else 1e-10

        # Colored intensity
        cf = ax.pcolormesh(R_fine, A_fine, magnitude, cmap=colormap,
                          norm=LogNorm(vmin=vmin, vmax=vmax), shading='auto')
        plt.colorbar(cf, ax=ax, label=colorbar_label or 'Field magnitude')
        # Black contour lines
        levels = np.logspace(np.log10(vmin), np.log10(vmax), 15)
        ax.contour(R_fine, A_fine, magnitude, levels=levels, colors='black', linewidths=0.5)
        # White streamlines
        ax.streamplot(r_fine, a_fine, Fr_fine, Fa_fine,
                     color='white', linewidth=0.8, density=1.5, arrowsize=0.8)
        ax.set_aspect('equal')

    else:
        raise ValueError(f"Unknown mode: {mode}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.set_xlim(r_min, r_max)
    ax.set_ylim(a_min, a_max)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_ring_cross_section(field_func, R_ring, extent=2.0, **kwargs):
    """
    Convenience function to plot a ring field cross-section.

    Shows the field in the XZ plane (r-a plane) with the ring at r=R, a=0.

    Parameters
    ----------
    field_func : callable
        Field function (r, a) -> (F_r, F_a)
    R_ring : float
        Ring radius, used to set default extent
    extent : float
        Plot extends from -extent*R to +extent*R in both directions
    **kwargs : dict
        Additional arguments passed to plot_field_2d

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """
    r_range = (-extent * R_ring, extent * R_ring)
    a_range = (-extent * R_ring, extent * R_ring)

    return plot_field_2d(field_func, r_range, a_range, **kwargs)
