## Copyright (C) 2007-2026 Indrek Mandre <indrek(at)mare.ee>
## Licensed under the MIT License.
##
## Visualize the electric field of a ring of charge.
## Generates a plot similar to efield_contours.png with:
## - Colored intensity (log scale)
## - Black contour lines
## - White streamlines
##
## Usage: plot_efield(radius, charge, extent, output_file)
##
## Example:
##   plot_efield(1.0, 1e-9, 2.0, "efield_octave.png")

function plot_efield(radius, charge, extent, output_file)
  ## Suppress gnuplot warning
  warning("off", "Octave:gnuplot-graphics");

  ## Use qt graphics toolkit if available
  try
    graphics_toolkit("qt");
  catch
    ## Fall back to gnuplot
  end_try_catch
  if nargin < 4
    output_file = "";
  endif
  if nargin < 3
    extent = 2.0;
  endif
  if nargin < 2
    charge = 1e-9;
  endif
  if nargin < 1
    radius = 1.0;
  endif

  R = radius;
  Q = charge;

  ## Create grid
  n_points = 200;
  r_vec = linspace(-extent, extent, n_points);
  a_vec = linspace(-extent, extent, n_points);
  [R_grid, A_grid] = meshgrid(r_vec, a_vec);

  ## Compute field (handle negative r by mirroring)
  R_abs = abs(R_grid);
  [Er, Ea] = loop_efield(R, Q, R_abs, A_grid);

  ## Flip Er sign for negative r
  Er(R_grid < 0) = -Er(R_grid < 0);

  ## Compute magnitude
  E_mag = sqrt(Er.^2 + Ea.^2);

  ## Cap values near the ring to avoid singularity dominating the colormap
  cap_dist = 0.10;
  [Er_cap, Ea_cap] = loop_efield(R, Q, R - cap_dist, cap_dist);
  cap_value = sqrt(Er_cap^2 + Ea_cap^2);
  E_mag(E_mag > cap_value) = NaN;

  ## Create figure
  fig = figure("visible", "off");
  set(fig, "paperunits", "inches");
  set(fig, "papersize", [10, 8]);
  set(fig, "paperposition", [0, 0, 10, 8]);

  ## Plot colored intensity (log scale)
  E_log = log10(E_mag);
  E_log(isinf(E_log)) = NaN;
  vmin = prctile(E_log(:)(~isnan(E_log(:))), 5);
  vmax = prctile(E_log(:)(~isnan(E_log(:))), 95);

  pcolor(R_grid, A_grid, E_log);
  shading interp;
  colormap(jet);
  caxis([vmin, vmax]);
  cb = colorbar();
  ylabel(cb, "|E| (V/m, log_{10} scale)");
  hold on;

  ## Add contour lines
  n_levels = 15;
  levels = linspace(vmin, vmax, n_levels);
  contour(R_grid, A_grid, E_log, levels, "linecolor", "k", "linewidth", 0.5);

  ## Add streamlines starting from a circle at 1.5m radius
  try
    ## Start points along a circle (32 points, offset so r=0 doesn't happen)
    start_radius = 1.5;
    n_starts = 32;
    theta = linspace(0, 2*pi, n_starts+1)(1:end-1) + pi/n_starts;  ## Offset by half step
    r_starts = start_radius * cos(theta);
    a_starts = start_radius * sin(theta);

    ## Trace field lines in both directions
    h1 = streamline(R_grid, A_grid, Er, Ea, r_starts, a_starts);
    set(h1, "color", "w", "linewidth", 0.8);
    h2 = streamline(R_grid, A_grid, -Er, -Ea, r_starts, a_starts);
    set(h2, "color", "w", "linewidth", 0.8);

    ## Add filled arrowheads along streamlines
    arrow_length = 0.04;   ## Length of arrowhead
    arrow_width = 0.6;     ## Width factor (relative to length)
    arrow_pos = 0.3;       ## Position along streamline (0-1) to place arrow

    ## Helper function to draw filled arrowhead
    function draw_arrow(px, py, dx, dy, len, wid)
      tx = px;
      ty = py;
      bx1 = px - len*dx - wid*len*dy;
      by1 = py - len*dy + wid*len*dx;
      bx2 = px - len*dx + wid*len*dy;
      by2 = py - len*dy - wid*len*dx;
      fill([tx, bx1, bx2], [ty, by1, by2], "w", "edgecolor", "w");
    endfunction

    ## Get vertex data and add arrows for forward direction
    xy_cells = stream2(R_grid, A_grid, Er, Ea, r_starts, a_starts);
    for i = 1:length(xy_cells)
      xy = xy_cells{i};
      if ~isempty(xy) && size(xy, 1) > 10
        idx = max(1, round(size(xy, 1) * arrow_pos));
        if idx < size(xy, 1)
          px = xy(idx, 1);
          py = xy(idx, 2);
          dx = xy(idx+1, 1) - xy(idx, 1);
          dy = xy(idx+1, 2) - xy(idx, 2);
          mag = sqrt(dx^2 + dy^2);
          if mag > 0
            draw_arrow(px, py, dx/mag, dy/mag, arrow_length, arrow_width);
          endif
        endif
      endif
    endfor

  catch err
    ## Fallback to quiver if streamline not available
    printf("Streamline error: %s\n", err.message);
    skip = 8;
    quiver(R_grid(1:skip:end, 1:skip:end), A_grid(1:skip:end, 1:skip:end), ...
           Er(1:skip:end, 1:skip:end), Ea(1:skip:end, 1:skip:end), ...
           "color", "w", "linewidth", 0.5, "autoscale", "on");
  end_try_catch

  hold off;

  ## Labels and title
  xlabel("r (m)");
  ylabel("a (m)");
  title(sprintf("Electric Field of Ring (R=%.1f m, Q=%.1e C) - Contours", R, Q));
  axis equal;
  xlim([-extent, extent]);
  ylim([-extent, extent]);

  ## Save or display
  if ~isempty(output_file)
    print(fig, output_file, "-dpng", "-r150");
    printf("Saved to %s\n", output_file);
  else
    set(fig, "visible", "on");
  endif
endfunction
