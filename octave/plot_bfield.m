## Copyright (C) 2007-2026 Indrek Mandre <indrek(at)mare.ee>
## Licensed under the MIT License.
##
## Visualize the magnetic field of a current loop.
## Generates a plot similar to bfield_contours.png with:
## - Colored intensity (log scale)
## - Black contour lines
## - White streamlines
##
## Usage: plot_bfield(radius, current, extent, output_file)
##
## Example:
##   plot_bfield(1.0, 1.0, 2.0, "bfield_octave.png")

function plot_bfield(radius, current, extent, output_file)
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
    current = 1.0;
  endif
  if nargin < 1
    radius = 1.0;
  endif

  R = radius;
  I = current;

  ## Create grid
  n_points = 200;
  r_vec = linspace(-extent, extent, n_points);
  a_vec = linspace(-extent, extent, n_points);
  [R_grid, A_grid] = meshgrid(r_vec, a_vec);

  ## Compute field (handle negative r by mirroring)
  R_abs = abs(R_grid);
  [Br, Ba] = loop_bfield(R, I, R_abs, A_grid);

  ## Flip Br sign for negative r
  Br(R_grid < 0) = -Br(R_grid < 0);

  ## Compute magnitude
  B_mag = sqrt(Br.^2 + Ba.^2);

  ## Cap values near the ring to avoid singularity dominating the colormap
  cap_dist = 0.10;
  [~, Ba_cap] = loop_bfield(R, I, R - cap_dist, 0);
  cap_value = abs(Ba_cap);
  B_mag(B_mag > cap_value) = NaN;

  ## Create figure
  fig = figure("visible", "off");
  set(fig, "paperunits", "inches");
  set(fig, "papersize", [10, 8]);
  set(fig, "paperposition", [0, 0, 10, 8]);

  ## Plot colored intensity (log scale)
  B_log = log10(B_mag);
  B_log(isinf(B_log)) = NaN;
  vmin = prctile(B_log(:)(~isnan(B_log(:))), 5);
  vmax = prctile(B_log(:)(~isnan(B_log(:))), 95);

  pcolor(R_grid, A_grid, B_log);
  shading interp;
  colormap(jet);
  caxis([vmin, vmax]);
  cb = colorbar();
  ylabel(cb, "|B| (T, log_{10} scale)");
  hold on;

  ## Add contour lines
  n_levels = 15;
  levels = linspace(vmin, vmax, n_levels);
  contour(R_grid, A_grid, B_log, levels, "linecolor", "k", "linewidth", 0.5);

  ## Add streamlines starting from the positive r axis (a=0, r>0)
  ## These will trace the field lines without trying to close loops
  try
    ## Start points along positive r axis, inside the coil
    r_starts = [0.1:0.1:0.9];
    a_starts = zeros(size(r_starts));

    ## Trace in both directions from each starting point
    h1 = streamline(R_grid, A_grid, Br, Ba, r_starts, a_starts);
    set(h1, "color", "w", "linewidth", 0.8);
    h2 = streamline(R_grid, A_grid, -Br, -Ba, r_starts, a_starts);
    set(h2, "color", "w", "linewidth", 0.8);

    ## Also start from negative r axis
    h3 = streamline(R_grid, A_grid, Br, Ba, -r_starts, a_starts);
    set(h3, "color", "w", "linewidth", 0.8);
    h4 = streamline(R_grid, A_grid, -Br, -Ba, -r_starts, a_starts);
    set(h4, "color", "w", "linewidth", 0.8);

    ## Add filled arrowheads along streamlines
    ## Get streamline data and place arrows at selected points
    arrow_length = 0.04;   ## Length of arrowhead
    arrow_width = 0.6;     ## Width factor (relative to length)
    arrow_pos = 0.3;       ## Position along streamline (0-1) to place arrow

    ## Helper function to draw filled arrowhead
    ## px, py = tip position, dx, dy = direction (normalized)
    function draw_arrow(px, py, dx, dy, len, wid)
      ## Triangle vertices: tip at (px,py), base perpendicular to direction
      ## Tip
      tx = px;
      ty = py;
      ## Base corners (perpendicular to direction, behind tip)
      bx1 = px - len*dx - wid*len*dy;
      by1 = py - len*dy + wid*len*dx;
      bx2 = px - len*dx + wid*len*dy;
      by2 = py - len*dy - wid*len*dx;
      fill([tx, bx1, bx2], [ty, by1, by2], "w", "edgecolor", "w");
    endfunction

    ## Get vertex data for positive-r forward streamlines
    xy_cells = stream2(R_grid, A_grid, Br, Ba, r_starts, a_starts);
    for i = 1:length(xy_cells)
      xy = xy_cells{i};
      if ~isempty(xy) && size(xy, 1) > 10
        ## Pick a point partway along the streamline
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

    ## Get vertex data for negative-r forward streamlines
    xy_cells = stream2(R_grid, A_grid, Br, Ba, -r_starts, a_starts);
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
           Br(1:skip:end, 1:skip:end), Ba(1:skip:end, 1:skip:end), ...
           "color", "w", "linewidth", 0.5, "autoscale", "on");
  end_try_catch

  hold off;

  ## Labels and title
  xlabel("r (m)");
  ylabel("a (m)");
  title(sprintf("Magnetic Field of Ring (R=%.1f m, I=%.1f A) - Contours", R, I));
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
