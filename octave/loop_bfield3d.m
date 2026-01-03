## Copyright (C) 2008-2026 Indrek Mandre <indrek(at)mare.ee>
## For more information please see http://www.mare.ee/indrek/octave/
## Licensed under the MIT License.
##
## Compute the 3D magnetic field of a current loop at arbitrary positions.
##
## Usage: B = loop_bfield3d(center, normal, radius, current, pos)
##
## Parameters:
##   center  - 3D vector [x, y, z] of loop center (m)
##   normal  - 3D unit normal vector of the loop plane
##   radius  - radius of the current loop (m)
##   current - current through the loop (A)
##   pos     - Nx3 matrix where each row is a position [x, y, z] (m)
##
## Returns:
##   B - Nx3 matrix where each row is the field vector [Bx, By, Bz] (T)
##
## Example:
##   B = loop_bfield3d([0 0 0], [1 0 0], 0.15, 1, [0.2 0 0; 1 2 3])

function B = loop_bfield3d(center, normal, radius, current, pos)
  n_points = size(pos, 1);

  ## Vector from loop center to each point
  v = pos - repmat(center, n_points, 1);

  ## Replicate normal vector for vectorized operations
  n_rep = repmat(normal, n_points, 1);

  ## Axial distance (projection onto normal)
  a = dot(v, n_rep, 2);

  ## Radial vector (perpendicular to normal)
  radial = v - n_rep .* repmat(a, 1, 3);

  ## Radial distance
  r = sqrt(dot(radial, radial, 2));

  ## Normalized radial direction (handle r=0 case)
  radial_norm = radial ./ repmat(r, 1, 3);
  idx_zero = find(r == 0);
  if ~isempty(idx_zero)
    radial_norm(idx_zero, :) = zeros(length(idx_zero), 3);
  endif

  ## Compute field components in cylindrical coordinates
  [Br, Ba] = loop_bfield(radius, current, r, a);

  ## Convert to Cartesian: B = Br * radial_hat + Ba * normal_hat
  B = radial_norm .* repmat(Br, 1, 3) + n_rep .* repmat(Ba, 1, 3);
endfunction
