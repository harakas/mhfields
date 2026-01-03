## Copyright (C) 2008-2026 Indrek Mandre <indrek(at)mare.ee>
## For more information please see http://www.mare.ee/indrek/octave/
## Licensed under the MIT License.
##
## Compute the 3D electric field of a ring of charge at arbitrary positions.
##
## Usage: E = loop_efield3d(center, normal, radius, charge, pos)
##
## Parameters:
##   center - 3D vector [x, y, z] of ring center (m)
##   normal - 3D unit normal vector of the ring plane
##   radius - radius of the ring (m)
##   charge - total charge on the ring (C)
##   pos    - Nx3 matrix where each row is a position [x, y, z] (m)
##
## Returns:
##   E - Nx3 matrix where each row is the field vector [Ex, Ey, Ez] (V/m)
##
## Example:
##   E = loop_efield3d([0 0 0], [0 0 1], 0.15, 1e-9, [0.2 0 0; 1 2 3])

function E = loop_efield3d(center, normal, radius, charge, pos)
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
  [Er, Ea] = loop_efield(radius, charge, r, a);

  ## Convert to Cartesian: E = Er * radial_hat + Ea * normal_hat
  E = radial_norm .* repmat(Er, 1, 3) + n_rep .* repmat(Ea, 1, 3);
endfunction
