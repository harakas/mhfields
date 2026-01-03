## Copyright (C) 2008-2026 Indrek Mandre <indrek(at)mare.ee>
## For more information please see http://www.mare.ee/indrek/octave/
## Licensed under the MIT License.
##
## Compute the magnetic field of a current loop (ring of current).
##
## Usage: [Br, Ba] = loop_bfield(radius, current, r, a)
##
## Parameters:
##   radius  - radius of the current loop (m)
##   current - current through the loop (A)
##   r       - radial distance from axis (m), can be a vector/matrix
##   a       - axial distance from loop plane (m), can be a vector/matrix
##
## Returns:
##   Br - radial component of magnetic field (T)
##   Ba - axial component of magnetic field (T)
##
## Notes:
##   Uses complete elliptic integrals K(m) and E(m).
##   Requires the 'specfun' package for ellipke().
##   Includes singularity handling for small m values (near axis).

function [Br, Ba] = loop_bfield(radius, current, r, a)
  ## Physical constant: mu_0 / (2*pi) = 2e-7
  f1 = 2e-7 * current;
  R = radius;

  ## Compute elliptic integral parameter
  q = r.^2 + R^2 + a.^2 + 2.*r.*R;
  m = 4.*r.*R ./ q;

  ## Complete elliptic integrals
  [K, E] = ellipke(m);

  ## Field calculation
  sqrt_q = sqrt(q);
  denom = q - 4.*r.*R;  # This is (r-R)^2 + a^2

  Br = f1 .* a .* (E .* (q - 2.*r.*R) ./ denom - K) ./ (sqrt_q .* r);
  Ba = f1 .* (E .* (R^2 - r.^2 - a.^2) ./ denom + K) ./ sqrt_q;

  ## Handle singularity near axis (small m values)
  ## Use cutoff interpolation to avoid numerical instability
  CUTOFF = 5e-7;
  idx = find(m < CUTOFF);

  if ~isempty(idx)
    a_cut = a(idx);
    r_orig = r(idx);

    ## Find the radius where m = CUTOFF
    ## Solve: CUTOFF = 4*r*R / (r^2 + R^2 + a^2 + 2*r*R)
    ## => CUTOFF*r^2 + 2*R*(CUTOFF-2)*r + CUTOFF*(R^2 + a^2) = 0
    m_cut = repmat(CUTOFF, size(a_cut));
    r_cut = -(sqrt((4 - m_cut.^2) .* R^2 - 2*R*a_cut.*m_cut.^2 - a_cut.^2.*m_cut.^2) - 2*R) ./ m_cut;

    ## Compute field at cutoff radius
    q_cut = r_cut.^2 + R^2 + a_cut.^2 + 2.*r_cut.*R;
    m_cut = 4.*r_cut.*R ./ q_cut;
    [K_cut, E_cut] = ellipke(m_cut);
    sqrt_q_cut = sqrt(q_cut);
    denom_cut = q_cut - 4.*r_cut.*R;

    ## Linear interpolation from cutoff: Br(r) = (r/r_cut) * Br(r_cut)
    Br_cut = f1 .* a_cut .* (E_cut .* (q_cut - 2.*r_cut.*R) ./ denom_cut - K_cut) ./ (sqrt_q_cut .* r_cut);
    Br(idx) = (r_orig ./ r_cut) .* Br_cut;
  endif
endfunction
