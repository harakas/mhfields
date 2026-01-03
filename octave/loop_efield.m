## Copyright (C) 2008-2026 Indrek Mandre <indrek(at)mare.ee>
## For more information please see http://www.mare.ee/indrek/octave/
## Licensed under the MIT License.
##
## Compute the electric field of a ring of charge.
##
## Usage: [Er, Ea] = loop_efield(radius, charge, r, a)
##
## Parameters:
##   radius - radius of the ring (m)
##   charge - total charge on the ring (C)
##   r      - radial distance from axis (m), can be a vector/matrix
##   a      - axial distance from ring plane (m), can be a vector/matrix
##
## Returns:
##   Er - radial component of electric field (V/m)
##   Ea - axial component of electric field (V/m)
##
## Notes:
##   Uses complete elliptic integrals K(m) and E(m).
##   Requires the 'specfun' package for ellipke().
##   Includes singularity handling for small m values (near axis).

function [Er, Ea] = loop_efield(radius, charge, r, a)
  ## Physical constant: 1/(4*pi*epsilon_0) = c^2 * mu_0 / (4*pi) ≈ 8.9875e9
  k_e = 8.9875517923e9;  # Coulomb constant (N·m²/C²)
  R = radius;
  Q = charge;

  ## Prefactor: Q/(4*pi*epsilon_0) * 2/pi
  prefactor = k_e * Q * 2 / pi;

  ## Compute elliptic integral parameter
  q = r.^2 + R^2 + a.^2 + 2.*r.*R;
  m = 4.*r.*R ./ q;

  ## Complete elliptic integrals
  [K, E] = ellipke(m);

  ## Field calculation
  sqrt_q = sqrt(q);
  q32 = q .* sqrt_q;  # q^(3/2)
  one_minus_m = 1 - m;

  ## Common factor
  common = prefactor ./ (q32 .* one_minus_m);

  ## Radial component:
  ## Er = common * (1/m) * [2*R*K*(1-m) - E*(2*R - m*(r+R))]
  Er = common .* (1./m) .* (2.*R.*K.*one_minus_m - E.*(2.*R - m.*(r + R)));

  ## Axial component:
  ## Ea = common * a * E
  Ea = common .* a .* E;

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
    q32_cut = q_cut .* sqrt_q_cut;
    one_minus_m_cut = 1 - m_cut;

    common_cut = prefactor ./ (q32_cut .* one_minus_m_cut);

    ## Linear interpolation from cutoff: Er(r) = (r/r_cut) * Er(r_cut)
    Er_cut = common_cut .* (1./m_cut) .* (2.*R.*K_cut.*one_minus_m_cut - E_cut.*(2.*R - m_cut.*(r_cut + R)));
    Er(idx) = (r_orig ./ r_cut) .* Er_cut;

    ## Axial component at actual position (well-behaved)
    q_sm = r_orig.^2 + R^2 + a_cut.^2 + 2.*r_orig.*R;
    m_sm = max(4.*r_orig.*R ./ q_sm, 1e-15);
    [~, E_sm] = ellipke(m_sm);
    sqrt_q_sm = sqrt(q_sm);
    q32_sm = q_sm .* sqrt_q_sm;
    one_minus_m_sm = 1 - m_sm;
    common_sm = prefactor ./ (q32_sm .* one_minus_m_sm);
    Ea(idx) = common_sm .* a_cut .* E_sm;
  endif
endfunction
