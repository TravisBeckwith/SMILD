# Verified two-branch WMTI-Watson / axially-symmetric DKI equations
# Source: Hansen/Jespersen-style axially symmetric DKI WMTI, arXiv:1610.02783 (doc [70]),
# consistent with Novikov et al. 2019 review (arXiv:1612.02059, doc [69], Eq. 3.10)
# and Fieremans et al. 2011 WMTI.

# Given voxel-level (or single-fiber, powder-corrected) axially symmetric diffusion+kurtosis:
#   D_par  = axial diffusivity (of the overall diffusion tensor)
#   D_perp = radial diffusivity
#   Dbar   = mean diffusivity  = (D_par + 2 D_perp)/3
#   Wbar   = mean kurtosis (appropriate invariant; here the axial/relevant W)
#   f      = intra-axonal (neurite) signal fraction
#
# The intra/extra split has a SIGN AMBIGUITY (the two branches), rooted in
# diffusivities-squared appearing in the kurtosis equation:
#
#   D_e_perp   = D_perp / (1 - f)                                          (b)
#
#   D_e_par(±) = D_par - (2/3)*(f/(1-f)) * ( D_perp ± sqrt( TERM ) )       (c)
#
#   D_a(∓)     = D_par - (2/3)      *      ( D_perp ∓ sqrt( TERM ) )       (d)
#
# where
#   TERM = (15(1-f)/(4 f)) * Dbar^2 * Wbar  -  5 * D_perp^2
#
# Branch 1: the "+" under-root choice for D_e_par (and "-" for D_a) => D_a <= D_e_par
# Branch 2: the opposite sign                                       => D_a >  D_e_par
#
# The two branches COINCIDE (degeneracy resolved by the data) when TERM -> 0,
# i.e. sqrt(TERM) -> 0. Then D_a and D_e_par converge. This is the analytic
# handle SMDI exploits: the separation between branches is governed by sqrt(TERM).
#
# NOTE for POC scope: f itself is estimated separately (or scanned); here we use
# the standard practice of taking f from the constraint / or treating the branch
# separation at the (D_par, D_perp, Wbar) level. For a clean POC we forward-simulate
# from known ground-truth SM params, add noise, recover both branches, and show
# SMDI separates degenerate from well-conditioned voxels.
