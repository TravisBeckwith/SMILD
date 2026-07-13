"""
smild/twin.py — Degenerate-twin construction for the Standard Model.

The Standard Model two-branch degeneracy (Jelescu et al. 2016; Novikov et al.
2018) is the existence of two distinct parameter sets that produce nearly
identical multi-shell single-encoding signals. In the single-fiber / powder-
averaged regime, the degenerate partner of (f, Da, De_par, De_perp) is the
parameter set that:

  1. Preserves the same intra-neurite fraction f and radial diffusivity De_perp.
  2. Preserves the moment-accessible quantities:
       M1 = f·Da + (1-f)·De_par          (mixture mean axial diffusivity)
       V  = f·(1-f)·(Da - De_par)²       (across-compartment axial variance)
  3. Flips the sign of the intra-/extra-axonal axial-diffusivity gap.

The twin construction gives the second root of the quadratic that shares
(M1, V) with the ground truth. This is the analytic heart of the degeneracy:
the two-branch solutions are the pair {(Da, De_par), (Da_twin, De_par_twin)}
that cannot be distinguished from (M1, V) alone — and M1 and V are (to leading
order) the only axial-diffusivity moments accessible from the powder signal at
finite b-values.

Reference
---------
Fieremans E, Jensen JH, Helpern JA. White matter characterization with
diffusional kurtosis imaging. NeuroImage. 2011;58(1):177-188.

Novikov DS, Veraart J, Jelescu IO, Fieremans E. Rotationally-invariant
mapping of scalar and orientational metrics of neuronal microstructure with
diffusion MRI. NeuroImage. 2018;174:518-538. (Eq. 5.3 and Appendix F.)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TwinResult:
    """Result of degenerate_twin()."""
    Da_twin: float
    De_par_twin: float
    M1: float          # preserved mixture mean axial diffusivity
    V: float           # preserved across-compartment axial variance
    gap: float         # |Da - De_par| = |Da_twin - De_par_twin| (same magnitude)


def degenerate_twin(f: float, Da: float, De_par: float) -> TwinResult:
    """
    Construct the degenerate twin of a Standard-Model axial parameter pair.

    Given the intra-neurite fraction f and axial diffusivities (Da, De_par),
    returns the twin (Da_twin, De_par_twin) that preserves both cumulant-
    accessible moments while flipping the sign of the axial gap.

    The twin is derived as follows. Write the axial diffusivities in terms of
    the mixture mean M1 and the gap g = Da - De_par:

        Da     = M1 + (1-f)·g
        De_par = M1 - f·g

    The twin preserves M1 and |g| but flips the sign of g:

        Da_twin     = M1 - (1-f)·g  = De_par
        De_par_twin = M1 + f·g      = Da   ... if f = 0.5 exactly, else:

    For general f, the twin is NOT a simple swap of Da and De_par. It is the
    specific reflection that preserves the fraction-weighted mean while
    exchanging which compartment is "fast". Only at f = 0.5 does it reduce to
    a literal swap.

    Parameters
    ----------
    f : float
        Intra-neurite signal fraction ∈ (0, 1).
    Da : float
        Intra-neurite axial diffusivity (µm²/ms).
    De_par : float
        Extra-neurite axial diffusivity (µm²/ms).

    Returns
    -------
    TwinResult
        Dataclass with Da_twin, De_par_twin, M1, V, gap.

    Notes
    -----
    The twin signal may differ from the original by a small but non-zero amount
    in the powder-averaged representation; this difference — normalized by noise
    — is the SMILD statistic. When the gap |Da - De_par| → 0, both the twin
    and the original converge, their signals become identical, and SMILD → 1
    (maximally degenerate).
    """
    if not (0.0 < f < 1.0):
        raise ValueError(f"f must be in (0, 1), got {f}")

    M1 = f * Da + (1.0 - f) * De_par
    g = Da - De_par
    V = f * (1.0 - f) * g**2

    Da_twin = M1 - (1.0 - f) * g
    De_par_twin = M1 + f * g

    return TwinResult(
        Da_twin=float(Da_twin),
        De_par_twin=float(De_par_twin),
        M1=float(M1),
        V=float(V),
        gap=float(abs(g)),
    )
