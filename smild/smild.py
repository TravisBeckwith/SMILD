"""
smild/smild.py — Core SMILD computation.

SMILD (Standard Model Inter-branch Likelihood Distance) is the noise-
normalized distance between a voxel's Standard Model signal and the signal of
its degenerate twin parameter set, transformed to [0, 1]:

    ΔR     = || M(θ) − M(θ_twin) ||_Σ     (Mahalanobis distance in signal space)
    SMILD  = exp(−ΔR² / 2)  ∈ [0, 1]

Interpretation:
    SMILD → 0   branches are far apart; data resolves the solution; estimate
                is trustworthy (low degeneracy, high practical identifiability).
    SMILD → 1   branches predict nearly identical signals; data cannot
                distinguish them; any reported microstructure value is
                arbitrary between the two solutions (high degeneracy).

This is Definition 1 from the SMILD theoretical framework document. See
smild/twin.py for the degenerate-twin construction and smild/forward.py for
the forward model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smild.forward import forward_powder
from smild.twin import TwinResult, degenerate_twin


@dataclass(frozen=True)
class SMILDResult:
    """Output of a single SMILD computation."""
    smild: float          # ∈ [0, 1]; 0 = identifiable, 1 = degenerate
    delta_R: float        # raw Mahalanobis branch separation (noise units)
    twin: TwinResult      # the degenerate twin parameter set
    S_original: np.ndarray   # original powder signal
    S_twin: np.ndarray        # twin powder signal


def smild(
    b: np.ndarray,
    f: float,
    Da: float,
    De_par: float,
    De_perp: float,
    sigma_S: np.ndarray | float,
) -> SMILDResult:
    """
    Compute SMILD for a single voxel from known Standard-Model parameters.

    This is the exact, ground-truth version used in validation experiments
    where the parameters are known. For real data, parameters must be estimated
    first (see smild_from_params).

    Parameters
    ----------
    b : array-like, shape (N,)
        b-values in ms/µm² (= b[s/mm²] / 1000).
    f : float
        Intra-neurite signal fraction ∈ (0, 1).
    Da : float
        Intra-neurite axial diffusivity (µm²/ms).
    De_par : float
        Extra-neurite axial diffusivity (µm²/ms).
    De_perp : float
        Extra-neurite radial diffusivity (µm²/ms).
    sigma_S : array-like, shape (N,) or float
        Per-shell powder-signal noise standard deviation. If scalar, broadcast
        to all shells. Typically: 1 / (b0_SNR * sqrt(n_directions)).

    Returns
    -------
    SMILDResult
        Dataclass containing smild, delta_R, twin, S_original, S_twin.

    Notes
    -----
    Polarity: SMILD high = degenerate = less trustworthy. The raw statistic
    delta_R increases with practical identifiability. Both are reported so
    downstream analyses can choose the appropriate sign convention.
    """
    b = np.asarray(b, dtype=float)
    sigma_S = np.broadcast_to(np.asarray(sigma_S, dtype=float), b.shape).copy()

    twin = degenerate_twin(f, Da, De_par)

    S_orig = forward_powder(b, f, Da, De_par, De_perp)
    S_twin = forward_powder(b, f, twin.Da_twin, twin.De_par_twin, De_perp)

    diff = S_orig - S_twin
    delta_R = float(np.sqrt(np.sum((diff / sigma_S) ** 2)))
    smild_val = float(np.exp(-0.5 * delta_R**2))

    return SMILDResult(
        smild=smild_val,
        delta_R=delta_R,
        twin=twin,
        S_original=S_orig,
        S_twin=S_twin,
    )


def smild_from_params(
    b: np.ndarray,
    params: dict,
    sigma_S: np.ndarray | float,
) -> SMILDResult:
    """
    Convenience wrapper: compute SMILD from a parameter dict.

    Parameters
    ----------
    b : array-like, shape (N,)
        b-values in ms/µm².
    params : dict
        Must contain keys: 'f', 'Da', 'De_par', 'De_perp'.
    sigma_S : array-like or float
        Per-shell noise std.

    Returns
    -------
    SMILDResult
    """
    return smild(
        b=b,
        f=params["f"],
        Da=params["Da"],
        De_par=params["De_par"],
        De_perp=params["De_perp"],
        sigma_S=sigma_S,
    )
