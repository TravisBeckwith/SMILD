"""
smild/forward.py — Standard Model powder-averaged forward model.

Implements the direction-averaged (spherical-mean / powder) signal of the
two-compartment Standard Model for a single coherent fiber population:

    Sbar(b)/S0 = f * A_stick(b, Da)
               + (1-f) * A_zeppelin(b, De_par, De_perp)

where:
    A_stick(b, Da)             = sqrt(pi) / (2 sqrt(b Da)) * erf(sqrt(b Da))
    A_zeppelin(b, Dp, Dt)      = exp(-b Dt) * sqrt(pi) / (2 sqrt(b (Dp-Dt)))
                                   * erf(sqrt(b (Dp-Dt)))

This is the standard spherical-mean form used by SMT (Kaden et al. 2016) and
is exact for the single-fiber kernel. It is the signal representation in which
the Standard Model two-branch degeneracy is most transparent: for fixed f and
De_perp, the signal is nearly invariant under swapping (Da, De_par) between
the two branch solutions.

Units: diffusivities in µm²/ms; b-values in ms/µm² (= b[s/mm²] / 1000).

References
----------
Kaden E, Kelm ND, Carson RP, Does MD, Alexander DC. Multi-compartment
microscopic diffusion imaging. NeuroImage. 2016;139:346-359.
doi:10.1016/j.neuroimage.2016.06.002.

Novikov DS, Fieremans E, Jespersen SN, Kiselev VG. Quantifying brain
microstructure with diffusion MRI: theory and parameter estimation.
NMR in Biomedicine. 2019;32(4):e3998. doi:10.1002/nbm.3998.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf


def powder_stick(b: np.ndarray, Da: float) -> np.ndarray:
    """
    Powder-averaged signal of an intra-neurite stick with axial diffusivity Da.

    Parameters
    ----------
    b : array-like, shape (N,)
        b-values in ms/µm².
    Da : float
        Intra-neurite axial diffusivity (µm²/ms). Must be > 0.

    Returns
    -------
    S : ndarray, shape (N,)
        Normalized powder signal (relative to S0 = 1).
    """
    b = np.asarray(b, dtype=float)
    out = np.ones_like(b)
    nz = b * Da > 1e-12
    x = np.sqrt(b[nz] * Da)
    out[nz] = (np.sqrt(np.pi) / 2.0) / x * erf(x)
    return out


def powder_zeppelin(b: np.ndarray, De_par: float, De_perp: float) -> np.ndarray:
    """
    Powder-averaged signal of an extra-neurite axially symmetric Gaussian
    (zeppelin) with axial diffusivity De_par and radial diffusivity De_perp.

    Parameters
    ----------
    b : array-like, shape (N,)
        b-values in ms/µm².
    De_par : float
        Extra-neurite axial diffusivity (µm²/ms).
    De_perp : float
        Extra-neurite radial (perpendicular) diffusivity (µm²/ms).

    Returns
    -------
    S : ndarray, shape (N,)
        Normalized powder signal.
    """
    b = np.asarray(b, dtype=float)
    delta = De_par - De_perp
    out = np.exp(-b * De_perp)
    nz = b * delta > 1e-12
    x = np.sqrt(b[nz] * delta)
    out[nz] *= (np.sqrt(np.pi) / 2.0) / x * erf(x)
    return out


def forward_powder(
    b: np.ndarray,
    f: float,
    Da: float,
    De_par: float,
    De_perp: float,
) -> np.ndarray:
    """
    Direction-averaged two-compartment Standard Model signal (normalized,
    S0 = 1) for a single coherent fiber population.

    Parameters
    ----------
    b : array-like, shape (N,)
        b-values in ms/µm².
    f : float
        Intra-neurite signal fraction, ∈ (0, 1).
    Da : float
        Intra-neurite axial diffusivity (µm²/ms).
    De_par : float
        Extra-neurite axial diffusivity (µm²/ms).
    De_perp : float
        Extra-neurite radial diffusivity (µm²/ms).

    Returns
    -------
    S : ndarray, shape (N,)
        Powder-averaged normalized signal.

    Notes
    -----
    The two-branch degeneracy of the Standard Model is the near-invariance of
    this signal under swapping (Da, De_par) between the two branch solutions
    (for fixed f and De_perp). The SMILD metric quantifies how distinguishable
    the two branches are given the noise level. See smild.twin and smild.smild.
    """
    b = np.asarray(b, dtype=float)
    if not (0.0 < f < 1.0):
        raise ValueError(f"f must be in (0, 1), got {f}")
    if Da <= 0 or De_par <= 0 or De_perp < 0:
        raise ValueError("Diffusivities must be positive (De_perp >= 0).")
    if De_perp >= De_par:
        raise ValueError("De_perp must be less than De_par for a zeppelin.")
    return f * powder_stick(b, Da) + (1.0 - f) * powder_zeppelin(b, De_par, De_perp)
