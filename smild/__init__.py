"""
SMILD — Standard Model Inter-branch Likelihood Distance

A voxelwise measure of practical parameter identifiability for
biophysical diffusion MRI.

The Standard Model of white-matter diffusion is an intrinsically degenerate
inverse problem under conventional single-encoding multi-shell acquisition:
two distinct parameter sets explain the measured signal equally well (the
two "branches"). SMILD quantifies the noise-normalized distance between the
branches in signal space, providing a per-voxel reliability map for any
Standard-Model-derived microstructure estimate.

Quick start
-----------
>>> from smild import forward_powder, degenerate_twin, smild
>>> import numpy as np
>>> b = np.array([0.5, 1.0, 2.0, 3.0])   # ms/um^2 (= b[s/mm^2] / 1000)
>>> S = forward_powder(b, f=0.5, Da=2.2, De_par=1.6, De_perp=0.6)
>>> sigma = np.full_like(b, 0.05)         # per-shell noise std
>>> score, delta_R, twin = smild(b, f=0.5, Da=2.2, De_par=1.6,
...                              De_perp=0.6, sigma_S=sigma)
>>> print(f"SMILD = {score:.3f}  (0=identifiable, 1=degenerate)")

References
----------
Jelescu IO, Veraart J, Fieremans E, Novikov DS. Degeneracy in model
parameter estimation for multi-compartmental diffusion in neuronal tissue.
NMR in Biomedicine. 2016;29(1):33-47. doi:10.1002/nbm.3450.

Novikov DS, Veraart J, Jelescu IO, Fieremans E. Rotationally-invariant
mapping of scalar and orientational metrics of neuronal microstructure
with diffusion MRI. NeuroImage. 2018;174:518-538.
doi:10.1016/j.neuroimage.2018.03.006.

Novikov DS, Fieremans E, Jespersen SN, Kiselev VG. Quantifying brain
microstructure with diffusion MRI: theory and parameter estimation.
NMR in Biomedicine. 2019;32(4):e3998. doi:10.1002/nbm.3998.
"""

from smild.forward import forward_powder, powder_stick, powder_zeppelin
from smild.twin import degenerate_twin
from smild.smild import smild, smild_from_params

__all__ = [
    "forward_powder",
    "powder_stick",
    "powder_zeppelin",
    "degenerate_twin",
    "smild",
    "smild_from_params",
]

__version__ = "0.1.0.dev0"
