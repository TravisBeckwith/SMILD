# SMILD v0.1.0 — First release

**Standard Model Inter-branch Likelihood Distance**  
A voxelwise measure of practical parameter identifiability for biophysical diffusion MRI.

---

## What this release contains

This first release establishes the theoretical foundation, a validated proof of concept, and a clean installable Python package.

### Python package (`smild/`)

Three modules, fully tested (18/18 tests pass):

| Module | Purpose |
|---|---|
| `smild/forward.py` | Standard Model powder-averaged forward model (stick + zeppelin, direction-averaged) |
| `smild/twin.py` | Analytic degenerate-twin construction — the moment-preserving parameter-swap that is the two-branch degeneracy |
| `smild/smild.py` | Core SMILD computation: noise-normalized Mahalanobis distance between branch signals, mapped to [0, 1] |

Install with:
```bash
pip install smild==0.1.0
# or from source:
pip install -e ".[dev]"
```

### Proof of concept (`poc/`)

Synthetic single-fiber Standard Model validation on ABCD-like acquisition (b = 500/1000/2000/3000 s/mm²). Three falsifiable predictions, all verified:

| Prediction | Spearman r | Status |
|---|---|---|
| SMILD decreases as ground-truth branch separation increases | −1.00 | ✅ PASS |
| SMILD → 1 as branches merge (separation → 0) | SMILD = 0.995 at min sep | ✅ PASS |
| SMILD decreases as SNR / information content rises | −1.00 | ✅ PASS |

### Methods document (`docs/`)

Full 13-page theoretical framework: knowledge base, formal SMILD definitions (Definition 1: branch-separation SMILD; Definition 2: posterior-dispersion SMILD), estimation procedure, five-experiment ABCD validation protocol, and 16 verified references with DOIs.

---

## Background

The Standard Model of white-matter diffusion (Novikov et al. 2019) unifies NODDI, WMTI, SMT, and related models as special cases of one signal kernel. Fitting this model to conventional multi-shell linear-encoding data is intrinsically degenerate: two distinct parameter sets explain the measured signal equally well (Jelescu et al. 2016; Novikov et al. 2018). Standard pipelines conceal this by fixing parameters, yielding a unique but potentially biased answer.

SMILD does not resolve the degeneracy — it measures and reports it. The output is a per-voxel reliability map: voxels where the two branches produce nearly identical signals (high SMILD) have microstructure estimates that are not meaningfully constrained by the data; voxels where the branches are distinguishable (low SMILD) have trustworthy estimates.

---

## What is not yet in this release

The production estimator for real multi-shell DWI data requires:
- Rotational-invariant estimation from directional DWI (mapping raw images to the invariant space where SMILD is computed)
- The LEMONADE / analytic two-branch solver for arbitrary orientation distributions (currently only the exact closed-form twin for the single-fiber case is implemented)
- dwiforge Stage 12a–12d integration

These are the active next development targets. This release establishes the mathematical foundation and confirms the quantity behaves as theory predicts before that engineering investment.

---

## How to cite

```bibtex
@software{Beckwith_SMILD_2026,
  author    = {Beckwith, Travis},
  title     = {{SMILD: Standard Model Inter-branch Likelihood Distance}},
  version   = {0.1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21346103},
  url       = {https://github.com/TravisBeckwith/SMILD}
}
```

Please also cite the foundational degeneracy papers:

- Jelescu et al. (2016) *NMR in Biomedicine* 29:33–47. doi:10.1002/nbm.3450
- Novikov et al. (2018) *NeuroImage* 174:518–538. doi:10.1016/j.neuroimage.2018.03.006
- Novikov et al. (2019) *NMR in Biomedicine* 32:e3998. doi:10.1002/nbm.3998
- Coelho et al. (2019) *Magnetic Resonance in Medicine* 82:395–410. doi:10.1002/mrm.27714

---

## Dependencies

- Python ≥ 3.10
- numpy ≥ 1.24
- scipy ≥ 1.10
- matplotlib ≥ 3.7 (POC only, not required for the core package)
