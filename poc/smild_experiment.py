#!/usr/bin/env python3
"""
smdi_experiment.py -- Validation experiments for the SMDI proof of concept.

Runs the three falsifiable predictions from the methods document that are
testable on synthetic data with known ground truth:

  E1 (specificity): SMDI is LOW where the two branches are truly far apart
      (well-conditioned) and HIGH where they nearly coincide (degenerate).
      We sweep the ground-truth separation |Da - De_par| from large to ~0.

  E2 (SNR dependence): SMDI increases as SNR decreases -- degeneracy is partly
      a function of information content.

  E3 (branch-merging point): as the ground-truth split -> 0, SMDI -> 1 and the
      two recovered branches converge (Da_branch1 - Da_branch2 -> 0).

Outputs a figure (smdi_validation.png) and a results table (smdi_results.csv).
"""
from __future__ import annotations
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smdi_poc import process_voxel, degenerate_twin, forward_powder, RNG

# ABCD-like multi-shell b-values in ms/um^2  (b=1000/2000/3000 s/mm^2).
# Include a low shell for cumulant stability.
B_SHELLS = np.array([0.5, 1.0, 2.0, 3.0])


# Realistic single-direction b0 SNR for ABCD-quality 1.7mm data, and a modest
# effective direction count (high-b shells, where the branch information lives,
# have far fewer usable directions and lower SNR than the nominal count).
SNR_B0 = 20
N_DIRS_EFF = 6


def experiment_specificity(snr=SNR_B0, n_noise=300, n_dirs=N_DIRS_EFF):
    """E1 + E3: sweep ground-truth axial-diffusivity separation."""
    # Hold the mixture mean roughly fixed; vary the split between Da and De_par.
    f = 0.5
    De_perp = 0.6
    mean_axial = 1.9  # um^2/ms, plausible WM mean axial diffusivity
    # separation from large (well-conditioned) to near-zero (degenerate)
    seps = np.linspace(1.4, 0.02, 14)
    rows = []
    for s in seps:
        # symmetric split around the mean for f=0.5
        Da = mean_axial + 0.5 * s
        De_par = mean_axial - 0.5 * s
        r = process_voxel(B_SHELLS, f=f, Da=Da, De_par=De_par,
                          De_perp=De_perp, snr=snr, n_noise=n_noise,
                          n_dirs_per_shell=n_dirs)
        rows.append(r)
        print(f"  sep(GT)={s:4.2f}  SMDI={r['smdi_mean']:.3f} "
              f"deltaR={r['delta_R_mean']:.2f}  n={r['n_valid']}")
    return rows


def experiment_snr(sep=0.7, n_noise=300, n_dirs=N_DIRS_EFF):
    """E2: fix a moderately-degenerate voxel, sweep SNR."""
    f = 0.5
    De_perp = 0.6
    mean_axial = 1.9
    Da = mean_axial + 0.5 * sep
    De_par = mean_axial - 0.5 * sep
    snrs = [8, 12, 16, 20, 30, 45, 65]
    rows = []
    for snr in snrs:
        r = process_voxel(B_SHELLS, f=f, Da=Da, De_par=De_par,
                          De_perp=De_perp, snr=snr, n_noise=n_noise,
                          n_dirs_per_shell=n_dirs)
        rows.append(r)
        print(f"  SNR={snr:3d}  SMDI={r['smdi_mean']:.3f} "
              f"deltaR={r['delta_R_mean']:.2f}")
    return rows


def make_figure(spec_rows, snr_rows):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # Panel A: SMDI vs ground-truth separation (specificity)
    ax = axes[0]
    gt = [r["ground_truth_sep"] for r in spec_rows]
    smdi = [r["smdi_mean"] for r in spec_rows]
    smdi_sd = [r["smdi_std"] for r in spec_rows]
    ax.errorbar(gt, smdi, yerr=smdi_sd, marker="o", color="#1F4E79",
                capsize=3, lw=2, ms=6)
    ax.set_xlabel("Ground-truth branch separation  |Da \u2212 De\u2225|  (\u00b5m\u00b2/ms)")
    ax.set_ylabel("SMDI  (degeneracy index)")
    ax.set_title("E1 \u00b7 Specificity: SMDI tracks true conditioning", fontsize=11)
    ax.axhline(0.5, ls="--", color="grey", lw=1)
    ax.invert_xaxis()  # degenerate (small sep) on the right
    ax.grid(alpha=0.3)
    ax.text(0.03, 0.92, "degenerate \u2192", transform=ax.transAxes,
            color="#C00000", fontsize=10, ha="left")
    ax.text(0.97, 0.08, "\u2190 well-conditioned", transform=ax.transAxes,
            color="#1F7A1F", fontsize=10, ha="right")

    # Panel B: SMDI vs SNR
    ax = axes[1]
    snrs = [r["snr"] for r in snr_rows]
    smdi = [r["smdi_mean"] for r in snr_rows]
    smdi_sd = [r["smdi_std"] for r in snr_rows]
    ax.errorbar(snrs, smdi, yerr=smdi_sd, marker="s", color="#2E74B5",
                capsize=3, lw=2, ms=6)
    ax.set_xlabel("SNR (b0, per-direction)")
    ax.set_ylabel("SMDI  (degeneracy index)")
    ax.set_title("E2 \u00b7 SMDI decreases as SNR / information rises", fontsize=11)
    ax.grid(alpha=0.3)

    # Panel C: recovered branch separation vs ground truth (merging)
    ax = axes[2]
    gt = np.array([r["ground_truth_sep"] for r in spec_rows])
    rec = np.array([r["sep_mean"] for r in spec_rows])
    ax.plot([0, gt.max()], [0, gt.max()], ls="--", color="grey", lw=1,
            label="identity")
    ax.plot(gt, rec, marker="^", color="#1F4E79", lw=2, ms=6,
            label="recovered")
    ax.set_xlabel("Ground-truth separation (\u00b5m\u00b2/ms)")
    ax.set_ylabel("Recovered branch separation (\u00b5m\u00b2/ms)")
    ax.set_title("E3 \u00b7 Branches merge as separation \u2192 0", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("SMDI proof of concept \u2014 synthetic single-fiber Standard Model, "
                 "b = 1000/2000/3000 s/mm\u00b2", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig("smdi_validation.png", dpi=130, bbox_inches="tight")
    print("wrote smdi_validation.png")


def write_csv(spec_rows, snr_rows):
    with open("smdi_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "ground_truth_sep", "snr", "smdi_mean",
                    "smdi_std", "delta_R_mean", "recovered_sep", "n_valid"])
        for r in spec_rows:
            w.writerow(["specificity", f"{r['ground_truth_sep']:.4f}", r["snr"],
                        f"{r['smdi_mean']:.5f}", f"{r['smdi_std']:.5f}",
                        f"{r['delta_R_mean']:.4f}", f"{r['sep_mean']:.4f}",
                        r["n_valid"]])
        for r in snr_rows:
            w.writerow(["snr_sweep", f"{r['ground_truth_sep']:.4f}", r["snr"],
                        f"{r['smdi_mean']:.5f}", f"{r['smdi_std']:.5f}",
                        f"{r['delta_R_mean']:.4f}", f"{r['sep_mean']:.4f}",
                        r["n_valid"]])
    print("wrote smdi_results.csv")


def main():
    print("=" * 62)
    print("E1/E3  Specificity sweep (ground-truth separation, SNR=40)")
    print("=" * 62)
    spec_rows = experiment_specificity(snr=40)

    print("\n" + "=" * 62)
    print("E2  SNR sweep (moderately degenerate voxel, sep=0.6)")
    print("=" * 62)
    snr_rows = experiment_snr(sep=0.6)

    print()
    make_figure(spec_rows, snr_rows)
    write_csv(spec_rows, snr_rows)

    # --- quantitative pass/fail checks ---
    print("\n" + "=" * 62)
    print("PASS/FAIL CHECKS")
    print("=" * 62)
    from scipy.stats import spearmanr
    gt = np.array([r["ground_truth_sep"] for r in spec_rows])
    smdi = np.array([r["smdi_mean"] for r in spec_rows])
    # E1: SMDI should be monotonically DECREASING in ground-truth separation.
    # Use Spearman rank correlation (robust to the saturated 0/1 tails).
    corr = spearmanr(gt, smdi).correlation
    print(f"E1 specificity: corr(SMDI, GT separation) = {corr:+.3f} "
          f"(expect strongly negative)  -> {'PASS' if corr < -0.7 else 'FAIL'}")
    # E3: most-degenerate voxel SMDI should be high, most-separated low
    print(f"E3 merging: SMDI at smallest GT sep = {smdi[np.argmin(gt)]:.3f} "
          f"(expect ~1)  -> {'PASS' if smdi[np.argmin(gt)] > 0.5 else 'FAIL'}")
    print(f"           SMDI at largest GT sep  = {smdi[np.argmax(gt)]:.3f} "
          f"(expect ~0)  -> {'PASS' if smdi[np.argmax(gt)] < 0.1 else 'FAIL'}")
    # E2: SMDI should DECREASE with SNR
    snrs = np.array([r["snr"] for r in snr_rows])
    smdi_snr = np.array([r["smdi_mean"] for r in snr_rows])
    corr_snr = spearmanr(snrs, smdi_snr).correlation
    print(f"E2 SNR: corr(SMDI, SNR) = {corr_snr:+.3f} "
          f"(expect negative)  -> {'PASS' if corr_snr < -0.5 else 'FAIL'}")


if __name__ == "__main__":
    main()
