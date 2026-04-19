# =============================================================================
# src/evaluation/significance_tests.py
#
# Runs Welch's t-tests and computes Cohen's d effect sizes to test whether
# differences in XAI metric scores between sentiment classes are statistically
# significant rather than due to chance (Section 6.2, Section 6.4).
#
# Welch's t-test (not Student's t-test) was chosen because the three sentiment
# classes have very different sample sizes in the test set (neutral n=120,
# positive n=55, negative n=25), which makes the equal-variance assumption
# of Student's t-test untenable (Section 6.2).
#
# Cohen's d quantifies practical effect magnitude independently of sample size.
# By convention: 0.2=small, 0.5=medium, 0.8=large (Cohen, 1988). The fidelity
# neutral vs negative comparison yields d=12.6 — among the largest effect sizes
# reportable, confirming the finding is practically substantial (Section 6.4.1).
#
# Tests run:
#   - Fidelity drop (Equation 6.5)  — all three class pairs
#   - Stability no-punct (Eq 6.6)   — all three class pairs
#   - Stability no-stopwords (Eq 6.6)— all three class pairs
#   - IG vs SHAP agreement (Eq 6.7) — all three class pairs
#   - XAI change: flip vs no-flip   — robustness interaction (Section 6.4.4)
#
# Output: reports/evaluation/significance_tests.txt
#
# Report: Section 6.2, Section 6.4.1–6.4.4, Appendix C.1
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class SigTestConfig:
    """Paths to the four evaluation CSVs produced by the individual eval scripts."""
    fidelity_csv: str = "reports/evaluation/fidelity_ig_results.csv"
    stability_csv: str = "reports/evaluation/stability_ig_results.csv"
    agreement_csv: str = "reports/evaluation/agreement_ig_shap.csv"
    robustness_csv: str = "reports/evaluation/robustness_xai.csv"
    out_dir: str = "reports/evaluation"
    out_txt: str = "significance_tests.txt"


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cohen's d for two independent samples using pooled standard deviation.
    Quantifies practical effect magnitude independently of sample size.
    Convention: 0.2=small, 0.5=medium, 0.8=large (Cohen, 1988, Section 6.2).
    Returns NaN if either group has fewer than 2 observations.
    """
    x = x.astype(float)
    y = y.astype(float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)


def welch_ttest(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Welch's t-test assuming unequal variances between groups.
    Chosen over Student's t-test because class sample sizes differ substantially
    (neutral n=120, positive n=55, negative n=25), making equal-variance
    assumption untenable (Section 6.2). Returns (t-statistic, p-value).
    """
    t, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return float(t), float(p)


def fmt_result(name: str, a_name: str, b_name: str,
               x: np.ndarray, y: np.ndarray) -> str:
    """Formats a single comparison result for the output report."""
    t, p = welch_ttest(x, y)
    d = cohens_d(x, y)
    return (
        f"{name}: {a_name} vs {b_name}\n"
        f"  n={len(x)} vs {len(y)}\n"
        f"  mean={np.mean(x):.4f} vs {np.mean(y):.4f}\n"
        f"  Welch t={t:.4f}, p={p:.4g}, Cohen's d={d:.4f}\n"
    )


def main():
    cfg = SigTestConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fid = pd.read_csv(cfg.fidelity_csv)
    stab = pd.read_csv(cfg.stability_csv)
    agr = pd.read_csv(cfg.agreement_csv)
    rob = pd.read_csv(cfg.robustness_csv)

    # Merge evaluation results by sentence index for aligned comparison
    df = fid.merge(
        stab[["idx", "stability_no_punct", "stability_no_stopwords", "orig_label"]],
        on="idx", how="left", suffixes=("", "_stab"),
    ).merge(
        agr[["idx", "topk_overlap"]],
        on="idx", how="left",
    )

    # All three pairwise class comparisons for each metric
    pairs = [("positive", "neutral"), ("negative", "neutral"), ("positive", "negative")]

    lines = []
    lines.append("STATISTICAL SIGNIFICANCE TESTING (Welch's t-test + Cohen's d)\n")
    lines.append(
        "Welch's t-test chosen over Student's t-test due to unequal class sizes "
        "(neutral n=120, positive n=55, negative n=25) — Section 6.2.\n"
        "Cohen's d: 0.2=small, 0.5=medium, 0.8=large (Cohen, 1988).\n"
    )

    # ------------------------------------------------------------------
    # Fidelity comparisons — drives Table results in Section 6.4.1
    # Key result: neutral vs negative d=12.6, neutral vs positive d=5.05
    # (among the largest effect sizes reportable by convention)
    # ------------------------------------------------------------------
    lines.append("\n=== Fidelity Drop (Equation 6.5, Section 6.4.1) ===\n")
    for a, b in pairs:
        x = df[df["orig_label"] == a]["fidelity_drop"].dropna().to_numpy()
        y = df[df["orig_label"] == b]["fidelity_drop"].dropna().to_numpy()
        lines.append(fmt_result("Fidelity", a, b, x, y))

    # ------------------------------------------------------------------
    # Stability comparisons — drives Section 6.4.2 analysis
    # Two perturbation types tested: punctuation removal and stopword removal
    # ------------------------------------------------------------------
    lines.append("\n=== Stability (Punctuation Removal, Equation 6.6, Section 6.4.2) ===\n")
    for a, b in pairs:
        x = df[df["orig_label"] == a]["stability_no_punct"].dropna().to_numpy()
        y = df[df["orig_label"] == b]["stability_no_punct"].dropna().to_numpy()
        lines.append(fmt_result("Stability_no_punct", a, b, x, y))

    lines.append("\n=== Stability (Stopword Removal, Equation 6.6, Section 6.4.2) ===\n")
    for a, b in pairs:
        x = df[df["orig_label"] == a]["stability_no_stopwords"].dropna().to_numpy()
        y = df[df["orig_label"] == b]["stability_no_stopwords"].dropna().to_numpy()
        lines.append(fmt_result("Stability_no_stopwords", a, b, x, y))

    # ------------------------------------------------------------------
    # IG vs SHAP agreement comparisons — drives Section 6.4.3 analysis
    # ~50% overlap overall; neutral consistently lowest (Section 6.4.3)
    # ------------------------------------------------------------------
    lines.append("\n=== IG vs SHAP Agreement (Equation 6.7, Section 6.4.3) ===\n")
    for a, b in pairs:
        x = df[df["orig_label"] == a]["topk_overlap"].dropna().to_numpy()
        y = df[df["orig_label"] == b]["topk_overlap"].dropna().to_numpy()
        lines.append(fmt_result("IG_SHAP_overlap", a, b, x, y))

    # ------------------------------------------------------------------
    # Robustness flip vs no-flip comparison — drives Section 6.4.4
    # Counterintuitive result: flip group shows LOWER XAI change (0.40)
    # than no-flip (0.61), Welch t=13.3, p=3.5e-29, d=0.96 (Section 6.4.4)
    # ------------------------------------------------------------------
    lines.append("\n=== Robustness: XAI Change by Label Flip (Section 6.4.4) ===\n")
    x0 = rob[rob["label_flip"] == 0]["xai_change"].dropna().to_numpy()
    x1 = rob[rob["label_flip"] == 1]["xai_change"].dropna().to_numpy()
    lines.append(fmt_result("XAI_change", "no_flip", "flip", x0, x1))

    # Save full report to text file (reproduced in Appendix C.1)
    out_path = out_dir / cfg.out_txt
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved statistical test report to: {out_path}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
    
