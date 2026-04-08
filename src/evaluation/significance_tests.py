from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class SigTestConfig:
    fidelity_csv: str = "reports/evaluation/fidelity_ig_results.csv"
    stability_csv: str = "reports/evaluation/stability_ig_results.csv"
    agreement_csv: str = "reports/evaluation/agreement_ig_shap.csv"
    robustness_csv: str = "reports/evaluation/robustness_xai.csv"

    out_dir: str = "reports/evaluation"
    out_txt: str = "significance_tests.txt"


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for two independent samples."""
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
    """Welch's t-test (unequal variances). Returns (t, p)."""
    t, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return float(t), float(p)


def fmt_result(name: str, a_name: str, b_name: str, x: np.ndarray, y: np.ndarray) -> str:
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

    # --- Load CSVs ---
    fid = pd.read_csv(cfg.fidelity_csv)
    stab = pd.read_csv(cfg.stability_csv)
    agr = pd.read_csv(cfg.agreement_csv)
    rob = pd.read_csv(cfg.robustness_csv)

    # --- Merge where helpful ---
    # Keep per-row alignment by idx
    df = fid.merge(
        stab[["idx", "stability_no_punct", "stability_no_stopwords", "orig_label"]],
        on="idx",
        how="left",
        suffixes=("", "_stab"),
    ).merge(
        agr[["idx", "topk_overlap"]],
        on="idx",
        how="left",
    )

    # Robustness uses stopword perturbation; may reuse idx
    df_rob = rob.copy()

    # --- Define class groups ---
    classes = ["negative", "neutral", "positive"]
    pairs = [("positive", "neutral"), ("negative", "neutral"), ("positive", "negative")]

    lines = []
    lines.append("STATISTICAL SIGNIFICANCE TESTING (Welch's t-test)\n")
    lines.append("Note: Cohen's d effect size interpretation (rough): 0.2 small, 0.5 medium, 0.8 large.\n")

    # --- 1) Fidelity drop comparisons ---
    lines.append("\n=== Fidelity Drop (P(orig_class) - P(masked)) ===\n")
    for a, b in pairs:
        x = df[df["orig_label"] == a]["fidelity_drop"].dropna().to_numpy()
        y = df[df["orig_label"] == b]["fidelity_drop"].dropna().to_numpy()
        lines.append(fmt_result("Fidelity", a, b, x, y))

    # --- 2) Stability comparisons ---
    lines.append("\n=== Stability (No Punctuation) ===\n")
    for a, b in pairs:
        x = df[df["orig_label"] == a]["stability_no_punct"].dropna().to_numpy()
        y = df[df["orig_label"] == b]["stability_no_punct"].dropna().to_numpy()
        lines.append(fmt_result("Stability_no_punct", a, b, x, y))

    lines.append("\n=== Stability (No Stopwords) ===\n")
    for a, b in pairs:
        x = df[df["orig_label"] == a]["stability_no_stopwords"].dropna().to_numpy()
        y = df[df["orig_label"] == b]["stability_no_stopwords"].dropna().to_numpy()
        lines.append(fmt_result("Stability_no_stopwords", a, b, x, y))

    # --- 3) IG vs SHAP agreement comparisons ---
    lines.append("\n=== IG vs SHAP Agreement (Top-k overlap) ===\n")
    for a, b in pairs:
        x = df[df["orig_label"] == a]["topk_overlap"].dropna().to_numpy()
        y = df[df["orig_label"] == b]["topk_overlap"].dropna().to_numpy()
        lines.append(fmt_result("IG_SHAP_overlap", a, b, x, y))

    # --- 4) Robustness: XAI change (flip vs no flip) ---
    lines.append("\n=== Robustness Interaction: XAI Change (1 - overlap) ===\n")
    x0 = df_rob[df_rob["label_flip"] == 0]["xai_change"].dropna().to_numpy()
    x1 = df_rob[df_rob["label_flip"] == 1]["xai_change"].dropna().to_numpy()
    lines.append(fmt_result("XAI_change", "no_flip", "flip", x0, x1))

    # --- Save + print ---
    out_path = out_dir / cfg.out_txt
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved statistical test report to: {out_path}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
