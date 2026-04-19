# =============================================================================
# src/evaluation/composite_score.py
#
# Computes the Explainability Reliability Score (ERS) — the novel composite
# metric that synthesises all four XAI quality dimensions into a single
# per-sentence reliability score (Equation 6.8, Section 6.2).
#
# Reads four input CSVs produced by the other evaluation scripts:
#   - fidelity_ig_results.csv       (from fidelity.py,            Equation 6.5)
#   - stability_ig_results.csv      (from stability.py,           Equation 6.6)
#   - agreement_ig_shap.csv         (from agreement_ig_shap.py,   Equation 6.7)
#   - robustness_xai.csv            (from robustness_xai_interaction.py)
#
# Output: reports/evaluation/explainability_reliability_score.csv
#   One row per test sentence with normalised component scores and final ERS.
#
# Key design decisions:
#   - Winsorized min-max normalisation clips each metric at its 1st-99th
#     percentile before scaling to [0,1], preventing outliers in any single
#     metric from compressing the rest of the distribution (Section 3.6.5)
#   - Equal weights (0.25 each) are used as the defensible default given the
#     absence of prior evidence for differential weighting (Section 6.2)
#   - Robustness is inverted (1 - xai_change) so that higher is always better
#     across all four components before combination
#   - Scripts are independently re-runnable — changing any parameter (e.g.
#     weights) requires re-running only this script (Section 3.6.5)
#
# Report: Section 6.2 (Equation 6.8), Section 6.4.5, Section 3.6.5
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ERSConfig:
    """
    Paths to input CSVs and output directory, plus normalisation and
    weighting parameters. Equal weights are the default (Section 6.2).
    Changing winsor_low/winsor_high adjusts outlier clipping thresholds.
    """
    fidelity_csv: str = "reports/evaluation/fidelity_ig_results.csv"
    stability_csv: str = "reports/evaluation/stability_ig_results.csv"
    agreement_csv: str = "reports/evaluation/agreement_ig_shap.csv"
    robustness_csv: str = "reports/evaluation/robustness_xai.csv"

    out_dir: str = "reports/evaluation"
    out_csv: str = "explainability_reliability_score.csv"

    # Winsorise at 1st and 99th percentile before min-max scaling (Section 3.6.5)
    winsor_low: float = 0.01
    winsor_high: float = 0.99

    # Equal weights — defensible given no prior evidence for differential
    # weighting in this context; stated as a limitation in Section 8.4
    w_fidelity: float = 0.25
    w_stability: float = 0.25
    w_agreement: float = 0.25
    w_robustness: float = 0.25


def winsor_minmax(x: pd.Series, low_q: float, high_q: float) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Robust min-max normalisation: clips x to [low_q, high_q] quantile range
    then scales to [0, 1]. Prevents a small number of extreme sentences from
    compressing the rest of the distribution into a narrow band (Section 3.6.5).

    Fidelity can legitimately be slightly negative for neutral-class predictions
    (masking top tokens occasionally increases neutral confidence) — winsorising
    handles this without requiring special-case logic.

    Returns the scaled Series and a dict with the clipping boundaries for
    diagnostic output.
    """
    xs = x.astype(float).copy()
    lo = float(xs.quantile(low_q))
    hi = float(xs.quantile(high_q))

    xs = xs.clip(lower=lo, upper=hi)

    # Degenerate case: all values identical — return zeros rather than divide by zero
    if np.isclose(hi, lo):
        return pd.Series(np.zeros(len(xs)), index=xs.index), {"lo": lo, "hi": hi}

    scaled = (xs - lo) / (hi - lo)
    return scaled, {"lo": lo, "hi": hi}


def main():
    cfg = ERSConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load the four component CSVs produced by the individual eval scripts
    # ------------------------------------------------------------------
    fid = pd.read_csv(cfg.fidelity_csv)
    stab = pd.read_csv(cfg.stability_csv)
    agr = pd.read_csv(cfg.agreement_csv)
    rob = pd.read_csv(cfg.robustness_csv)

    # Build merged DataFrame aligned on sentence index
    # Base is fidelity (contains sentence index, label, confidence)
    df = fid[["idx", "orig_label", "fidelity_drop", "orig_conf_max"]].rename(
        columns={"orig_conf_max": "confidence"}
    ).copy()

    df = df.merge(
        stab[["idx", "stability_no_stopwords", "stability_no_punct"]],
        on="idx",
        how="left",
    ).merge(
        agr[["idx", "topk_overlap"]],
        on="idx",
        how="left",
    ).merge(
        rob[["idx", "xai_change", "label_flip"]],
        on="idx",
        how="left",
    )

    # ------------------------------------------------------------------
    # Invert robustness so that higher is always better across all components.
    # xai_change = fraction of top-5 tokens that changed after perturbation,
    # so (1 - xai_change) = fraction that stayed the same = explanation stability
    # ------------------------------------------------------------------
    df["robust_expl"] = 1.0 - df["xai_change"].astype(float)

    # ------------------------------------------------------------------
    # Winsorized min-max normalisation — scales each component to [0, 1]
    # using the robust clipping approach described in Section 3.6.5
    # ------------------------------------------------------------------
    df["fidelity_norm"], fid_stats = winsor_minmax(df["fidelity_drop"], cfg.winsor_low, cfg.winsor_high)
    df["stability_norm"], stab_stats = winsor_minmax(df["stability_no_stopwords"], cfg.winsor_low, cfg.winsor_high)
    df["agreement_norm"], agr_stats = winsor_minmax(df["topk_overlap"], cfg.winsor_low, cfg.winsor_high)
    df["robustness_norm"], rob_stats = winsor_minmax(df["robust_expl"], cfg.winsor_low, cfg.winsor_high)

    # ------------------------------------------------------------------
    # Composite ERS = weighted sum of four normalised components (Equation 6.8)
    # Weights are auto-normalised if they do not sum to 1.0, so the formula
    # remains valid even if ERSConfig is modified experimentally
    # ------------------------------------------------------------------
    w_sum = cfg.w_fidelity + cfg.w_stability + cfg.w_agreement + cfg.w_robustness
    if not np.isclose(w_sum, 1.0):
        wf = cfg.w_fidelity / w_sum
        ws = cfg.w_stability / w_sum
        wa = cfg.w_agreement / w_sum
        wr = cfg.w_robustness / w_sum
    else:
        wf, ws, wa, wr = cfg.w_fidelity, cfg.w_stability, cfg.w_agreement, cfg.w_robustness

    df["ERS"] = (
        wf * df["fidelity_norm"]
        + ws * df["stability_norm"]
        + wa * df["agreement_norm"]
        + wr * df["robustness_norm"]
    )

    # Save full per-sentence results
    out_path = out_dir / cfg.out_csv
    df.to_csv(out_path, index=False)
    print("\nSaved:", out_path)

    # ------------------------------------------------------------------
    # Diagnostic summaries — these drive Table 9 and Figure 48 (Section 6.4.5)
    # ------------------------------------------------------------------
    print("\nERS summary:")
    print(df["ERS"].describe())

    print("\nERS by predicted class:")
    print(df.groupby("orig_label")["ERS"].describe())

    print("\nComponent scaling (winsorized min/max):")
    print("  fidelity_drop:", fid_stats)
    print("  stability_no_stopwords:", stab_stats)
    print("  topk_overlap:", agr_stats)
    print("  robust_expl (1-xai_change):", rob_stats)

    # Best and worst ERS examples — used for the high/low ERS case analysis
    # in Section 6.4.5 (ERS > 0.85 and ERS < 0.05 samples)
    cols_show = ["idx", "orig_label", "confidence", "fidelity_drop",
                 "stability_no_stopwords", "topk_overlap", "robust_expl", "ERS"]

    print("\nTop 3 ERS samples:")
    print(df.sort_values("ERS", ascending=False)[cols_show].head(3))

    print("\nBottom 3 ERS samples:")
    print(df.sort_values("ERS", ascending=True)[cols_show].head(3))


if __name__ == "__main__":
    main()
    
