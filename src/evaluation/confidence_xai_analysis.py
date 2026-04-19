# =============================================================================
# src/evaluation/confidence_xai_analysis.py
#
# Analyses the relationship between model confidence and XAI quality metrics
# (fidelity, stability, agreement) across the 200-sentence test set.
#
# Produces binned correlation data reported in Section 6.4.6. The key finding
# is that high-confidence neutral predictions do NOT have higher fidelity —
# confidence cannot be used as a proxy for explanation quality when predicting
# neutral sentiment (Figure 43, Section 6.4.1).
#
# Output: reports/evaluation/confidence_xai.csv
#   Grouped by confidence decile with mean XAI metric per bin.
#
# Note: this script is intentionally separate from composite_score.py so
# the confidence-XAI relationship can be re-examined without recomputing ERS
# (independent re-runnability principle, Section 3.6.5).
#
# Report: Section 6.4.6 — Confidence and XAI Correlation
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ConfXAICfg:
    """Paths to the pre-computed evaluation CSVs needed for this analysis."""
    fidelity_csv: str = "reports/evaluation/fidelity_ig_results.csv"
    stability_csv: str = "reports/evaluation/stability_ig_results.csv"
    agreement_csv: str = "reports/evaluation/agreement_ig_shap.csv"
    out_dir: str = "reports/evaluation"
    out_csv: str = "confidence_xai.csv"


def main():
    cfg = ConfXAICfg()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fid = pd.read_csv(cfg.fidelity_csv)
    stab = pd.read_csv(cfg.stability_csv)

    # Merge fidelity and stability on sentence index
    df = fid.merge(
        stab[["idx", "stability_no_punct", "stability_no_stopwords", "orig_label"]],
        on="idx",
        how="left",
        suffixes=("", "_stab"),
    )

    # orig_conf_max = softmax probability of the winning class (model confidence)
    df = df.rename(columns={"orig_conf_max": "confidence"})

    # Optionally join cross-method agreement if the CSV exists
    try:
        agr = pd.read_csv(cfg.agreement_csv)
        df = df.merge(agr[["idx", "topk_overlap"]], on="idx", how="left")
    except Exception:
        df["topk_overlap"] = pd.NA

    # ------------------------------------------------------------------
    # Pearson and Spearman correlations between confidence and XAI metrics.
    # The near-zero correlations for neutral-class predictions are the key
    # result supporting Section 6.4.6's finding that the vast majority of
    # predictions cluster at confidence=1.0, creating vertical bands rather
    # than meaningful correlation gradients (Section 6.4.6).
    # ------------------------------------------------------------------
    numeric_cols = ["confidence", "fidelity_drop", "stability_no_punct", "stability_no_stopwords"]
    corr_pearson = df[numeric_cols].corr(method="pearson")["confidence"]
    corr_spearman = df[numeric_cols].corr(method="spearman")["confidence"]

    print("\nPearson correlation with confidence:")
    print(corr_pearson)

    print("\nSpearman correlation with confidence:")
    print(corr_spearman)

    # ------------------------------------------------------------------
    # Bin by confidence decile to show how XAI metrics vary across the
    # confidence range — drives the scatter plots in Figure 43 (Section 6.4.1)
    # and the ceiling-effect observation in Section 6.4.6
    # ------------------------------------------------------------------
    df["conf_bin"] = pd.qcut(df["confidence"], 10, duplicates="drop")
    grouped = df.groupby("conf_bin").agg(
        n=("idx", "count"),
        mean_conf=("confidence", "mean"),
        mean_fidelity=("fidelity_drop", "mean"),
        mean_stab_punct=("stability_no_punct", "mean"),
        mean_stab_stop=("stability_no_stopwords", "mean"),
    ).reset_index()

    out_csv = out_dir / cfg.out_csv
    grouped.to_csv(out_csv, index=False)
    print("\nSaved binned analysis:", out_csv)
    print(grouped)


if __name__ == "__main__":
    main()
    
