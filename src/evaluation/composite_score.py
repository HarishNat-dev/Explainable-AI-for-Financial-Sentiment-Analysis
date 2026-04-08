from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ERSConfig:
    fidelity_csv: str = "reports/evaluation/fidelity_ig_results.csv"
    stability_csv: str = "reports/evaluation/stability_ig_results.csv"
    agreement_csv: str = "reports/evaluation/agreement_ig_shap.csv"
    robustness_csv: str = "reports/evaluation/robustness_xai.csv"

    out_dir: str = "reports/evaluation"
    out_csv: str = "explainability_reliability_score.csv"

    # Winsorize to reduce outlier influence (robust min-max scaling)
    winsor_low: float = 0.01
    winsor_high: float = 0.99

    # Equal weights by default (defensible)
    w_fidelity: float = 0.25
    w_stability: float = 0.25
    w_agreement: float = 0.25
    w_robustness: float = 0.25


def winsor_minmax(x: pd.Series, low_q: float, high_q: float) -> Tuple[pd.Series, Dict[str, float]]:
    """Robust min-max scaling after winsorizing to quantiles."""
    xs = x.astype(float).copy()
    lo = float(xs.quantile(low_q))
    hi = float(xs.quantile(high_q))

    xs = xs.clip(lower=lo, upper=hi)

    # If hi == lo (degenerate), return zeros
    if np.isclose(hi, lo):
        return pd.Series(np.zeros(len(xs)), index=xs.index), {"lo": lo, "hi": hi}

    scaled = (xs - lo) / (hi - lo)
    return scaled, {"lo": lo, "hi": hi}


def main():
    cfg = ERSConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    fid = pd.read_csv(cfg.fidelity_csv)
    stab = pd.read_csv(cfg.stability_csv)
    agr = pd.read_csv(cfg.agreement_csv)
    rob = pd.read_csv(cfg.robustness_csv)

    # Build base by idx
    df = fid[["idx", "orig_label", "fidelity_drop", "orig_conf_max"]].rename(columns={"orig_conf_max": "confidence"}).copy()

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

    # Convert robustness into "higher is better"
    # xai_change = 1 - overlap; so overlap = 1 - xai_change
    df["robust_expl"] = 1.0 - df["xai_change"].astype(float)

    # ---- Normalize each component into [0,1] robustly ----
    # Fidelity can be slightly negative (seen in your results), so scaling matters.
    df["fidelity_norm"], fid_stats = winsor_minmax(df["fidelity_drop"], cfg.winsor_low, cfg.winsor_high)

    # Stability and agreement already in [0,1], but we still robust-scale for consistency
    df["stability_norm"], stab_stats = winsor_minmax(df["stability_no_stopwords"], cfg.winsor_low, cfg.winsor_high)
    df["agreement_norm"], agr_stats = winsor_minmax(df["topk_overlap"], cfg.winsor_low, cfg.winsor_high)
    df["robustness_norm"], rob_stats = winsor_minmax(df["robust_expl"], cfg.winsor_low, cfg.winsor_high)

    # ---- Composite score ----
    w_sum = cfg.w_fidelity + cfg.w_stability + cfg.w_agreement + cfg.w_robustness
    if not np.isclose(w_sum, 1.0):
        # Normalize weights automatically so user doesn’t accidentally break it
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

    # Save
    out_path = out_dir / cfg.out_csv
    df.to_csv(out_path, index=False)
    print("\nSaved:", out_path)

    # Summaries
    print("\nERS summary:")
    print(df["ERS"].describe())

    print("\nERS by predicted class:")
    print(df.groupby("orig_label")["ERS"].describe())

    print("\nComponent scaling (winsorized min/max):")
    print("  fidelity_drop:", fid_stats)
    print("  stability_no_stopwords:", stab_stats)
    print("  topk_overlap:", agr_stats)
    print("  robust_expl (1-xai_change):", rob_stats)

    # Helpful: show 3 best and 3 worst examples for report case studies
    cols_show = ["idx", "orig_label", "confidence", "fidelity_drop", "stability_no_stopwords", "topk_overlap", "robust_expl", "ERS"]

    print("\nTop 3 ERS samples:")
    print(df.sort_values("ERS", ascending=False)[cols_show].head(3))

    print("\nBottom 3 ERS samples:")
    print(df.sort_values("ERS", ascending=True)[cols_show].head(3))


if __name__ == "__main__":
    main()