from __future__ import annotations

import sys
from pathlib import Path

# --- Make project root importable ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ConfXAICfg:
    fidelity_csv: str = "reports/evaluation/fidelity_ig_results.csv"
    stability_csv: str = "reports/evaluation/stability_ig_results.csv"
    agreement_csv: str = "reports/evaluation/agreement_ig_shap.csv"  # if you ran it
    out_dir: str = "reports/evaluation"
    out_csv: str = "confidence_xai.csv"


def main():
    cfg = ConfXAICfg()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fid = pd.read_csv(cfg.fidelity_csv)
    stab = pd.read_csv(cfg.stability_csv)

    # Join on idx where possible
    df = fid.merge(
        stab[["idx", "stability_no_punct", "stability_no_stopwords", "orig_label"]],
        on="idx",
        how="left",
        suffixes=("", "_stab"),
    )

    # Confidence is in fid as orig_conf_max (max class confidence)
    df = df.rename(columns={"orig_conf_max": "confidence"})

    # Optional: join agreement if it exists
    try:
        agr = pd.read_csv(cfg.agreement_csv)
        df = df.merge(agr[["idx", "topk_overlap"]], on="idx", how="left")
    except Exception:
        df["topk_overlap"] = pd.NA

    # Correlations
    numeric_cols = ["confidence", "fidelity_drop", "stability_no_punct", "stability_no_stopwords"]
    corr_pearson = df[numeric_cols].corr(method="pearson")["confidence"]
    corr_spearman = df[numeric_cols].corr(method="spearman")["confidence"]

    print("\nPearson correlation with confidence:")
    print(corr_pearson)

    print("\nSpearman correlation with confidence:")
    print(corr_spearman)

    # Bin by confidence deciles
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
