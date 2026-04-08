from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class PlotConfig:
    fidelity_csv: str = "reports/evaluation/fidelity_ig_results.csv"
    stability_csv: str = "reports/evaluation/stability_ig_results.csv"
    agreement_csv: str = "reports/evaluation/agreement_ig_shap.csv"
    robustness_csv: str = "reports/evaluation/robustness_xai.csv"
    out_dir: str = "reports/figures"


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)


def boxplot_by_class(df: pd.DataFrame, value_col: str, class_col: str, title: str, ylabel: str, out_path: Path):
    classes = ["negative", "neutral", "positive"]
    data = [df[df[class_col] == c][value_col].dropna().astype(float).values for c in classes]

    plt.figure()
    plt.boxplot(data, labels=classes, showfliers=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Predicted class")
    save_fig(out_path)


def histogram(df: pd.DataFrame, value_col: str, title: str, xlabel: str, out_path: Path, bins: int = 20):
    plt.figure()
    plt.hist(df[value_col].dropna().astype(float).values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    save_fig(out_path)


def scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, out_path: Path):
    plt.figure()
    plt.scatter(df[x_col].astype(float).values, df[y_col].astype(float).values, s=15)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_fig(out_path)


def main():
    cfg = PlotConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    fid = pd.read_csv(cfg.fidelity_csv)
    stab = pd.read_csv(cfg.stability_csv)
    agr = pd.read_csv(cfg.agreement_csv)
    rob = pd.read_csv(cfg.robustness_csv)

    # -----------------------------
    # 1) Fidelity boxplot by class
    # -----------------------------
    boxplot_by_class(
        fid,
        value_col="fidelity_drop",
        class_col="orig_label",
        title="Fidelity (IG) by Predicted Class",
        ylabel="P(orig_class|text) - P(orig_class|masked)",
        out_path=out_dir / "fidelity_by_class.png",
    )

    # -----------------------------
    # 2) Stability boxplots by class
    # -----------------------------
    boxplot_by_class(
        stab,
        value_col="stability_no_punct",
        class_col="orig_label",
        title="IG Stability under Punctuation Removal (Top-k overlap / k)",
        ylabel="Stability score",
        out_path=out_dir / "stability_no_punct_by_class.png",
    )

    boxplot_by_class(
        stab,
        value_col="stability_no_stopwords",
        class_col="orig_label",
        title="IG Stability under Stopword Removal (Top-k overlap / k)",
        ylabel="Stability score",
        out_path=out_dir / "stability_no_stopwords_by_class.png",
    )

    # -----------------------------
    # 3) IG vs SHAP agreement by class
    # -----------------------------
    boxplot_by_class(
        agr,
        value_col="topk_overlap",
        class_col="orig_label",
        title="IG vs SHAP Agreement by Predicted Class (Top-k overlap / k)",
        ylabel="Top-k overlap",
        out_path=out_dir / "ig_shap_agreement_by_class.png",
    )

    # -----------------------------
    # 4) Robustness: XAI change distribution
    # -----------------------------
    histogram(
        rob,
        value_col="xai_change",
        title="Explanation Change under Stopword Perturbation (1 - overlap)",
        xlabel="XAI change",
        out_path=out_dir / "xai_change_hist.png",
        bins=20,
    )

    # Optional: boxplot by flip/no-flip
    plt.figure()
    data_nf = rob[rob["label_flip"] == 0]["xai_change"].dropna().astype(float).values
    data_f = rob[rob["label_flip"] == 1]["xai_change"].dropna().astype(float).values
    plt.boxplot([data_nf, data_f], labels=["no_flip", "flip"], showfliers=True)
    plt.title("XAI Change by Label Flip (Stopword Perturbation)")
    plt.ylabel("XAI change (1 - overlap)")
    plt.xlabel("Flip condition")
    save_fig(out_dir / "xai_change_by_flip.png")

    # -----------------------------
    # 5) Confidence vs explainability scatter plots
    # -----------------------------
    # Join confidence from fidelity with stability
    df_cs = fid[["idx", "orig_conf_max", "fidelity_drop", "orig_label"]].merge(
        stab[["idx", "stability_no_punct", "stability_no_stopwords"]],
        on="idx",
        how="left",
    ).rename(columns={"orig_conf_max": "confidence"})

    scatter(
        df_cs.dropna(subset=["confidence", "fidelity_drop"]),
        x_col="confidence",
        y_col="fidelity_drop",
        title="Confidence vs Fidelity (IG)",
        xlabel="Confidence (max class probability)",
        ylabel="Fidelity drop",
        out_path=out_dir / "confidence_vs_fidelity.png",
    )

    scatter(
        df_cs.dropna(subset=["confidence", "stability_no_stopwords"]),
        x_col="confidence",
        y_col="stability_no_stopwords",
        title="Confidence vs Stability (Stopword Removal)",
        xlabel="Confidence (max class probability)",
        ylabel="Stability score",
        out_path=out_dir / "confidence_vs_stability_stopwords.png",
    )

    scatter(
        df_cs.dropna(subset=["confidence", "stability_no_punct"]),
        x_col="confidence",
        y_col="stability_no_punct",
        title="Confidence vs Stability (Punctuation Removal)",
        xlabel="Confidence (max class probability)",
        ylabel="Stability score",
        out_path=out_dir / "confidence_vs_stability_punct.png",
    )

    print("\nAll plots saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()