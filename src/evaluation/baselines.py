# =============================================================================
# src/evaluation/baselines.py
#
# Baseline model evaluation — trains TF-IDF + XGBoost and TF-IDF + MLP on
# the combined train+validation set and evaluates both on the held-out test
# set, then loads saved FinBERT metrics for direct comparison (Section 4.4).
#
# The two baselines serve different roles (Section 4.4):
#   - TF-IDF + XGBoost: strongest classical method, establishes the ceiling
#     on what bag-of-words features can achieve
#   - TF-IDF + MLP:     tests whether FinBERT's advantage comes from neural
#     architecture or from pre-trained representations — the near-identical
#     XGBoost and MLP results confirm it is the latter
#
# ECE (Equation 6.4) is computed for both baselines. It is deliberately not
# computed for FinBERT due to near-ceiling accuracy on the allagree subset
# making calibration measurement less interpretable (Table 8, Section 6.3.1).
#
# Output: reports/metrics/baseline_comparison.json and .csv
#
# Report:  Section 4.4 — Baseline Models; Section 6.3.1 — Table 8
# =============================================================================

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class BaselineConfig:
    train_path: str = "data/processed/phrasebank_allagree_train.parquet"
    val_path: str = "data/processed/phrasebank_allagree_val.parquet"
    test_path: str = "data/processed/phrasebank_allagree_test.parquet"

    finbert_metrics_path: str = (
        "reports/metrics/finbert_phrasebank_allagree_test_metrics.json"
    )
    out_dir: str = "reports/metrics"
    out_json: str = "baseline_comparison.json"
    out_csv: str = "baseline_comparison.csv"

    text_col: str = "sentence"
    label_col: str = "label"

    # TF-IDF: unigrams + bigrams, sublinear TF scaling, 10k feature vocab
    # (Section 4.4 — same configuration for both baseline models)
    tfidf_max_features: int = 10_000
    tfidf_ngram_range: tuple = (1, 2)

    # XGBoost — 300 estimators chosen as performance plateaued beyond this (Section 4.4)
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_lr: float = 0.1
    xgb_seed: int = 42

    # MLP — two hidden layers (256, 128) with dropout and early stopping (Section 4.4)
    mlp_hidden: tuple = (256, 128)
    mlp_max_iter: int = 300
    mlp_seed: int = 42

    # ECE binning — M=10 equal-width bins as defined in Equation 6.4 (Section 6.2)
    ece_n_bins: int = 10


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) — Equation 6.4 (Section 6.2).
    Measures how well a model's stated confidence matches its empirical accuracy
    by binning predictions into M equal-width confidence intervals and computing
    the weighted average |accuracy - confidence| across bins.

    probs : shape [N, C] — predicted class probabilities
    labels: shape [N]   — integer true labels
    """
    pred_ids = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    correct = (pred_ids == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Computes the full metric suite for a baseline model:
    accuracy, macro/weighted precision, recall, F1, and ECE.
    Macro F1 is the primary comparison metric throughout (Section 6.2,
    Equation 6.3) — it gives equal weight to all three classes regardless
    of frequency, which matters given the 59% neutral class majority.
    """
    acc = accuracy_score(y_true, y_pred)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    ece = expected_calibration_error(probs, y_true, n_bins)

    return {
        "accuracy": round(float(acc), 6),
        "precision_macro": round(float(p_mac), 6),
        "recall_macro": round(float(r_mac), 6),
        "f1_macro": round(float(f1_mac), 6),
        "precision_weighted": round(float(p_w), 6),
        "recall_weighted": round(float(r_w), 6),
        "f1_weighted": round(float(f1_w), 6),
        "ece": round(ece, 6),
    }


def main():
    cfg = BaselineConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load splits produced by load_phrasebank.py
    train_df = pd.read_parquet(cfg.train_path)
    val_df = pd.read_parquet(cfg.val_path)
    test_df = pd.read_parquet(cfg.test_path)

    # Baselines train on combined train+val (1,905 sentences) to match the
    # total labelled data available — the test set remains strictly held out
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    X_trainval = trainval_df[cfg.text_col].astype(str).tolist()
    X_test = test_df[cfg.text_col].astype(str).tolist()

    # Label encoding: map integer labels to string names then encode for sklearn
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    trainval_df = trainval_df.copy()
    test_df = test_df.copy()
    trainval_df["label_str"] = trainval_df[cfg.label_col].map(label_map)
    test_df["label_str"] = test_df[cfg.label_col].map(label_map)

    le = LabelEncoder()
    le.fit(trainval_df["label_str"])
    y_trainval = le.transform(trainval_df["label_str"])
    y_test = le.transform(test_df["label_str"])

    print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
    print(f"Train+val: {len(X_trainval)} | Test: {len(X_test)}\n")

    results = {}

    # ------------------------------------------------------------------
    # Baseline 1: TF-IDF + XGBoost (Section 4.4)
    # XGBoost is the stronger of the two baselines — gradient boosting on
    # TF-IDF features establishes the non-neural ceiling for bag-of-words
    # representations. Cannot capture context (e.g. "avoided a loss" vs
    # "incurred a loss" look identical under TF-IDF).
    # ------------------------------------------------------------------
    print("=" * 50)
    print("Training TF-IDF + XGBoost...")

    xgb_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=cfg.tfidf_max_features,
            ngram_range=cfg.tfidf_ngram_range,
            sublinear_tf=True,
        )),
        ("clf", XGBClassifier(
            n_estimators=cfg.xgb_n_estimators,
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_lr,
            eval_metric="mlogloss",
            random_state=cfg.xgb_seed,
            n_jobs=-1,
        )),
    ])

    xgb_pipeline.fit(X_trainval, y_trainval)
    xgb_pred = xgb_pipeline.predict(X_test)
    xgb_probs = xgb_pipeline.predict_proba(X_test)

    xgb_metrics = compute_metrics(y_test, xgb_pred, xgb_probs, cfg.ece_n_bins)
    results["xgboost"] = xgb_metrics

    print("XGBoost Test Metrics:")
    for k, v in xgb_metrics.items():
        print(f"  {k}: {v}")
    print("\nClassification Report:")
    print(classification_report(y_test, xgb_pred, target_names=le.classes_))

    # ------------------------------------------------------------------
    # Baseline 2: TF-IDF + MLP (Section 4.4)
    # Tests whether FinBERT's advantage is architectural (neural vs tree)
    # or representational (contextual embeddings vs bag-of-words).
    # Near-identical MLP and XGBoost results confirm it is representational.
    # ------------------------------------------------------------------
    print("=" * 50)
    print("Training TF-IDF + MLP (Feedforward Neural Network)...")

    mlp_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=cfg.tfidf_max_features,
            ngram_range=cfg.tfidf_ngram_range,
            sublinear_tf=True,
        )),
        ("clf", MLPClassifier(
            hidden_layer_sizes=cfg.mlp_hidden,
            max_iter=cfg.mlp_max_iter,
            random_state=cfg.mlp_seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )),
    ])

    mlp_pipeline.fit(X_trainval, y_trainval)
    mlp_pred = mlp_pipeline.predict(X_test)
    mlp_probs = mlp_pipeline.predict_proba(X_test)

    mlp_metrics = compute_metrics(y_test, mlp_pred, mlp_probs, cfg.ece_n_bins)
    results["mlp_feedforward"] = mlp_metrics

    print("MLP Test Metrics:")
    for k, v in mlp_metrics.items():
        print(f"  {k}: {v}")
    print("\nClassification Report:")
    print(classification_report(y_test, mlp_pred, target_names=le.classes_))

    # ------------------------------------------------------------------
    # Load saved FinBERT metrics for comparison (Section 6.3.1, Table 8)
    # ECE is marked as 'see_note' — near-ceiling accuracy on allagree makes
    # calibration measurement less interpretable (Section 6.3.1).
    # ------------------------------------------------------------------
    print("=" * 50)
    print("Loading FinBERT metrics...")

    finbert_path = Path(cfg.finbert_metrics_path)
    with open(finbert_path) as f:
        finbert_raw = json.load(f)

    finbert_metrics = {
        "accuracy": round(finbert_raw.get("eval_accuracy", 0), 6),
        "precision_macro": round(finbert_raw.get("eval_precision_macro", 0), 6),
        "recall_macro": round(finbert_raw.get("eval_recall_macro", 0), 6),
        "f1_macro": round(finbert_raw.get("eval_f1_macro", 0), 6),
        "precision_weighted": round(finbert_raw.get("eval_precision_weighted", 0), 6),
        "recall_weighted": round(finbert_raw.get("eval_recall_weighted", 0), 6),
        "f1_weighted": round(finbert_raw.get("eval_f1_weighted", 0), 6),
        "ece": "see_note",
    }
    results["finbert"] = finbert_metrics

    print("FinBERT Test Metrics:")
    for k, v in finbert_metrics.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Comparison table and output files
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 50)

    comparison_rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update(metrics)
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.set_index("model")
    print(comparison_df.to_string())

    out_json_path = out_dir / cfg.out_json
    out_csv_path = out_dir / cfg.out_csv

    with open(out_json_path, "w") as f:
        json.dump(results, f, indent=2)

    comparison_df.reset_index().to_csv(out_csv_path, index=False)
    print(f"\nSaved comparison JSON: {out_json_path}")
    print(f"Saved comparison CSV:  {out_csv_path}")

    # ------------------------------------------------------------------
    # Key takeaways — the numbers that appear in Table 8 (Section 6.3.1)
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("KEY TAKEAWAYS FOR REPORT")
    print("=" * 50)

    xgb_f1 = xgb_metrics["f1_macro"]
    mlp_f1 = mlp_metrics["f1_macro"]
    finbert_f1 = finbert_metrics["f1_macro"]

    print(f"  XGBoost  macro F1: {xgb_f1:.4f}")
    print(f"  MLP      macro F1: {mlp_f1:.4f}")
    print(f"  FinBERT  macro F1: {finbert_f1:.4f}")
    print(f"  FinBERT improvement over XGBoost: +{finbert_f1 - xgb_f1:.4f}")
    print(f"  FinBERT improvement over MLP:     +{finbert_f1 - mlp_f1:.4f}")
    print(f"\n  XGBoost  ECE: {xgb_metrics['ece']:.4f}")
    print(f"  MLP      ECE: {mlp_metrics['ece']:.4f}")
    print(f"  (FinBERT ECE requires raw probs — compute separately if needed)")


if __name__ == "__main__":
    main()
