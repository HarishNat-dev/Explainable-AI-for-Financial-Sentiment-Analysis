# =============================================================================
# src/evaluation/agreement_ig_shap.py
#
# Cross-method agreement evaluation — measures how much IG and SHAP agree
# on which tokens matter most for each prediction (Equation 6.7, Section 6.2).
#
# For each sentence in the test set, runs both IG and SHAP, extracts the
# top-k tokens by absolute attribution from each, and computes the
# normalised set overlap (Jaccard-style) as the agreement score.
#
# Output: reports/evaluation/agreement_ig_shap.csv
#   One row per test sentence with columns:
#     idx, orig_label, confidence, ig_top_tokens, shap_top_tokens, topk_overlap
#
# Results feed into the cross-method agreement analysis in Section 6.4.3
# and the composite ERS computation in composite_score.py (Equation 6.8).
#
# Report:  Section 6.4.3 — IG vs SHAP Agreement; Equation 6.7
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass
from typing import List, Set

import pandas as pd

from src.inference.predict import FinBERTPredictor, InferenceConfig
from src.xai.ig_explainer import FinBERTIGExplainer, IGConfig
from src.xai.shap_explainer import FinBERTSHAPExplainer, SHAPConfig


@dataclass(frozen=True)
class AgreementConfig:
    test_path: str = "data/processed/phrasebank_allagree_test.parquet"
    text_col: str = "sentence"
    max_rows: int = 200          # evaluation runs on the 200-sentence test set
    top_k: int = 8               # k=8 as defined in Equation 6.7 (Section 6.2)
    out_dir: str = "reports/evaluation"
    out_csv: str = "agreement_ig_shap.csv"


def normalize(tok: str) -> str:
    """
    Strips WordPiece subword prefix (##) and lowercases for fair token comparison.
    Without this, 'profit' and '##profit' would be treated as different tokens,
    artificially deflating agreement scores.
    """
    return tok.replace("##", "").strip().lower()


def topk_from_out(tokens: List[str], scores: List[float], k: int) -> List[str]:
    """
    Extracts the top-k tokens by absolute attribution score from an explainer output.
    Normalises tokens (strips ## prefix) and filters empty strings before ranking.
    Used identically for both IG and SHAP outputs to ensure fair comparison.
    """
    pairs = [(t, float(s)) for t, s in zip(tokens, scores) if str(t).strip() != ""]
    pairs = [(t, s) for t, s in pairs if normalize(t)]
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:k]
    return [normalize(t) for t, _ in pairs_sorted]


def overlap_k(a: List[str], b: List[str], k: int) -> float:
    """
    Computes the normalised top-k overlap between two token lists (Equation 6.7).
    Implements: |top-k_IG ∩ top-k_SHAP| / k
    A score of 1.0 means both methods agree on all top-k tokens.
    A score of 0.0 means no overlap at all.
    """
    sa: Set[str] = set(a)
    sb: Set[str] = set(b)
    if k <= 0:
        return 0.0
    return len(sa.intersection(sb)) / float(k)


def main():
    cfg = AgreementConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.test_path).head(cfg.max_rows).copy()

    predictor = FinBERTPredictor(InferenceConfig())
    ig = FinBERTIGExplainer(IGConfig(n_steps=30))
    # max_evals=200 used here for evaluation; dashboard uses 250 (Section 4.5.2)
    shapx = FinBERTSHAPExplainer(SHAPConfig(max_evals=200))

    rows = []
    for idx, row in df.iterrows():
        text = str(row[cfg.text_col])

        # Get FinBERT prediction for this sentence
        pred = predictor.predict_one(text)
        label = pred["label"]
        conf = float(pred["confidence"])

        # Get top-k tokens from IG (gradient-based, model-specific)
        ig_out = ig.explain(text)
        ig_top = topk_from_out(ig_out["tokens"], ig_out["attributions"], cfg.top_k)

        # Get top-k tokens from SHAP (coalition-based, model-agnostic)
        shap_out = shapx.explain(text)
        shap_top = topk_from_out(shap_out["tokens"], shap_out["attributions"], cfg.top_k)

        # Compute normalised overlap — the cross-method agreement score (Eq 6.7)
        agree = overlap_k(ig_top, shap_top, cfg.top_k)

        rows.append({
            "idx": int(idx),
            "orig_label": label,
            "confidence": conf,
            "ig_top_tokens": ", ".join(ig_top),
            "shap_top_tokens": ", ".join(shap_top),
            "topk_overlap": agree,
        })

    res = pd.DataFrame(rows)
    out_csv = out_dir / cfg.out_csv
    res.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)

    print("\nAgreement summary (top-k overlap):")
    print(res["topk_overlap"].describe())

    print("\nAgreement by predicted class:")
    print(res.groupby("orig_label")["topk_overlap"].describe())


if __name__ == "__main__":
    main()
    
