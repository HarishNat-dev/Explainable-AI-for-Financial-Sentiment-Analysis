# =============================================================================
# src/evaluation/robustness_xai_interaction.py
#
# Computes the Robustness metric for the XAI evaluation framework.
# Tests whether IG explanations remain consistent when stopwords are removed
# from the input, and whether the prediction itself flips as a result.
#
# For each sentence in the test set:
#   1. Get original prediction and top-k IG tokens
#   2. Remove stopwords (semantically irrelevant perturbation)
#   3. Get perturbed prediction and top-k IG tokens
#   4. XAI change = 1 - overlap(original top-k, perturbed top-k)
#   5. Label flip = 1 if predicted class changed, else 0
#
# The key counterintuitive finding (Section 6.4.4): the 4 sentences where
# the label flipped show LOWER mean XAI change (0.40) than the 196 that
# did not flip (0.61). This is discussed and cautiously interpreted in
# Section 6.4.4 given the small flip group size.
#
# Output: reports/evaluation/robustness_xai.csv
#   Consumed by composite_score.py (ERS, Equation 6.8) and make_plots.py.
#
# Report: Section 6.2 (robustness component of Equation 6.8), Section 6.4.4
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
from dataclasses import dataclass
from typing import List, Set

import pandas as pd

from src.inference.predict import FinBERTPredictor, InferenceConfig
from src.xai.ig_explainer import FinBERTIGExplainer, IGConfig


@dataclass(frozen=True)
class RobustCfg:
    """
    top_k=5 is the same k used in fidelity (Equation 6.5) for consistency.
    max_rows=200 covers the full held-out test set.
    """
    test_path: str = "data/processed/phrasebank_allagree_test.parquet"
    text_col: str = "sentence"
    max_rows: int = 200
    top_k: int = 5
    out_dir: str = "reports/evaluation"
    out_csv: str = "robustness_xai.csv"


# Stopwords treated as semantically irrelevant — removing them should not
# change the IG attribution set if the explanation is robust (Section 6.2)
STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","when","while",
    "to","of","in","on","for","with","at","by","from","as","is","are",
    "was","were","be","been","being","it","its","this","that","these","those",
    "has","have","had","will","would","can","could","may","might","should",
}


def _clean_spaces(text: str) -> str:
    """Remove repeated spaces and fix spacing around punctuation."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text


def perturb_remove_stopwords(text: str) -> str:
    """
    Semantically irrelevant perturbation: removes function words (stopwords)
    while preserving all content tokens. This is the perturbation type used
    in the robustness evaluation and also in stability (Equation 6.6).
    A robust explanation should keep the same top-k tokens after this change.
    """
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    kept = [tok for tok in tokens if not (tok.isalpha() and tok.lower() in STOPWORDS)]
    return _clean_spaces(" ".join(kept))


def normalize(tok: str) -> str:
    """Strip WordPiece ## prefix and lowercase for token set comparison."""
    return tok.replace("##", "").strip().lower()


def topk_ig(ig_out: dict, k: int) -> List[str]:
    """
    Returns top-k token strings by absolute IG attribution magnitude.
    Tokens are normalised (## stripped, lowercased) for reliable set overlap.
    """
    pairs = [(t, float(a)) for t, a in zip(ig_out["tokens"], ig_out["attributions"])
             if str(t).strip() != ""]
    pairs = [(normalize(t), a) for t, a in pairs if normalize(t)]
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:k]
    return [t for t, _ in pairs_sorted]


def overlap(a: List[str], b: List[str], k: int) -> float:
    """
    Normalised top-k token set overlap: |A ∩ B| / k.
    Used directly in the stability metric (Equation 6.6) and inverted
    (1 - overlap) to give XAI change for the robustness component.
    """
    sa: Set[str] = set(a)
    sb: Set[str] = set(b)
    return len(sa.intersection(sb)) / float(k) if k else 0.0


def main():
    cfg = RobustCfg()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.test_path).head(cfg.max_rows).copy()

    predictor = FinBERTPredictor(InferenceConfig())
    ig = FinBERTIGExplainer(IGConfig(n_steps=30))

    rows = []
    for idx, row in df.iterrows():
        text = str(row[cfg.text_col])

        # Step 1: original prediction and IG attribution
        pred0 = predictor.predict_one(text)
        label0 = pred0["label"]
        conf0 = float(pred0["confidence"])
        ig0 = topk_ig(ig.explain(text), cfg.top_k)

        # Step 2: stopword-removed perturbation — same perturbation type
        # used in stability (Equation 6.6) for consistency
        pert = perturb_remove_stopwords(text)
        pred1 = predictor.predict_one(pert)
        label1 = pred1["label"]
        conf1 = float(pred1["confidence"])
        ig1 = topk_ig(ig.explain(pert), cfg.top_k)

        # Step 3: compute robustness metrics
        # label_flip=1 means the prediction changed — only 4/200 cases (2%)
        # confirming FinBERT is robust at the prediction level (Section 6.4.4)
        flip = int(label0 != label1)

        # xai_change = 1 - overlap; higher means explanation changed more
        # In composite_score.py this is inverted: robust_expl = 1 - xai_change
        xai_change = 1.0 - overlap(ig0, ig1, cfg.top_k)

        rows.append({
            "idx": int(idx),
            "orig_label": label0,
            "orig_conf": conf0,
            "pert_label": label1,
            "pert_conf": conf1,
            "label_flip": flip,
            "topk_overlap": 1.0 - xai_change,
            "xai_change": xai_change,
            "orig_top_tokens": ", ".join(ig0),
            "pert_top_tokens": ", ".join(ig1),
        })

    res = pd.DataFrame(rows)
    out_csv = out_dir / cfg.out_csv
    res.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)

    # ------------------------------------------------------------------
    # Summary statistics — feed into Section 6.4.4 analysis
    # Label flip rate = 4/200 = 2% (Section 6.4.4)
    # Mean XAI change ≈ 0.605 across all 200 sentences (Section 6.4.4)
    # ------------------------------------------------------------------
    print("\nFlip rate:", res["label_flip"].mean())
    print("\nXAI change summary (1 - overlap):")
    print(res["xai_change"].describe())

    # The counterintuitive flip vs no-flip comparison reported in Section 6.4.4:
    # flip group shows LOWER mean XAI change (0.40) than no-flip (0.61)
    print("\nXAI change when flip vs no flip (cf. Section 6.4.4):")
    print(res.groupby("label_flip")["xai_change"].describe())

    worst = res.sort_values("xai_change", ascending=False).iloc[0]
    print("\n--- Max XAI Change Case ---")
    print(worst[["orig_label", "pert_label", "label_flip", "orig_conf",
                 "pert_conf", "orig_top_tokens", "pert_top_tokens", "xai_change"]])


if __name__ == "__main__":
    main()
    
