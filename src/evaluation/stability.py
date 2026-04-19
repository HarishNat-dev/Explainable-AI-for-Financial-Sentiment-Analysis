# =============================================================================
# src/evaluation/stability.py
#
# Computes the Stability metric (Equation 6.6, Section 6.2) for all 200
# sentences in the held-out test set.
#
# Stability measures how consistently IG identifies the same top-k tokens
# when the input is subjected to a semantically irrelevant perturbation.
# Two perturbation types are tested (Section 6.4.2):
#   - Punctuation removal: removes all non-alphanumeric characters
#   - Stopword removal:    removes function words (a, the, and, etc.)
#
# Stability score = |top-k(original) ∩ top-k(perturbed)| / k  (Equation 6.6)
# A score near 1.0 means the same tokens are identified before and after;
# near 0.0 means the explanation is sensitive to irrelevant surface changes.
#
# The neutral class consistently scores lower on both perturbation types,
# reinforcing the fidelity finding that neutral explanations are less reliable
# (Section 6.4.2, Figures 44-45).
#
# Output: reports/evaluation/stability_ig_results.csv
#   Consumed by composite_score.py (ERS), significance_tests.py, make_plots.py.
#
# Report: Section 6.2 (Equation 6.6), Section 6.4.2, Figures 44-45
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
class StabilityConfig:
    """
    top_k=5 matches the k in Equation 6.6 and is consistent with fidelity.
    max_rows=200 covers the full held-out test set (Section 4.2).
    n_steps_ig=30 matches the integration step count in fidelity.py and 4.5.1.
    """
    test_path: str = "data/processed/phrasebank_allagree_test.parquet"
    text_col: str = "sentence"
    top_k: int = 5
    max_rows: int = 200
    out_dir: str = "reports/evaluation"
    out_csv: str = "stability_ig_results.csv"
    n_steps_ig: int = 30


# Shared stopword list with robustness_xai_interaction.py for consistency
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "to", "of", "in", "on", "for", "with", "at", "by", "from", "as", "is", "are",
    "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those",
    "has", "have", "had", "will", "would", "can", "could", "may", "might", "should",
}


def _clean_spaces(text: str) -> str:
    """Remove repeated spaces and fix spacing around punctuation."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text


def perturb_remove_punctuation(text: str) -> str:
    """
    Perturbation type 1: removes all punctuation while keeping words and numbers.
    A stable explanation should identify the same top-k content tokens with or
    without punctuation marks (Section 6.4.2, Figure 44).
    """
    t = re.sub(r"[^\w\s]", "", text)
    return _clean_spaces(t)


def perturb_remove_stopwords(text: str) -> str:
    """
    Perturbation type 2: removes function words (stopwords) while keeping
    all content tokens. Stopword removal causes larger attribution shifts than
    punctuation removal because it changes sentence structure more substantially
    — this is why stability_no_stopwords scores are consistently lower than
    stability_no_punct (Section 6.4.2).
    """
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    kept = [tok for tok in tokens
            if not (tok.isalpha() and tok.lower() in STOPWORDS)]
    return _clean_spaces(" ".join(kept))


def topk_ig_tokens(ig_out: dict, k: int) -> List[str]:
    """Returns top-k tokens by absolute IG attribution magnitude."""
    tokens = ig_out["tokens"]
    attrs = ig_out["attributions"]
    pairs = [(t, float(a)) for t, a in zip(tokens, attrs) if str(t).strip() != ""]
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:k]
    return [t for t, _ in pairs_sorted]


def normalize_token(tok: str) -> str:
    """Strip WordPiece ## prefix and lowercase for reliable set overlap."""
    return tok.replace("##", "").strip().lower()


def stability_score(tokens_a: List[str], tokens_b: List[str], k: int) -> float:
    """
    Equation 6.6: |top-k(original) ∩ top-k(perturbed)| / k
    Tokens are normalised before comparison to avoid ## prefix mismatches.
    """
    set_a: Set[str] = {normalize_token(t) for t in tokens_a if normalize_token(t)}
    set_b: Set[str] = {normalize_token(t) for t in tokens_b if normalize_token(t)}
    if k <= 0:
        return 0.0
    return len(set_a.intersection(set_b)) / float(k)


def main():
    cfg = StabilityConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.test_path).head(cfg.max_rows).copy()

    predictor = FinBERTPredictor(InferenceConfig())
    ig = FinBERTIGExplainer(IGConfig(n_steps=cfg.n_steps_ig))

    rows = []
    for idx, row in df.iterrows():
        text = str(row[cfg.text_col])

        # Original prediction and IG explanation for the predicted class
        pred = predictor.predict_one(text)
        orig_label = pred["label"]

        ig_orig = ig.explain(text)
        top_orig = topk_ig_tokens(ig_orig, cfg.top_k)

        # Perturbation 1: punctuation removal (Figure 44, Section 6.4.2)
        text_nopunct = perturb_remove_punctuation(text)
        ig_nopunct = ig.explain(text_nopunct)
        top_nopunct = topk_ig_tokens(ig_nopunct, cfg.top_k)
        stab_nopunct = stability_score(top_orig, top_nopunct, cfg.top_k)

        # Perturbation 2: stopword removal (Figure 45, Section 6.4.2)
        # This perturbation is also used in robustness_xai_interaction.py
        text_nostop = perturb_remove_stopwords(text)
        ig_nostop = ig.explain(text_nostop)
        top_nostop = topk_ig_tokens(ig_nostop, cfg.top_k)
        stab_nostop = stability_score(top_orig, top_nostop, cfg.top_k)

        rows.append({
            "idx": int(idx),
            "orig_label": orig_label,
            "text": text,
            "top_tokens_orig": ", ".join([t.replace("##", "") for t in top_orig]),
            "text_no_punct": text_nopunct,
            "top_tokens_no_punct": ", ".join([t.replace("##", "") for t in top_nopunct]),
            "stability_no_punct": stab_nopunct,
            "text_no_stopwords": text_nostop,
            "top_tokens_no_stopwords": ", ".join([t.replace("##", "") for t in top_nostop]),
            "stability_no_stopwords": stab_nostop,
        })

    results = pd.DataFrame(rows)

    out_csv = out_dir / cfg.out_csv
    results.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)

    # ------------------------------------------------------------------
    # Summary statistics — drive Figures 44/45 and Section 6.4.2 analysis
    # ------------------------------------------------------------------
    print("\nStability summary (punctuation removal):")
    print(results["stability_no_punct"].describe())

    print("\nStability summary (stopword removal):")
    print(results["stability_no_stopwords"].describe())

    print("\nStability by predicted class (punctuation removal):")
    print(results.groupby("orig_label")["stability_no_punct"].describe())

    print("\nStability by predicted class (stopword removal):")
    print(results.groupby("orig_label")["stability_no_stopwords"].describe())

    # Case studies: lowest stability examples for qualitative discussion
    worst_punct = results.sort_values("stability_no_punct", ascending=True).iloc[0]
    worst_stop = results.sort_values("stability_no_stopwords", ascending=True).iloc[0]

    print("\n==============================")
    print("CASE STUDIES (LOW STABILITY)")
    print("==============================")

    print("\n--- Lowest Stability (Punctuation Removal) ---")
    print(worst_punct[[
        "text", "orig_label", "top_tokens_orig",
        "text_no_punct", "top_tokens_no_punct", "stability_no_punct"
    ]])

    print("\n--- Lowest Stability (Stopword Removal) ---")
    print(worst_stop[[
        "text", "orig_label", "top_tokens_orig",
        "text_no_stopwords", "top_tokens_no_stopwords", "stability_no_stopwords"
    ]])


if __name__ == "__main__":
    main()
    
