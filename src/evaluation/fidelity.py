# =============================================================================
# src/evaluation/fidelity.py
#
# Computes the Fidelity metric (Equation 6.5, Section 6.2) for all 200
# sentences in the held-out test set.
#
# Fidelity measures how much the model's confidence in its original predicted
# class drops when the top-k IG-attributed tokens are masked from the input.
# A high drop (near 1.0) means those tokens genuinely drove the prediction.
# A near-zero or negative drop — consistently found for the neutral class —
# means the IG explanation does not reflect the model's actual decision process
# (Section 6.4.1, Figure 42).
#
# Method:
#   1. Run FinBERT on the original sentence → record P(predicted_class | text)
#   2. Run IG explainer → get top-k tokens by absolute attribution magnitude
#   3. Remove those tokens from the text (whole-word, case-insensitive)
#   4. Run FinBERT on the masked sentence → record P(predicted_class | masked)
#   5. Fidelity = P(predicted_class | text) - P(predicted_class | masked)
#
# Output: reports/evaluation/fidelity_ig_results.csv
#   One row per sentence with fidelity score, top tokens, and masked text.
#   This CSV is consumed by composite_score.py (ERS) and make_plots.py.
#
# Report: Section 6.2 (Equation 6.5), Section 6.4.1, Table 7 (Section 5)
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from src.inference.predict import FinBERTPredictor, InferenceConfig
from src.xai.ig_explainer import FinBERTIGExplainer, IGConfig


LABELS = ["negative", "neutral", "positive"]


@dataclass(frozen=True)
class FidelityConfig:
    """
    Parameters for the fidelity evaluation run.
    top_k=5 matches the definition in Equation 6.5 and Section 6.2.
    max_rows=200 is the full held-out test set (Section 4.2).
    """
    test_path: str = "data/processed/phrasebank_allagree_test.parquet"
    text_col: str = "sentence"

    # k in Equation 6.5 — top-k tokens masked to measure confidence drop
    top_k: int = 5
    max_rows: int = 200

    out_dir: str = "reports/evaluation"
    out_csv: str = "fidelity_ig_results.csv"

    # Remove all occurrences of each top token (not just first) for cleaner masking
    remove_all_occurrences: bool = True


def _clean_spaces(text: str) -> str:
    """Remove repeated spaces and fix spacing around punctuation after token removal."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text


def mask_top_tokens(text: str, top_tokens: List[str], remove_all: bool = True) -> str:
    """
    Removes the top-k IG-attributed tokens from the input text to produce
    the masked sentence used in the fidelity calculation (Equation 6.5).

    Strips WordPiece ## prefixes before matching. Sorts longer tokens first
    to avoid partial-match issues (e.g. removing 'earn' before 'earnings').
    Uses whole-word boundary matching where possible to avoid false removals.
    """
    masked = text

    tokens = [t.replace("##", "").strip() for t in top_tokens]
    tokens = [t for t in tokens if t]
    # Longest first to prevent partial token matches corrupting longer tokens
    tokens = sorted(set(tokens), key=len, reverse=True)

    for tok in tokens:
        pattern = re.escape(tok)
        # Whole-word boundary for alphanumeric tokens; plain match otherwise
        if re.match(r"^\w+$", tok):
            regex = re.compile(rf"\b{pattern}\b", flags=re.IGNORECASE)
        else:
            regex = re.compile(pattern, flags=re.IGNORECASE)

        masked = regex.sub("", masked) if remove_all else regex.sub("", masked, count=1)

    return _clean_spaces(masked)


def topk_ig_tokens(ig_out: dict, k: int) -> List[str]:
    """
    Extracts the top-k tokens by absolute IG attribution magnitude.
    Sorting by |attribution| surfaces the most influential tokens regardless
    of direction, consistent with the fidelity metric definition (Equation 6.5).
    """
    tokens = ig_out["tokens"]
    attrs = ig_out["attributions"]
    pairs = [(t, float(a)) for t, a in zip(tokens, attrs) if str(t).strip() != ""]
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:k]
    return [t for t, _ in pairs_sorted]


def main():
    cfg = FidelityConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the held-out test set — never seen during training or
    # hyperparameter selection (strict holdout protocol, Table 4, Section 4.2)
    df = pd.read_parquet(cfg.test_path)
    df = df.head(cfg.max_rows).copy()

    predictor = FinBERTPredictor(InferenceConfig())
    ig = FinBERTIGExplainer(IGConfig(n_steps=30))

    rows = []
    for idx, row in df.iterrows():
        text = str(row[cfg.text_col])

        # Step 1: original prediction — record label and full probability distribution
        orig = predictor.predict_one(text)
        orig_label = orig["label"]
        orig_conf = float(orig["confidence"])
        orig_label_id = int(orig["label_id"])

        # P(predicted_class | original text) — the "before" value in Equation 6.5
        p_orig_class_before = float(orig["probabilities"][orig_label])

        # Step 2: IG explanation — get top-k tokens for the predicted class
        ig_out = ig.explain(text)
        top_tokens = topk_ig_tokens(ig_out, cfg.top_k)

        # Step 3: mask text by removing top-k tokens
        masked_text = mask_top_tokens(text, top_tokens, remove_all=cfg.remove_all_occurrences)

        # Step 4: run FinBERT on masked text
        masked = predictor.predict_one(masked_text)
        masked_label = masked["label"]
        masked_conf = float(masked["confidence"])

        # P(predicted_class | masked text) — the "after" value in Equation 6.5
        # Note: we always track the ORIGINAL predicted class, not the new one,
        # so the drop is interpretable even when masking changes the prediction
        p_orig_class_after = float(masked["probabilities"][orig_label])

        # Step 5: fidelity = probability drop for the original predicted class
        fidelity_drop = p_orig_class_before - p_orig_class_after

        # Legacy metric kept for comparison: drop in max-class confidence
        # (not the primary metric — fidelity_drop above is Equation 6.5)
        fidelity_drop_maxconf = orig_conf - masked_conf

        rows.append({
            "idx": int(idx),
            "text": text,
            "orig_label": orig_label,
            "orig_label_id": orig_label_id,
            "orig_conf_max": orig_conf,
            "p_orig_class_before": p_orig_class_before,
            "top_ig_tokens": ", ".join([t.replace("##", "") for t in top_tokens]),
            "masked_text": masked_text,
            "masked_label": masked_label,
            "masked_conf_max": masked_conf,
            "p_orig_class_after": p_orig_class_after,
            "fidelity_drop": fidelity_drop,
            "fidelity_drop_maxconf": fidelity_drop_maxconf,
        })

    results = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Case study extraction — best, worst, and median neutral examples.
    # These drive the qualitative discussion in Section 6.4.1 and the
    # concrete batch examples shown in Figures 13-14 (Section 4.7.4).
    # ------------------------------------------------------------------
    best_case = results.sort_values("fidelity_drop", ascending=False).iloc[0]
    worst_case = results.sort_values("fidelity_drop", ascending=True).iloc[0]

    neutral_df = results[results["orig_label"] == "neutral"]
    neutral_case = neutral_df.iloc[len(neutral_df) // 2] if len(neutral_df) > 0 else None

    case_cols = ["text", "orig_label", "p_orig_class_before",
                 "top_ig_tokens", "masked_text", "p_orig_class_after", "fidelity_drop"]

    print("\n==============================")
    print("CASE STUDIES")
    print("==============================")
    print("\n--- Highest Fidelity Drop (ERS > 0.85 region, Section 6.4.5) ---")
    print(best_case[case_cols])
    print("\n--- Lowest Fidelity Drop (ERS < 0.05 region, Section 6.4.5) ---")
    print(worst_case[case_cols])
    if neutral_case is not None:
        print("\n--- Representative Neutral Case (near-zero fidelity, Section 6.4.1) ---")
        print(neutral_case[case_cols])

    # ------------------------------------------------------------------
    # Summary statistics — drive Table 8 class-level results and the
    # Welch t-test inputs reported in Section 6.4.1
    # ------------------------------------------------------------------
    out_csv = out_dir / cfg.out_csv
    results.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)
    print("\nFidelity drop summary (Equation 6.5):")
    print(results["fidelity_drop"].describe())

    print("\nFidelity drop by original predicted class:")
    print(results.groupby("orig_label")["fidelity_drop"].describe())


if __name__ == "__main__":
    main()
    
