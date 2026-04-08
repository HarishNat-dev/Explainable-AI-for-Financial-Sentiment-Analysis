from __future__ import annotations

import sys
from pathlib import Path

# --- Make project root importable ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.inference.predict import FinBERTPredictor, InferenceConfig
from src.xai.ig_explainer import FinBERTIGExplainer, IGConfig


LABELS = ["negative", "neutral", "positive"]


@dataclass(frozen=True)
class FidelityConfig:
    # Data
    test_path: str = "data/processed/phrasebank_allagree_test.parquet"
    text_col: str = "sentence"

    # Fidelity parameters
    top_k: int = 5
    max_rows: int = 200  # keep CPU-friendly

    # Output
    out_dir: str = "reports/evaluation"
    out_csv: str = "fidelity_ig_results.csv"

    # Optional: when removing tokens, also remove duplicates?
    remove_all_occurrences: bool = True


def _clean_spaces(text: str) -> str:
    # remove repeated spaces and weird spacing around punctuation
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text


def mask_top_tokens(text: str, top_tokens: List[str], remove_all: bool = True) -> str:
    """
    Remove top tokens from the original text (case-insensitive).
    Token list may include WordPiece fragments; we strip ## and ignore empties.
    """
    masked = text

    # sort longer tokens first to avoid partial masking issues (e.g., "earn" before "earnings")
    tokens = [t.replace("##", "").strip() for t in top_tokens]
    tokens = [t for t in tokens if t]
    tokens = sorted(set(tokens), key=len, reverse=True)

    for tok in tokens:
        # escape for regex; match whole-word-ish boundaries where possible
        pattern = re.escape(tok)

        # Try word boundary replacement; if token contains non-word chars, fall back to simple replace
        if re.match(r"^\w+$", tok):
            regex = re.compile(rf"\b{pattern}\b", flags=re.IGNORECASE)
        else:
            regex = re.compile(pattern, flags=re.IGNORECASE)

        if remove_all:
            masked = regex.sub("", masked)
        else:
            masked = regex.sub("", masked, count=1)

    return _clean_spaces(masked)


def topk_ig_tokens(ig_out: dict, k: int) -> List[str]:
    tokens = ig_out["tokens"]
    attrs = ig_out["attributions"]
    pairs = [(t, float(a)) for t, a in zip(tokens, attrs) if str(t).strip() != ""]
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:k]
    return [t for t, _ in pairs_sorted]


def main():
    cfg = FidelityConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.test_path)
    df = df.head(cfg.max_rows).copy()

    predictor = FinBERTPredictor(InferenceConfig())
    ig = FinBERTIGExplainer(IGConfig(n_steps=30))

    rows = []
    for idx, row in df.iterrows():
        text = str(row[cfg.text_col])

        # Original prediction
        orig = predictor.predict_one(text)
        orig_label = orig["label"]
        orig_conf = float(orig["confidence"])
        orig_label_id = int(orig["label_id"])

        # Probability of the ORIGINAL predicted class (before masking)
        p_orig_class_before = float(
    orig["probabilities"][orig_label]  # e.g. P(positive|text) if positive was predicted
)

        # IG explanation for predicted class (default)
        ig_out = ig.explain(text)
        top_tokens = topk_ig_tokens(ig_out, cfg.top_k)

        # Mask text by removing top tokens
        masked_text = mask_top_tokens(text, top_tokens, remove_all=cfg.remove_all_occurrences)

        masked = predictor.predict_one(masked_text)
        masked_label = masked["label"]
        masked_conf = float(masked["confidence"])

        # Probability of the ORIGINAL predicted class after masking
        p_orig_class_after = float(
    masked["probabilities"][orig_label]
)

        # Proper fidelity: drop in probability of the original predicted class
        fidelity_drop = p_orig_class_before - p_orig_class_after

        # (Optional) keeping old metric too for comparison
        fidelity_drop_maxconf = orig_conf - masked_conf


        rows.append(
            {
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
            }
        )

    results = pd.DataFrame(rows)

    # ------------------------------
    # CASE STUDY EXTRACTION
    # ------------------------------

    # Highest fidelity drop
    best_case = results.sort_values("fidelity_drop", ascending=False).iloc[0]

    # Lowest fidelity drop
    worst_case = results.sort_values("fidelity_drop", ascending=True).iloc[0]

    # Representative neutral (median neutral fidelity)
    neutral_df = results[results["orig_label"] == "neutral"]
    neutral_case = neutral_df.iloc[len(neutral_df) // 2] if len(neutral_df) > 0 else None

    print("\n==============================")
    print("CASE STUDIES")
    print("==============================")

    print("\n--- Highest Fidelity Drop ---")
    print(best_case[[
        "text",
        "orig_label",
        "p_orig_class_before",
        "top_ig_tokens",
        "masked_text",
        "p_orig_class_after",
        "fidelity_drop"
    ]])

    print("\n--- Lowest Fidelity Drop ---")
    print(worst_case[[
        "text",
        "orig_label",
        "p_orig_class_before",
        "top_ig_tokens",
        "masked_text",
        "p_orig_class_after",
        "fidelity_drop"
    ]])

    if neutral_case is not None:
        print("\n--- Representative Neutral Case ---")
        print(neutral_case[[
            "text",
            "orig_label",
            "p_orig_class_before",
            "top_ig_tokens",
            "masked_text",
            "p_orig_class_after",
            "fidelity_drop"
        ]])


    # Summary stats
    summary = results["fidelity_drop"].describe()
    by_class = results.groupby("orig_label")["fidelity_drop"].describe()

    out_csv = out_dir / cfg.out_csv
    results.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)
    print("\nFidelity drop summary (P(orig_class|text) - P(orig_class|masked_text)):")
    print(summary)

    print("\nFidelity drop by original predicted class:")
    print(by_class)


if __name__ == "__main__":
    main()
