# =============================================================================
# src/app/batch_utils.py
#
# Batch processing utilities for the Streamlit dashboard (Section 4.7.4).
# Handles CSV column detection, bulk sentiment prediction, and optional
# Integrated Gradients explanation for each row.
#
# Used by: src/app/streamlit_app.py (Batch CSV tab)
# Report:  Section 4.7.4 — Batch CSV Processing and Results
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


# Common column name variants the dashboard will try to auto-detect
# as the text/headline column when a CSV is uploaded (Section 4.7.4)
COMMON_TEXT_COLUMNS = [
    "text", "sentence", "headline", "title", "news", "content", "description"
]


@dataclass(frozen=True)
class BatchResultConfig:
    top_k_tokens: int = 8


def guess_text_column(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detects which CSV column contains the text/headlines.
    Checks column names against COMMON_TEXT_COLUMNS (case-insensitive).
    Returns None if no match found — user is then prompted to select manually.
    Satisfies NF5 usability requirement (Section 3.4.2).
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for c in COMMON_TEXT_COLUMNS:
        if c in cols_lower:
            return cols_lower[c]
    return None


def run_batch_predictions(df: pd.DataFrame, text_col: str, predictor) -> pd.DataFrame:
    """
    Runs FinBERT sentiment prediction on every row in the DataFrame.

    predictor: FinBERTPredictor (src/inference/predict.py)
    Returns df with added columns:
      - sentiment        : predicted class label (negative/neutral/positive)
      - confidence       : softmax probability of the winning class
      - prob_negative, prob_neutral, prob_positive : full class distribution

    All three class probabilities sum to 1.0 — verified in test T1 (Section 5.1,
    Table 7). Displaying all three is a deliberate design decision (Section 4.7.1)
    so analysts can calibrate trust rather than treating output as a binary verdict.
    """
    texts = df[text_col].astype(str).fillna("").tolist()
    probs = predictor.predict_proba(texts).numpy()  # shape [N, 3]
    pred_ids = probs.argmax(axis=1)

    # Label mapping matches FinBERT's output head order: 0=negative, 1=neutral, 2=positive
    id_to_label = {0: "negative", 1: "neutral", 2: "positive"}
    out = df.copy()
    out["sentiment"] = [id_to_label[int(i)] for i in pred_ids]
    out["confidence"] = probs.max(axis=1).astype(float)

    out["prob_negative"] = probs[:, 0].astype(float)
    out["prob_neutral"] = probs[:, 1].astype(float)
    out["prob_positive"] = probs[:, 2].astype(float)
    return out


def add_top_ig_tokens(
    df: pd.DataFrame,
    text_col: str,
    ig_explainer,
    top_k: int = 5,
    max_rows_to_explain: int = 200,
) -> pd.DataFrame:
    """
    Appends a 'top_ig_tokens' column listing the top-k tokens by absolute
    IG attribution magnitude for each row.

    ig_explainer: FinBERTIGExplainer (src/xai/ig_explainer.py)
    top_k:        how many tokens to surface per row (controlled by UI slider)
    max_rows_to_explain: CPU safety cap — IG takes ~1-3s per sentence on CPU
                         so full-batch explanation of 500 rows is avoided
                         unless explicitly configured (Section 4.7.4, NF1/NF6).

    Rows beyond max_rows_to_explain are marked 'NOT_EXPLAINED' rather than
    silently omitted — relates to the silent-skip usability gap documented
    in test T11 (Section 5.2, Table 7).

    Token attribution sorting by |attribution| surfaces the most influential
    tokens regardless of direction (green or red), consistent with the
    fidelity metric definition (Equation 6.5, Section 6.2).
    """
    out = df.copy()
    out["top_ig_tokens"] = ""

    n = min(len(out), max_rows_to_explain)
    for i in range(n):
        text = str(out.loc[out.index[i], text_col])
        try:
            exp = ig_explainer.explain(text)
            tokens = exp["tokens"]
            attrs = exp["attributions"]

            # Strip WordPiece subword prefix (##) and sort by absolute attribution
            pairs = [(t.replace("##", ""), float(a)) for t, a in zip(tokens, attrs) if str(t).strip() != ""]
            pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_k]
            top_tokens = [t for t, _ in pairs_sorted]

            out.loc[out.index[i], "top_ig_tokens"] = ", ".join(top_tokens)
        except Exception:
            out.loc[out.index[i], "top_ig_tokens"] = ""

    if len(out) > max_rows_to_explain:
        # Rows not explained are flagged explicitly rather than left blank
        out.loc[out.index[max_rows_to_explain:], "top_ig_tokens"] = "NOT_EXPLAINED"

    return out
