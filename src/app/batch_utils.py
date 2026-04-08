from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


COMMON_TEXT_COLUMNS = [
    "text", "sentence", "headline", "title", "news", "content", "description"
]


@dataclass(frozen=True)
class BatchResultConfig:
    top_k_tokens: int = 8


def guess_text_column(df: pd.DataFrame) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in COMMON_TEXT_COLUMNS:
        if c in cols_lower:
            return cols_lower[c]
    return None


def run_batch_predictions(df: pd.DataFrame, text_col: str, predictor) -> pd.DataFrame:
    """
    predictor: FinBERTPredictor (from src.inference.predict)
    Returns df with:
      - sentiment
      - confidence
      - prob_negative, prob_neutral, prob_positive
    """
    texts = df[text_col].astype(str).fillna("").tolist()
    probs = predictor.predict_proba(texts).numpy()  # shape [N, 3]
    pred_ids = probs.argmax(axis=1)

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
    Adds a column 'top_ig_tokens' with the top-k tokens by |IG attribution| for each row.
    CPU-friendly: only explains up to max_rows_to_explain rows.
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

            pairs = [(t.replace("##", ""), float(a)) for t, a in zip(tokens, attrs) if str(t).strip() != ""]
            pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_k]
            top_tokens = [t for t, _ in pairs_sorted]

            out.loc[out.index[i], "top_ig_tokens"] = ", ".join(top_tokens)
        except Exception:
            out.loc[out.index[i], "top_ig_tokens"] = ""

    if len(out) > max_rows_to_explain:
        # mark rows we didn't explain
        out.loc[out.index[max_rows_to_explain:], "top_ig_tokens"] = "NOT_EXPLAINED"

    return out
