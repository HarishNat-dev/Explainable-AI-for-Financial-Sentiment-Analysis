# =============================================================================
# src/xai/shap_explainer.py
#
# SHAP explainer for fine-tuned FinBERT using the partition algorithm
# with a tokeniser-aware masker (Section 4.5.2, Equation 2.1).
#
# The partition algorithm approximates Shapley values by sampling token
# coalitions hierarchically, keeping model evaluations to max_evals rather
# than the exponential full computation (Section 4.5.2).
#
# Tokeniser-aware masker: replaces absent tokens with [MASK] rather than
# blank strings, preserving sequence length and positional encodings so
# model outputs are comparable across coalitions (Section 4.5.2).
#
# max_evals=300 for evaluation scripts; 250 for real-time Streamlit use
# (8-15s on CPU at either setting).
#
# The unified explain() interface matches ig_explainer.py exactly so the
# Streamlit tabs and evaluation scripts can consume both without branching
# logic (Section 3.6.4, Section 4.5).
#
# Report: Section 4.5.2, Section 2.3.2 (Equation 2.1), Section 4.7.3
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


@dataclass(frozen=True)
class SHAPConfig:
    model_dir: str  = "models/finbert/finbert_phrasebank_allagree/best_model"
    max_length: int = 128
    max_evals: int  = 300


def _normalize_attributions(attrs: np.ndarray) -> np.ndarray:
    """Normalise SHAP values to [-1, 1] — same as IG for consistent visualisation."""
    if np.allclose(attrs, 0):
        return attrs
    max_abs = np.max(np.abs(attrs))
    return attrs / (max_abs + 1e-12)


def tokens_to_html(tokens: List[str], scores: List[float]) -> str:
    """
    Renders colour-coded SHAP token attribution HTML for the Streamlit SHAP tab
    (Section 4.7.3). Identical colour scheme to ig_explainer.py for direct
    side-by-side comparison. Green = supports target class; red = opposes.
    """
    def style(score: float) -> str:
        s = max(-1.0, min(1.0, float(score)))
        if s >= 0:
            alpha = 0.15 + 0.55 * s
            return f"background-color: rgba(0, 200, 0, {alpha:.3f});"
        alpha = 0.15 + 0.55 * (-s)
        return f"background-color: rgba(255, 0, 0, {alpha:.3f});"

    rendered = []
    for tok, sc in zip(tokens, scores):
        rendered.append(
            "<span style='"
            + style(sc)
            + " padding:2px 4px; margin:1px; border-radius:6px; display:inline-block;'"
            + f" title='{sc:.3f}'>"
            + str(tok)
            + "</span>"
        )
    return "<div style='line-height: 2.2; font-size: 16px;'>" + " ".join(rendered) + "</div>"


class FinBERTSHAPExplainer:
    """
    SHAP text explainer using a tokeniser-aware masker and partition algorithm.
    Loaded once at startup via @st.cache_resource (Section 4.7.5).
    Also used by agreement_ig_shap.py for cross-method agreement evaluation.
    The explainer is built once in __init__ and reused across all explain() calls.
    """

    def __init__(self, cfg: SHAPConfig):
        self.cfg = cfg
        self.model_path = Path(cfg.model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Tokeniser-aware masker — replaces absent tokens with [MASK] to preserve
        # sequence length and positional encodings (Section 4.5.2)
        self.masker = shap.maskers.Text(self.tokenizer)

        self.explainer = shap.Explainer(
            self._predict_proba_np,
            self.masker,
            algorithm="partition",
            output_names=[LABEL_MAP[i] for i in range(3)],
        )

    @torch.no_grad()
    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """Batched forward pass returning softmax probabilities as numpy array [N, 3]."""
        enc    = self.tokenizer(texts, truncation=True, max_length=self.cfg.max_length,
                                return_tensors="pt", padding=True)
        enc    = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        probs  = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()

    def _predict_proba_np(self, texts) -> np.ndarray:
        """
        Normalises SHAP's various input formats (numpy arrays, token lists, strings)
        to List[str] before calling _predict_proba (Section 4.5.2).
        """
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        if not isinstance(texts, list):
            texts = [texts]

        cleaned: List[str] = []
        for t in texts:
            if isinstance(t, list):
                cleaned.append("".join(t) if all(isinstance(x, str) for x in t) else str(t))
            else:
                cleaned.append(str(t))
        return self._predict_proba(cleaned)

    def explain(self, text: str, target_label: int | None = None) -> Dict:
        """
        Computes SHAP token attributions for the given text (Section 4.5.2).
        exp.values shape is (1, num_tokens, num_classes) — extracts the slice
        for target_label. Returns the standard explanation dict matching
        ig_explainer.py: tokens, attributions, html, predicted{}, target_explained{}
        """
        probs   = self._predict_proba([text])[0]
        pred_id = int(probs.argmax())
        if target_label is None:
            target_label = pred_id

        exp    = self.explainer([text], max_evals=self.cfg.max_evals)
        tokens = list(exp.data[0])
        values = exp.values[0, :, int(target_label)]

        values_norm = _normalize_attributions(np.array(values)).tolist()
        html        = tokens_to_html(tokens, values_norm)

        return {
            "text": text,
            "predicted": {
                "label_id": pred_id,
                "label": LABEL_MAP[pred_id],
                "probabilities": {
                    "negative": float(probs[0]),
                    "neutral":  float(probs[1]),
                    "positive": float(probs[2]),
                },
                "confidence": float(probs[pred_id]),
            },
            "target_explained": {
                "label_id": int(target_label),
                "label": LABEL_MAP[int(target_label)],
            },
            "tokens":       tokens,
            "attributions": values_norm,
            "html":         html,
        }


if __name__ == "__main__":
    explainer    = FinBERTSHAPExplainer(SHAPConfig(max_evals=250))
    sample       = "Company shares rise after strong earnings report."
    out          = explainer.explain(sample)
    print("Predicted:", out["predicted"])
    pairs        = list(zip(out["tokens"], out["attributions"]))
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:10]
    print("Top tokens by |SHAP|:", pairs_sorted)
    
