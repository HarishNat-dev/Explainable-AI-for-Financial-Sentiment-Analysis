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
    model_dir: str = "models/finbert/finbert_phrasebank_allagree/best_model"
    max_length: int = 128
    # Controls SHAP runtime. Increase for smoother explanations; decrease for speed.
    max_evals: int = 300


def _normalize_attributions(attrs: np.ndarray) -> np.ndarray:
    if np.allclose(attrs, 0):
        return attrs
    max_abs = np.max(np.abs(attrs))
    return attrs / (max_abs + 1e-12)


def tokens_to_html(tokens: List[str], scores: List[float]) -> str:
    """
    Green = supports target class (positive SHAP value)
    Red   = opposes target class (negative SHAP value)
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
        clean = str(tok)
        rendered.append(
            "<span style='"
            + style(sc)
            + " padding:2px 4px; margin:1px; border-radius:6px; display:inline-block;'"
            + f" title='{sc:.3f}'>"
            + clean
            + "</span>"
        )
    return "<div style='line-height: 2.2; font-size: 16px;'>" + " ".join(rendered) + "</div>"


class FinBERTSHAPExplainer:
    """
    SHAP text explainer using a token-aware masker driven by the tokenizer.
    Works for multiclass by selecting a target class (default: predicted class).
    """

    def __init__(self, cfg: SHAPConfig):
        self.cfg = cfg
        self.model_path = Path(cfg.model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # SHAP masker uses the tokenizer to create meaningful masks
        self.masker = shap.maskers.Text(self.tokenizer)

        # Build an explainer once (expensive) and reuse it
        self.explainer = shap.Explainer(
            self._predict_proba_np,
            self.masker,
            algorithm="partition",  # fewer model calls than kernel for text
            output_names=[LABEL_MAP[i] for i in range(3)],
        )

    @torch.no_grad()
    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
            padding=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()

    def _predict_proba_np(self, texts) -> np.ndarray:
        """
        SHAP may pass:
          - list[str]
          - numpy array of strings
          - list of token lists (pretokenized)
          - nested arrays
        We normalize everything to List[str].
        """
        # Convert numpy -> python list
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()

        # Ensure list
        if not isinstance(texts, list):
            texts = [texts]

        cleaned: List[str] = []
        for t in texts:
            # If SHAP gives a token list, join tokens into a string
            if isinstance(t, list):
                # For typical token lists, join without spaces (masker returns tokens including spaces)
                if all(isinstance(x, str) for x in t):
                    cleaned.append("".join(t))
                else:
                    cleaned.append(str(t))
            else:
                cleaned.append(str(t))

        return self._predict_proba(cleaned)

    def explain(self, text: str, target_label: int | None = None) -> Dict:
        # Prediction
        probs = self._predict_proba([text])[0]
        pred_id = int(probs.argmax())
        if target_label is None:
            target_label = pred_id

        # SHAP explanation
        # For multiclass: exp.values shape is (1, num_tokens, num_classes)
        exp = self.explainer([text], max_evals=self.cfg.max_evals)

        tokens = list(exp.data[0])  # token strings from the masker
        values = exp.values[0, :, int(target_label)]  # SHAP values for chosen class

        values_norm = _normalize_attributions(np.array(values)).tolist()
        html = tokens_to_html(tokens, values_norm)

        return {
            "text": text,
            "predicted": {
                "label_id": pred_id,
                "label": LABEL_MAP[pred_id],
                "probabilities": {
                    "negative": float(probs[0]),
                    "neutral": float(probs[1]),
                    "positive": float(probs[2]),
                },
                "confidence": float(probs[pred_id]),
            },
            "target_explained": {
                "label_id": int(target_label),
                "label": LABEL_MAP[int(target_label)],
            },
            "tokens": tokens,
            "attributions": values_norm,
            "html": html,
        }


if __name__ == "__main__":
    explainer = FinBERTSHAPExplainer(SHAPConfig(max_evals=250))
    sample = "Company shares rise after strong earnings report."
    out = explainer.explain(sample)
    print("Predicted:", out["predicted"])
    pairs = list(zip(out["tokens"], out["attributions"]))
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:10]
    print("Top tokens by |SHAP|:", pairs_sorted)
