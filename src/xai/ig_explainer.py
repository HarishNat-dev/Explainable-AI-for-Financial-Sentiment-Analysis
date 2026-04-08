from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from captum.attr import IntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


@dataclass(frozen=True)
class IGConfig:
    model_dir: str = "models/finbert/finbert_phrasebank_allagree/best_model"
    max_length: int = 128
    n_steps: int = 30
    internal_batch_size: int = 8


def _normalize_attributions(attrs: np.ndarray) -> np.ndarray:
    if np.allclose(attrs, 0):
        return attrs
    max_abs = np.max(np.abs(attrs))
    return attrs / (max_abs + 1e-12)


def tokens_to_html(tokens: List[str], scores: List[float]) -> str:
    """
    Green = supports target class (positive attribution)
    Red   = opposes target class (negative attribution)
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
        clean = tok.replace("##", "")
        rendered.append(
            "<span style='"
            + style(sc)
            + " padding:2px 4px; margin:1px; border-radius:6px; display:inline-block;'"
            + f" title='{sc:.3f}'>"
            + clean
            + "</span>"
        )
    return "<div style='line-height: 2.2; font-size: 16px;'>" + " ".join(rendered) + "</div>"


class FinBERTIGExplainer:
    def __init__(self, cfg: IGConfig):
        self.cfg = cfg
        self.model_path = Path(cfg.model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # We'll attribute w.r.t. embeddings
        self.embedding_layer = self.model.get_input_embeddings()

        # Captum IG needs a forward function that accepts embeddings
        self.ig = IntegratedGradients(self._forward_from_embeds)

        # Precompute PAD token id for baseline
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            # BERT should have PAD, but just in case:
            self.pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")

    def _forward_from_embeds(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        """
        Forward pass using inputs_embeds (continuous) instead of input_ids (integers).
        Returns logits.
        """
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs.logits

    def explain(self, text: str, target_label: int | None = None) -> Dict:
        # Tokenize
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
            padding=False,
        )
        input_ids = enc["input_ids"].to(self.device)              # [1, seq_len] (long)
        attention_mask = enc["attention_mask"].to(self.device)    # [1, seq_len]

        # Predict
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
            pred_id = int(probs.argmax())

        if target_label is None:
            target_label = pred_id

        # Convert to embeddings (float) for IG
        inputs_embeds = self.embedding_layer(input_ids)  # [1, seq_len, hidden]

        # Baseline: same shape, all PAD embeddings
        baseline_ids = torch.full_like(input_ids, fill_value=self.pad_id)
        baseline_embeds = self.embedding_layer(baseline_ids)

        # Compute IG attributions on embeddings
        attributions = self.ig.attribute(
            inputs=inputs_embeds,
            baselines=baseline_embeds,
            additional_forward_args=(attention_mask,),
            target=int(target_label),
            n_steps=self.cfg.n_steps,
            internal_batch_size=self.cfg.internal_batch_size,
        )  # [1, seq_len, hidden]

        # Reduce hidden dim -> per-token score
        attrs = attributions.sum(dim=-1)[0].detach().cpu().numpy()  # [seq_len]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())

        # Remove special tokens for display
        keep = [i for i, t in enumerate(tokens) if t not in ("[CLS]", "[SEP]")]
        tokens_kept = [tokens[i] for i in keep]
        attrs_kept = attrs[keep]

        attrs_norm = _normalize_attributions(attrs_kept).tolist()
        html = tokens_to_html(tokens_kept, attrs_norm)

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
            "tokens": tokens_kept,
            "attributions": attrs_norm,
            "html": html,
        }


if __name__ == "__main__":
    explainer = FinBERTIGExplainer(IGConfig())
    sample = "Company shares rise after strong earnings report."
    out = explainer.explain(sample)

    print("Predicted:", out["predicted"])
    # show a few top absolute attributions
    pairs = list(zip(out["tokens"], out["attributions"]))
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:10]
    print("Top tokens by |attribution|:", pairs_sorted)
