# =============================================================================
# src/inference/predict.py
#
# FinBERT inference wrapper — loads the fine-tuned model and provides
# prediction methods used by the Streamlit dashboard and all evaluation scripts.
#
# Two public methods:
#   predict_proba(texts)  — batched, returns raw softmax probabilities [N, 3]
#   predict_one(text)     — single sentence, returns full prediction dict
#
# The prediction dict returned by predict_one() is the standard format
# consumed throughout the codebase: label, label_id, confidence, and the
# full three-class probability distribution. Displaying all three probabilities
# is a deliberate design decision (Section 4.7.1) so analysts can calibrate
# trust rather than treating output as a binary verdict.
#
# Label order: 0=negative, 1=neutral, 2=positive (consistent across all scripts)
# Model: fine-tuned ProsusAI/FinBERT checkpoint (Section 4.3.1)
# Truncation: max_length=128 tokens — sufficient for >95% of PhraseBank
#             sentences; silent truncation for longer inputs is a documented
#             usability limitation (T10, Section 5.2, Table 7)
#
# Report: Section 4.3.1, Section 4.7.1, Section 4.7.5
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Label mapping consistent with FinBERT output head order across all scripts
LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


@dataclass(frozen=True)
class InferenceConfig:
    """
    Points to the best model checkpoint saved by finetune_finbert.py.
    max_length=128 matches the training configuration (Table 5, Section 4.3.2).
    """
    model_dir: str = "models/finbert/finbert_phrasebank_allagree/best_model"
    max_length: int = 128


class FinBERTPredictor:
    """
    Wraps the fine-tuned FinBERT model for inference.
    Loaded once at startup via @st.cache_resource in streamlit_app.py —
    keeping the 440 MB weights in memory rather than reloading on every
    request (Section 4.7.5). Also used directly by all evaluation scripts.
    """

    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.model_path = Path(cfg.model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.eval()

        # CPU-first; automatically uses GPU if available without code changes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def predict_proba(self, texts: List[str]) -> torch.Tensor:
        """
        Batched prediction — tokenises all texts together and runs a single
        forward pass. Returns softmax probabilities as a CPU tensor of
        shape [N, 3] where columns are [negative, neutral, positive].

        Used by run_batch_predictions() in batch_utils.py for CSV processing
        and by evaluation scripts that need raw probabilities for metric computation.
        """
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
        return probs.cpu()

    def predict_one(self, text: str) -> Dict:
        """
        Single-sentence prediction — convenience wrapper around predict_proba.
        Returns the standard prediction dict used throughout the codebase:
          - label:         string class name (negative/neutral/positive)
          - label_id:      integer class index (0/1/2)
          - confidence:    softmax probability of the winning class
          - probabilities: full three-class distribution (sums to 1.0)

        The full probability distribution is returned (not just the winner)
        so the dashboard can display all three values and analysts can
        calibrate trust accordingly (Section 4.7.1). Verified to sum to 1.0
        in test T1 (Section 5.1, Table 7).
        """
        probs = self.predict_proba([text])[0].numpy()
        pred_id = int(probs.argmax())
        return {
            "text": text,
            "label_id": pred_id,
            "label": LABEL_MAP[pred_id],
            "probabilities": {
                "negative": float(probs[0]),
                "neutral":  float(probs[1]),
                "positive": float(probs[2]),
            },
            "confidence": float(probs[pred_id]),
        }


if __name__ == "__main__":
    predictor = FinBERTPredictor(InferenceConfig())
    sample = "Company shares rise after strong earnings report."
    print(predictor.predict_one(sample))
    
