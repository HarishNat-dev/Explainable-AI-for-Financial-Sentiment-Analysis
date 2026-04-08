from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


@dataclass(frozen=True)
class InferenceConfig:
    model_dir: str = "models/finbert/finbert_phrasebank_allagree/best_model"
    max_length: int = 128


class FinBERTPredictor:
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.model_path = Path(cfg.model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.eval()

        # CPU by default; if you later have GPU, this still works
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def predict_proba(self, texts: List[str]) -> torch.Tensor:
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
        probs = self.predict_proba([text])[0].numpy()
        pred_id = int(probs.argmax())
        return {
            "text": text,
            "label_id": pred_id,
            "label": LABEL_MAP[pred_id],
            "probabilities": {
                "negative": float(probs[0]),
                "neutral": float(probs[1]),
                "positive": float(probs[2]),
            },
            "confidence": float(probs[pred_id]),
        }


if __name__ == "__main__":
    predictor = FinBERTPredictor(InferenceConfig())
    sample = "Company shares rise after strong earnings report."
    print(predictor.predict_one(sample))
