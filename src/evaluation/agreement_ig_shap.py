from __future__ import annotations

import sys
from pathlib import Path

# --- Make project root importable ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from dataclasses import dataclass
from typing import List, Set

import pandas as pd

from src.inference.predict import FinBERTPredictor, InferenceConfig
from src.xai.ig_explainer import FinBERTIGExplainer, IGConfig
from src.xai.shap_explainer import FinBERTSHAPExplainer, SHAPConfig


@dataclass(frozen=True)
class AgreementConfig:
    test_path: str = "data/processed/phrasebank_allagree_test.parquet"
    text_col: str = "sentence"
    max_rows: int = 200
    top_k: int = 8
    out_dir: str = "reports/evaluation"
    out_csv: str = "agreement_ig_shap.csv"


def normalize(tok: str) -> str:
    return tok.replace("##", "").strip().lower()


def topk_from_out(tokens: List[str], scores: List[float], k: int) -> List[str]:
    pairs = [(t, float(s)) for t, s in zip(tokens, scores) if str(t).strip() != ""]
    pairs = [(t, s) for t, s in pairs if normalize(t)]
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:k]
    return [normalize(t) for t, _ in pairs_sorted]


def overlap_k(a: List[str], b: List[str], k: int) -> float:
    sa: Set[str] = set(a)
    sb: Set[str] = set(b)
    if k <= 0:
        return 0.0
    return len(sa.intersection(sb)) / float(k)


def main():
    cfg = AgreementConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.test_path).head(cfg.max_rows).copy()

    predictor = FinBERTPredictor(InferenceConfig())
    ig = FinBERTIGExplainer(IGConfig(n_steps=30))
    shapx = FinBERTSHAPExplainer(SHAPConfig(max_evals=200))

    rows = []
    for idx, row in df.iterrows():
        text = str(row[cfg.text_col])
        pred = predictor.predict_one(text)
        label = pred["label"]
        conf = float(pred["confidence"])

        ig_out = ig.explain(text)
        ig_top = topk_from_out(ig_out["tokens"], ig_out["attributions"], cfg.top_k)

        shap_out = shapx.explain(text)
        shap_top = topk_from_out(shap_out["tokens"], shap_out["attributions"], cfg.top_k)

        agree = overlap_k(ig_top, shap_top, cfg.top_k)

        rows.append({
            "idx": int(idx),
            "orig_label": label,
            "confidence": conf,
            "ig_top_tokens": ", ".join(ig_top),
            "shap_top_tokens": ", ".join(shap_top),
            "topk_overlap": agree,
        })

    res = pd.DataFrame(rows)
    out_csv = out_dir / cfg.out_csv
    res.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)

    print("\nAgreement summary (top-k overlap):")
    print(res["topk_overlap"].describe())

    print("\nAgreement by predicted class:")
    print(res.groupby("orig_label")["topk_overlap"].describe())


if __name__ == "__main__":
    main()
