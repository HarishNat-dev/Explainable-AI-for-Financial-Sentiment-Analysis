from __future__ import annotations

import sys
from pathlib import Path

# --- Make project root importable ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
from dataclasses import dataclass
from typing import List, Set

import pandas as pd

from src.inference.predict import FinBERTPredictor, InferenceConfig
from src.xai.ig_explainer import FinBERTIGExplainer, IGConfig


@dataclass(frozen=True)
class RobustCfg:
    test_path: str = "data/processed/phrasebank_allagree_test.parquet"
    text_col: str = "sentence"
    max_rows: int = 200
    top_k: int = 5
    out_dir: str = "reports/evaluation"
    out_csv: str = "robustness_xai.csv"


STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","when","while",
    "to","of","in","on","for","with","at","by","from","as","is","are",
    "was","were","be","been","being","it","its","this","that","these","those",
    "has","have","had","will","would","can","could","may","might","should",
}

def _clean_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text

def perturb_remove_stopwords(text: str) -> str:
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    kept = []
    for tok in tokens:
        if tok.isalpha() and tok.lower() in STOPWORDS:
            continue
        kept.append(tok)
    return _clean_spaces(" ".join(kept))

def normalize(tok: str) -> str:
    return tok.replace("##","").strip().lower()

def topk_ig(ig_out: dict, k: int) -> List[str]:
    pairs = [(t, float(a)) for t, a in zip(ig_out["tokens"], ig_out["attributions"]) if str(t).strip()!=""]
    pairs = [(normalize(t), a) for t, a in pairs if normalize(t)]
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:k]
    return [t for t,_ in pairs_sorted]

def overlap(a: List[str], b: List[str], k: int) -> float:
    sa: Set[str] = set(a)
    sb: Set[str] = set(b)
    return len(sa.intersection(sb))/float(k) if k else 0.0

def main():
    cfg = RobustCfg()
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.test_path).head(cfg.max_rows).copy()

    predictor = FinBERTPredictor(InferenceConfig())
    ig = FinBERTIGExplainer(IGConfig(n_steps=30))

    rows = []
    for idx, row in df.iterrows():
        text = str(row[cfg.text_col])

        pred0 = predictor.predict_one(text)
        label0 = pred0["label"]
        conf0 = float(pred0["confidence"])
        ig0 = topk_ig(ig.explain(text), cfg.top_k)

        pert = perturb_remove_stopwords(text)
        pred1 = predictor.predict_one(pert)
        label1 = pred1["label"]
        conf1 = float(pred1["confidence"])
        ig1 = topk_ig(ig.explain(pert), cfg.top_k)

        flip = int(label0 != label1)
        xai_change = 1.0 - overlap(ig0, ig1, cfg.top_k)

        rows.append({
            "idx": int(idx),
            "orig_label": label0,
            "orig_conf": conf0,
            "pert_label": label1,
            "pert_conf": conf1,
            "label_flip": flip,
            "topk_overlap": 1.0 - xai_change,
            "xai_change": xai_change,
            "orig_top_tokens": ", ".join(ig0),
            "pert_top_tokens": ", ".join(ig1),
        })

    res = pd.DataFrame(rows)
    out_csv = out_dir / cfg.out_csv
    res.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)

    print("\nFlip rate:", res["label_flip"].mean())
    print("\nXAI change summary (1 - overlap):")
    print(res["xai_change"].describe())

    print("\nXAI change when flip vs no flip:")
    print(res.groupby("label_flip")["xai_change"].describe())

    worst = res.sort_values("xai_change", ascending=False).iloc[0]
    print("\n--- Max XAI Change Case ---")
    print(worst[["orig_label","pert_label","label_flip","orig_conf","pert_conf","orig_top_tokens","pert_top_tokens","xai_change"]])


if __name__ == "__main__":
    main()
