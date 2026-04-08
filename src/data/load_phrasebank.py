from __future__ import annotations

from dataclasses import dataclass
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class PhraseBankConfig:
    dataset_name: str = "takala/financial_phrasebank"
    subset: str = "sentences_allagree"   # you can change to sentences_75agree later
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42


LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def load_phrasebank(cfg: PhraseBankConfig) -> pd.DataFrame:
    ds = load_dataset(cfg.dataset_name, cfg.subset, trust_remote_code=True)
    df = pd.DataFrame(ds["train"])
    # expected columns: "sentence", "label"
    df["label_text"] = df["label"].map(LABEL_MAP)
    return df


def make_splits(df: pd.DataFrame, cfg: PhraseBankConfig):
    # First split off test
    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=df["label"],
    )

    # Then split train into train/val
    val_relative = cfg.val_size / (1.0 - cfg.test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_relative,
        random_state=cfg.seed,
        stratify=train_df["label"],
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    cfg = PhraseBankConfig()
    df = load_phrasebank(cfg)
    train_df, val_df, test_df = make_splits(df, cfg)

    print("Total:", len(df))
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
    print("\nLabel distribution (total):")
    print(df["label_text"].value_counts(normalize=True).round(3))

    out_path = "src/data/processed/phrasebank_allagree.parquet"
    train_df.assign(split="train").to_parquet(out_path.replace(".parquet", "_train.parquet"), index=False)
    val_df.assign(split="val").to_parquet(out_path.replace(".parquet", "_val.parquet"), index=False)
    test_df.assign(split="test").to_parquet(out_path.replace(".parquet", "_test.parquet"), index=False)
    print(f"\nSaved splits to data/processed/")
