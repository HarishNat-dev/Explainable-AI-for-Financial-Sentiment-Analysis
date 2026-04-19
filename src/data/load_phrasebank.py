# =============================================================================
# src/data/load_phrasebank.py
#
# Data pipeline — downloads the Financial PhraseBank dataset and produces
# the stratified train/validation/test splits used throughout the project.
#
# This is the first step in the data flow described in Section 4.2 and
# illustrated in Figure 5. Run this script once before training or evaluation.
#
# Output: three Parquet files written to src/data/processed/
#   - phrasebank_allagree_train.parquet  (70% — 1,569 sentences)
#   - phrasebank_allagree_val.parquet    (15% — 336 sentences)
#   - phrasebank_allagree_test.parquet   (15% — 340 sentences)
#
# Key design decisions documented here:
#   - sentences_allagree subset chosen for highest-confidence labels (Section 4.2)
#   - Stratified splitting preserves class ratios ±1% across all three splits
#     (Table 4, Section 4.2) — essential given 59% neutral class imbalance
#   - seed=42 fixed globally for reproducibility (NF2, Section 3.4.2)
#   - Parquet format loads ~10x faster than CSV and preserves dtypes (Section 4.2)
#
# Report: Section 4.2 — Data Pipeline
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class PhraseBankConfig:
    """
    Configuration for dataset loading and splitting.
    subset can be changed to 'sentences_75agree', 'sentences_66agree', or
    'sentences_50agree' to test on more ambiguous data — recommended as
    future work in Section 8.5.
    """
    dataset_name: str = "takala/financial_phrasebank"
    subset: str = "sentences_allagree"
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42


# Integer label to string mapping matching FinBERT's output head order:
# 0=negative, 1=neutral, 2=positive (consistent across all scripts)
LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def load_phrasebank(cfg: PhraseBankConfig) -> pd.DataFrame:
    """
    Downloads the Financial PhraseBank from Hugging Face using the
    takala/financial_phrasebank dataset identifier (Section 4.2).
    Returns a DataFrame with columns: sentence, label, label_text.
    Raw dataset contains 2,264 sentences across three sentiment classes
    with ~59% neutral, ~28% positive, ~13% negative distribution.
    """
    ds = load_dataset(cfg.dataset_name, cfg.subset, trust_remote_code=True)
    df = pd.DataFrame(ds["train"])
    # expected columns from Hugging Face: "sentence", "label"
    df["label_text"] = df["label"].map(LABEL_MAP)
    return df


def make_splits(df: pd.DataFrame, cfg: PhraseBankConfig):
    """
    Produces stratified 70/15/15 train/validation/test splits.

    Stratification on the label column is critical: without it, random
    sampling risks producing a test set with no negative-class examples
    given their 13% representation in the dataset (Section 4.2, Table 4).
    The two-step split (test first, then val from remaining train) ensures
    the val_size fraction is computed correctly relative to the full dataset.

    Returns: (train_df, val_df, test_df) — all reset to zero-based index.
    """
    # Step 1: split off held-out test set — never touched during training
    # or hyperparameter selection (strict holdout protocol, Table 4)
    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=df["label"],
    )

    # Step 2: split remaining data into train and validation
    # val_relative rescales val_size to the reduced pool after test removal
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

    # Save splits as Parquet — columnar format, ~10x faster to load than CSV,
    # preserves dtypes without re-parsing (Section 4.2)
    out_path = "src/data/processed/phrasebank_allagree.parquet"
    train_df.assign(split="train").to_parquet(out_path.replace(".parquet", "_train.parquet"), index=False)
    val_df.assign(split="val").to_parquet(out_path.replace(".parquet", "_val.parquet"), index=False)
    test_df.assign(split="test").to_parquet(out_path.replace(".parquet", "_test.parquet"), index=False)
    print(f"\nSaved splits to data/processed/")
    
