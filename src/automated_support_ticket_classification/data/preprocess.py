import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from automated_support_ticket_classification.config import load_config
from automated_support_ticket_classification.logger import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_frame(
    df: pd.DataFrame, split: str, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) by the official split column or a stratified re-split."""
    if split == "official":
        if "split" not in df.columns:
            raise ValueError(
                "data.split is 'official' but the raw data has no split column; "
                "only banking77 ships one. Use data.split: random for synthetic data."
            )
        train_df = df[df["split"] == "train"].drop(columns="split")
        test_df = df[df["split"] == "test"].drop(columns="split")
        return train_df, test_df
    if split == "random":
        # The official column, if present, means nothing after a re-split.
        df = df.drop(columns="split", errors="ignore")
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["label"],
        )
    raise ValueError(f"Unknown data.split {split!r}; expected 'official' or 'random'")


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(cfg.data.raw_path)
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 0]
    train_df, test_df = split_frame(df, cfg.data.split, cfg.data.test_size, cfg.data.random_state)
    out_dir = Path(cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    logger.info("Train rows: %d | Test rows: %d", len(train_df), len(test_df))


if __name__ == "__main__":
    main()
