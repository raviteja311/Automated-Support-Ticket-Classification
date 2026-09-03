import re

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from ticket_triage.config import load_config
from ticket_triage.logger import get_logger

logger = get_logger(__name__)

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def main() -> None:
    cfg = load_config()
    df = pd.read_csv(cfg.data.raw_path)
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 0]
    train_df, test_df = train_test_split(
        df,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
        stratify=df["label"],
    )
    out_dir = Path(cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    logger.info("Train rows: %d | Test rows: %d", len(train_df), len(test_df))

if __name__ == "__main__":
    main()
