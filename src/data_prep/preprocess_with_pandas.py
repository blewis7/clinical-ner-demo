import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTION_LABELS = ["HPI", "MEDS", "ASSESSMENT", "PLAN"]
SECTION2ID = {s: i for i, s in enumerate(SECTION_LABELS)}


def load_notes_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        note = json.loads(line)
        for ex in note["lines"]:
            rows.append(
                {
                    "note_id": ex["note_id"],
                    "line_id": ex["line_id"],
                    "section": ex["section_label"],
                    "text": ex["text"],
                    "tokens": ex["tokens"],
                    "ner_tags": ex["ner_tags"],
                }
            )
    return pd.DataFrame(rows)


def main():
    train_df = load_notes_jsonl(RAW_DIR / "train.jsonl")
    val_df = load_notes_jsonl(RAW_DIR / "validation.jsonl")
    test_df = load_notes_jsonl(RAW_DIR / "test.jsonl")

    # sanity checks
    print("\nSection distribution (train):")
    print(train_df["section"].value_counts())
    print("\nToken length stats (train):")
    print(train_df["tokens"].map(len).describe())

    for df in (train_df, val_df, test_df):
        df["section_label"] = df["section"].map(SECTION2ID)

    # NER dataset
    ner = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df[["tokens", "ner_tags"]].reset_index(drop=True)),
            "validation": Dataset.from_pandas(val_df[["tokens", "ner_tags"]].reset_index(drop=True)),
            "test": Dataset.from_pandas(test_df[["tokens", "ner_tags"]].reset_index(drop=True)),
        }
    )

    # Section dataset
    sections = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df[["text", "section_label"]].rename(columns={"section_label": "label"}).reset_index(drop=True)),
            "validation": Dataset.from_pandas(val_df[["text", "section_label"]].rename(columns={"section_label": "label"}).reset_index(drop=True)),
            "test": Dataset.from_pandas(test_df[["text", "section_label"]].rename(columns={"section_label": "label"}).reset_index(drop=True)),
        }
    )

    ner.save_to_disk(str(OUT_DIR / "ner"))
    sections.save_to_disk(str(OUT_DIR / "sections"))

    print(f"\nSaved: {OUT_DIR / 'ner'}")
    print(f"Saved: {OUT_DIR / 'sections'}")


if __name__ == "__main__":
    main()
