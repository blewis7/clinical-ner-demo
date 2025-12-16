import argparse
from typing import cast

import numpy as np
from datasets import load_from_disk, Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

DEFAULT_MODEL = "bert-base-cased"
SECTION_LABELS = ["HPI", "MEDS", "ASSESSMENT", "PLAN"]


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed/sections")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out_dir", type=str, default="outputs/sections")
    args = ap.parse_args()

    ds = load_from_disk(args.data_dir)
    ds = cast(DatasetDict, ds)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tok(batch):
        # You can optionally set max_length=128 for speed:
        # return tokenizer(batch["text"], truncation=True, max_length=128)
        return tokenizer(batch["text"], truncation=True)

    tokenized = ds.map(tok, batched=True)
    tokenized = cast(DatasetDict, tokenized)

    # Don't force torch format here; let the collator pad dynamically.
    # tokenized.set_format("torch")

    train_ds = cast(Dataset, tokenized["train"])
    val_ds = cast(Dataset, tokenized["validation"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(SECTION_LABELS),
        id2label={i: s for i, s in enumerate(SECTION_LABELS)},
        label2id={s: i for i, s in enumerate(SECTION_LABELS)},
    )

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",   # Transformers 4.57+ uses eval_strategy
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved section model to: {args.out_dir}")


if __name__ == "__main__":
    main()
