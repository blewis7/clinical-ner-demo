import argparse
from collections import Counter
from typing import cast

import numpy as np
from datasets import load_from_disk, Dataset, DatasetDict
from seqeval.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

DEFAULT_MODEL = "bert-base-cased"


def build_label_maps(dataset: DatasetDict):
    """
    Build label maps from string BIO tags in the dataset.
    Ensures 'O' is index 0.
    """
    tags = []
    for split in ["train", "validation"]:
        for tags_in_row in dataset[split]["ner_tags"]:
            tags.extend(tags_in_row)


    # Keep O first, then the rest sorted for deterministic ordering
    label_list = sorted(set(tags) - {"O"})
    label_list = ["O"] + label_list

    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}
    return label_list, id2label, label2id


def tokenize_and_align_labels(tokenizer, examples, label2id):
    """
    Tokenize pre-split tokens and align word-level BIO labels to subword tokens.
    - First subword of each word gets the label
    - Subsequent subwords get -100 (ignored by loss/metrics)
    """
    tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev:
                label_ids.append(label2id[tags[word_id]])
            else:
                label_ids.append(-100)
            prev = word_id

        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


def compute_metrics_factory(id2label):
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        labels = p.label_ids

        true_preds, true_labels = [], []
        for pred, lab in zip(preds, labels):
            cur_preds, cur_labels = [], []
            for p_i, l_i in zip(pred, lab):
                if l_i == -100:
                    continue
                cur_preds.append(id2label[int(p_i)])
                cur_labels.append(id2label[int(l_i)])
            true_preds.append(cur_preds)
            true_labels.append(cur_labels)

        return {"f1": f1_score(true_labels, true_preds)}

    return compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed/ner")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out_dir", type=str, default="outputs/ner")
    args = ap.parse_args()

    # Load HF dataset
    ds = load_from_disk(args.data_dir)
    ds = cast(DatasetDict, ds)

    label_list, id2label, label2id = build_label_maps(ds)

    # Quick label distribution peek
    c = Counter()
    for tags in ds["train"]["ner_tags"]:
        c.update(tags)
    print("Top tag counts (train):", c.most_common(12))
    print("Num labels:", len(label_list))

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    tokenized = ds.map(
        lambda ex: tokenize_and_align_labels(tokenizer, ex, label2id),
        batched=True,
    )
    tokenized = cast(DatasetDict, tokenized)

    # Optional: remove original columns to avoid any tensor formatting surprises
    # (Trainer only needs the tokenized inputs + labels)
    cols_to_remove = []
    for col in ["tokens", "ner_tags"]:
        if col in tokenized["train"].column_names:
            cols_to_remove.append(col)
    if cols_to_remove:
        tokenized = tokenized.remove_columns(cols_to_remove)

    # Explicitly set torch format to be safe across environments/versions
    tokenized.set_format("torch")

    train_ds = cast(Dataset, tokenized["train"])
    val_ds = cast(Dataset, tokenized["validation"])

    data_collator = DataCollatorForTokenClassification(tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
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
        compute_metrics=compute_metrics_factory(id2label),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    # Save final model + tokenizer
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved NER model to: {args.out_dir}")


if __name__ == "__main__":
    main()
