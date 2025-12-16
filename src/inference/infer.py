import argparse
from typing import List, Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline,
)

SECTION_LABELS = ["HPI", "MEDS", "ASSESSMENT", "PLAN"]


def normalize_section_label(raw_label: str) -> str:
    if raw_label.startswith("LABEL_"):
        idx = int(raw_label.split("_")[1])
        if 0 <= idx < len(SECTION_LABELS):
            return SECTION_LABELS[idx]
    return raw_label


def clean_wordpiece(text: str) -> str:
    """
    Merge WordPiece artifacts for display:
    - "al ##but ##ero ##l" -> "albuterol"
    - "chest X - ray" -> "chest X-ray"
    - remove doubled spaces
    """
    if not text:
        return text
    text = text.replace(" ##", "")
    text = text.replace(" - ", "-")
    text = " ".join(text.split())
    return text.strip()


def run_section_classifier(sec_pipe, line: str) -> Dict[str, Any]:
    out = sec_pipe(line, truncation=True)
    label = normalize_section_label(out[0]["label"])
    score = float(out[0].get("score", 0.0))
    return {"label": label, "score": score}


def run_ner(ner_pipe, line: str) -> List[Dict[str, Any]]:
    return ner_pipe(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ner_dir", type=str, default="outputs/ner")
    ap.add_argument("--sec_dir", type=str, default="outputs/sections")
    ap.add_argument("--text", type=str, default=None, help="Optional text input. If omitted, uses a built-in example.")
    ap.add_argument("--min_score", type=float, default=0.70, help="Minimum confidence to display an entity.")
    ap.add_argument("--show_all", action="store_true", help="Show all entities (ignores --min_score).")
    args = ap.parse_args()

    text = args.text or (
        "HPI: Patient reports wheezing, shortness of breath for 2 days. Denies chest pain.\n"
        "MEDS: Started albuterol 2 puffs inhaled q4h PRN.\n"
        "ASSESSMENT: Likely asthma. Consider chest X-ray to evaluate.\n"
        "PLAN: Continue albuterol 2 puffs inhaled q4h PRN. Follow up in 2 weeks."
    )

    # Section classifier
    sec_tok = AutoTokenizer.from_pretrained(args.sec_dir)
    sec_model = AutoModelForSequenceClassification.from_pretrained(args.sec_dir)
    sec_pipe = pipeline("text-classification", model=sec_model, tokenizer=sec_tok)

    # NER
    ner_tok = AutoTokenizer.from_pretrained(args.ner_dir)
    ner_model = AutoModelForTokenClassification.from_pretrained(args.ner_dir)
    ner_pipe = pipeline(
        "token-classification",
        model=ner_model,
        tokenizer=ner_tok,
        aggregation_strategy="simple",
    )

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    print("\nINPUT:\n" + text + "\n")

    for line in lines:
        sec = run_section_classifier(sec_pipe, line)
        ents = run_ner(ner_pipe, line)

        # sort entities by position
        ents = sorted(ents, key=lambda e: (e.get("start", 0), e.get("end", 0)))

        # filter out low confidence fragments unless show_all is set
        if not args.show_all:
            ents = [e for e in ents if float(e.get("score", 0.0)) >= args.min_score]

        print("=" * 80)
        print(f"LINE: {line}")
        print(f"SECTION_PRED: {sec['label']} (score={sec['score']:.3f})")
        print("ENTITIES:")

        if not ents:
            print("  (none)")
            continue

        for e in ents:
            ent_type = e.get("entity_group", "UNKNOWN")
            word = clean_wordpiece(e.get("word", ""))
            score = float(e.get("score", 0.0))
            print(f"  - {ent_type:12s} | {word:25s} | score={score:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
