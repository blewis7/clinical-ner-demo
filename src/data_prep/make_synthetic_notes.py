import argparse
import json
import random
import uuid
from pathlib import Path

random.seed(42)

MEDS = ["metformin", "lisinopril", "atorvastatin", "albuterol", "amoxicillin", "ibuprofen", "insulin"]
DOSES = ["5 mg", "10 mg", "20 mg", "500 mg", "1 g", "2 puffs", "10 units"]
ROUTES = ["PO", "IV", "IM", "inhaled"]
FREQS = ["daily", "BID", "TID", "qHS", "q4h PRN"]

# "Conditions" we want to label as CONDITION in NER
CONDS = ["hypertension", "type 2 diabetes", "asthma", "pneumonia", "hyperlipidemia", "back pain"]

# Symptoms: what patients report in HPI (NOT necessarily a diagnosis)
SYMPTOMS = [
    "fatigue",
    "polyuria",
    "polydipsia",
    "wheezing",
    "shortness of breath",
    "productive cough",
    "fever",
    "chest tightness",
    "low back pain",
    "headache",
]

PROCS = ["appendectomy", "colonoscopy", "knee arthroscopy", "CT scan", "chest X-ray", "blood culture"]
DURS = ["2 days", "1 week", "3 months", "since yesterday", "for 6 hours"]

# Simple mapping from symptom patterns -> likely diagnosis (purely synthetic, but more realistic)
SYMPTOM_TO_DX = [
    ({"wheezing", "shortness of breath", "chest tightness"}, "asthma"),
    ({"productive cough", "fever", "shortness of breath"}, "pneumonia"),
    ({"polyuria", "polydipsia", "fatigue"}, "type 2 diabetes"),
    ({"low back pain"}, "back pain"),
]

SECTION_ORDER = ["HPI", "MEDS", "ASSESSMENT", "PLAN"]


def tokenize(text: str):
    text = text.replace(".", " .").replace(":", " :").replace(",", " ,")
    return text.split()


def tag_span(tags, start_idx, end_idx, label):
    if start_idx < 0 or end_idx > len(tags) or start_idx >= end_idx:
        return tags
    tags[start_idx] = f"B-{label}"
    for i in range(start_idx + 1, end_idx):
        tags[i] = f"I-{label}"
    return tags


def find_sublist(tokens, phrase_tokens):
    for i in range(len(tokens) - len(phrase_tokens) + 1):
        if tokens[i : i + len(phrase_tokens)] == phrase_tokens:
            return i
    return -1


def pick_symptoms():
    # pick 1-3 symptoms
    k = random.choice([1, 2, 3])
    return random.sample(SYMPTOMS, k=k)


def infer_diagnosis_from_symptoms(symptoms):
    s = set(symptoms)
    for pattern, dx in SYMPTOM_TO_DX:
        if pattern.issubset(s) or len(pattern.intersection(s)) >= max(1, len(pattern) - 1):
            return dx
    # fallback: random condition
    return random.choice(CONDS)


def build_note():
    med = random.choice(MEDS)
    dose = random.choice(DOSES)
    route = random.choice(ROUTES)
    freq = random.choice(FREQS)

    symptoms = pick_symptoms()
    dur = random.choice(DURS)

    # assessment diagnosis is correlated with symptoms (more realistic)
    assess_cond = infer_diagnosis_from_symptoms(symptoms)

    # optional: add a chronic history condition sometimes
    history_cond = random.choice(CONDS) if random.random() < 0.4 else None

    # procedure choice loosely linked (still synthetic)
    proc = random.choice(PROCS)

    note_id = str(uuid.uuid4())

    # keep all 4 sections; optionally shuffle for robustness
    sections = SECTION_ORDER[:]
    random.shuffle(sections)

    lines = []
    for i, sec in enumerate(sections):
        if sec == "HPI":
            # Patient reports symptoms, not a diagnosis
            sym_text = ", ".join(symptoms)
            if history_cond:
                line_body = (
                    f"Patient reports {sym_text} for {dur}. Past history includes {history_cond}. "
                    "Denies chest pain."
                )
            else:
                line_body = f"Patient reports {sym_text} for {dur}. Denies chest pain. No fever."
            line_text = f"{sec}: {line_body}"

        elif sec == "MEDS":
            line_text = f"{sec}: Started {med} {dose} {route} {freq}."

        elif sec == "ASSESSMENT":
            # Make assessment feel like clinical thinking
            style = random.random()
            if style < 0.45:
                line_body = f"Likely {assess_cond}. Consider {proc} to evaluate."
            elif style < 0.75:
                alt = random.choice([c for c in CONDS if c != assess_cond])
                line_body = f"Assessment: {assess_cond}. Differential includes {alt}. Consider {proc}."
            else:
                line_body = f"Assessment: rule out {assess_cond}. Will monitor and reassess."
            line_text = f"{sec}: {line_body}"

        else:  # PLAN
            style = random.random()
            if style < 0.6:
                line_text = f"{sec}: Continue {med} {dose} {route} {freq}. Follow up in 2 weeks."
            else:
                line_text = f"{sec}: Order {proc}. Provide return precautions."

        tokens = tokenize(line_text)
        tags = ["O"] * len(tokens)

        # Tag entities (keep your original labels)
        # NOTE: we tag assess_cond and history_cond as CONDITION; symptoms are not tagged (by design).
        condition_phrases = [assess_cond]
        if history_cond:
            condition_phrases.append(history_cond)

        for phrase, label in [
            (med, "MED"),
            (dose, "DOSE"),
            (route, "ROUTE"),
            (freq, "FREQ"),
            (proc, "PROCEDURE"),
        ]:
            phrase_tokens = tokenize(phrase)
            idx = find_sublist(tokens, phrase_tokens)
            if idx != -1:
                tags = tag_span(tags, idx, idx + len(phrase_tokens), label)

        for cond in condition_phrases:
            cond_tokens = tokenize(cond)
            idx = find_sublist(tokens, cond_tokens)
            if idx != -1:
                tags = tag_span(tags, idx, idx + len(cond_tokens), "CONDITION")

        lines.append(
            {
                "note_id": note_id,
                "line_id": i,
                "section_label": sec,
                "text": line_text,
                "tokens": tokens,
                "ner_tags": tags,
            }
        )

    full_text = "\n".join([x["text"] for x in lines])
    return {"note_id": note_id, "full_text": full_text, "lines": lines}


def write_jsonl(path: Path, items):
    path.write_text("\n".join(json.dumps(x) for x in items), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--out_dir", type=str, default="data/raw")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = [build_note() for _ in range(args.train)]
    val = [build_note() for _ in range(args.val)]
    test = [build_note() for _ in range(args.test)]

    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "validation.jsonl", val)
    write_jsonl(out / "test.jsonl", test)

    print(f"Wrote synthetic notes to: {out.resolve()}")
    print(
        f"Notes: train={args.train}, val={args.val}, test={args.test} "
        f"(lines: train={args.train*4}, val={args.val*4}, test={args.test*4})"
    )
    print("\nExample note:\n")
    print(train[0]["full_text"])


if __name__ == "__main__":
    main()
