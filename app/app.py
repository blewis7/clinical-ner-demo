import html
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import streamlit as st
import streamlit.components.v1 as components
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline,
)

SECTION_LABELS = ["HPI", "MEDS", "ASSESSMENT", "PLAN"]

ENTITY_COLORS = {
    "MED": "#90CAF9",        # blue
    "DOSE": "#EF9A9A",       # red
    "ROUTE": "#A5D6A7",      # green
    "FREQ": "#CE93D8",       # purple
    "CONDITION": "#FFCC80",  # orange
    "PROCEDURE": "#FFF59D",  # yellow
}


@dataclass
class EntitySpan:
    entity_group: str
    text: str        # cleaned display text
    score: float
    start: int
    end: int


def clean_wordpiece(text: str) -> str:
    """Merge WordPiece-ish artifacts for display."""
    if not text:
        return text
    text = text.replace(" ##", "")
    text = text.replace(" - ", "-")
    text = " ".join(text.split())
    return text.strip()


def can_merge_between(line: str, end_a: int, start_b: int) -> bool:
    """
    Allow merging if the characters between spans are ignorable.
    This handles cases like:
      - within a word: "" (empty)
      - whitespace: " "
      - punctuation tokenization: "-" or "/" etc.
    """
    if start_b < end_a:
        # overlapping spans -> treat as mergeable
        return True

    between = line[end_a:start_b]
    if between == "":
        return True

    # merge if between is only whitespace or light punctuation
    ignorable = set([" ", "\t", "\n", "-", "/", ".", ","])
    return all(ch in ignorable for ch in between)


def merge_adjacent_spans(spans: List[EntitySpan], line: str) -> List[EntitySpan]:
    """
    Merge spans that are adjacent/nearby AND have the same entity type,
    where the gap between them is only whitespace/punctuation.
    """
    if not spans:
        return spans

    spans = sorted(spans, key=lambda s: (s.start, s.end))
    merged: List[EntitySpan] = [spans[0]]

    for s in spans[1:]:
        last = merged[-1]

        if (
            s.entity_group == last.entity_group
            and can_merge_between(line, last.end, s.start)
        ):
            new_start = min(last.start, s.start)
            new_end = max(last.end, s.end)

            # Take the actual substring from the original line (best truth)
            new_text = clean_wordpiece(line[new_start:new_end])

            # Keep the max score (simple + stable)
            new_score = max(last.score, s.score)

            merged[-1] = EntitySpan(
                entity_group=last.entity_group,
                text=new_text,
                score=new_score,
                start=new_start,
                end=new_end,
            )
        else:
            merged.append(s)

    return merged


@st.cache_resource
def load_pipelines(ner_dir: str, sec_dir: str):
    # Section classifier
    sec_tok = AutoTokenizer.from_pretrained(sec_dir)
    sec_model = AutoModelForSequenceClassification.from_pretrained(sec_dir)
    sec_pipe = pipeline("text-classification", model=sec_model, tokenizer=sec_tok)

    # NER (aggregation_strategy helps but we still post-process)
    ner_tok = AutoTokenizer.from_pretrained(ner_dir)
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_dir)
    ner_pipe = pipeline(
        "token-classification",
        model=ner_model,
        tokenizer=ner_tok,
        aggregation_strategy="simple",
    )
    return ner_pipe, sec_pipe


def normalize_section_label(raw_label: str) -> str:
    if raw_label.startswith("LABEL_"):
        idx = int(raw_label.split("_")[1])
        if 0 <= idx < len(SECTION_LABELS):
            return SECTION_LABELS[idx]
    return raw_label


def predict_section(sec_pipe, line: str) -> Tuple[str, float]:
    out = sec_pipe(line, truncation=True)
    label = normalize_section_label(out[0]["label"])
    score = float(out[0].get("score", 0.0))
    return label, score


def predict_entities(ner_pipe, line: str) -> List[EntitySpan]:
    raw = ner_pipe(line)

    spans: List[EntitySpan] = []
    for r in raw:
        start = int(r.get("start", 0))
        end = int(r.get("end", 0))
        if start >= end:
            continue
        if start < 0 or end > len(line):
            continue

        # Use slice from original line so offsets stay correct
        slice_text = line[start:end]

        spans.append(
            EntitySpan(
                entity_group=r.get("entity_group", "UNKNOWN"),
                text=clean_wordpiece(slice_text),
                score=float(r.get("score", 0.0)),
                start=start,
                end=end,
            )
        )

    spans = merge_adjacent_spans(spans, line)
    spans.sort(key=lambda s: (s.start, s.end))
    return spans


def render_highlighted_html(text: str, spans: List[EntitySpan]) -> str:
    parts: List[str] = []
    cursor = 0

    spans = [s for s in spans if 0 <= s.start < s.end <= len(text)]

    for s in spans:
        if s.start < cursor:
            continue

        before = text[cursor:s.start]
        ent_text = text[s.start:s.end]

        parts.append(html.escape(before))

        bg_color = ENTITY_COLORS.get(s.entity_group, "#E0E0E0")
        tooltip = html.escape(f"{s.entity_group} · {s.score:.2f}")

        parts.append(
            f'<span title="{tooltip}" '
            f'style="'
            f'background:{bg_color};'
            f'color:#111;'
            f'border-radius:4px;'
            f'padding:0 4px;'
            f'margin:0 1px;'
            f'font-weight:500;'
            f'">'
            f'{html.escape(ent_text)}'
            f'</span>'
        )

        cursor = s.end

    parts.append(html.escape(text[cursor:]))

    body = "".join(parts)
    return (
        '<div style="'
        'font-family:inherit;'
        'font-size:1rem;'
        'line-height:1.6;'
        'color:#EAEAEA;'
        '">'
        f'{body}'
        '</div>'
    )

def render_entity_legend() -> str:
    items = []
    for label, color in ENTITY_COLORS.items():
        items.append(
            f'<div style="display:flex; align-items:center; gap:8px; margin:6px 0;">'
            f'  <span style="width:14px; height:14px; border-radius:3px; background:{color}; display:inline-block; border:1px solid rgba(255,255,255,0.25);"></span>'
            f'  <span style="color:#EAEAEA; font-size:0.95rem;">{html.escape(label)}</span>'
            f'</div>'
        )

    return (
        '<div style="padding:10px 12px; border-radius:10px; '
        'background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);">'
        '<div style="color:#EAEAEA; font-weight:600; margin-bottom:8px;">Entity legend</div>'
        f'{"".join(items)}'
        '</div>'
    )



def main():
    st.set_page_config(page_title="Clinical Doc Understanding Demo", layout="wide")
    st.title("Clinical Document Understanding Demo")
    st.caption("Project 1 — Section Classification + NER (fine-tuned BERT)")

    with st.sidebar:
        st.header("Model Paths")
        ner_dir = st.text_input("NER model directory", value="outputs/ner")
        sec_dir = st.text_input("Section model directory", value="outputs/sections")

        st.divider()
        st.header("Controls")
        show_entity_table = st.checkbox("Show entity table", value=True)
        min_score = st.slider("Min entity confidence", 0.0, 1.0, 0.80, 0.05)
        min_len = st.slider("Min entity length (chars)", 1, 10, 3, 1)

    ner_pipe, sec_pipe = load_pipelines(ner_dir, sec_dir)

    default_text = (
        "HPI: Patient reports wheezing, shortness of breath for 2 days. Denies chest pain.\n"
        "MEDS: Started albuterol 2 puffs inhaled q4h PRN.\n"
        "ASSESSMENT: Likely asthma. Consider chest X-ray to evaluate.\n"
        "PLAN: Continue albuterol 2 puffs inhaled q4h PRN. Follow up in 2 weeks."
    )

    note_text = st.text_area(
        "Paste a clinical note (one section per line works great):",
        value=default_text,
        height=180,
    )

    run = st.button("Run models", type="primary")
    if not run:
        st.info("Click **Run models** to see predictions.")
        return

    lines = [l.strip() for l in note_text.splitlines() if l.strip()]
    if not lines:
        st.warning("No text provided.")
        return

    results = []
    all_entity_rows: List[Dict[str, Any]] = []

    for i, line in enumerate(lines, start=1):
        sec_label, sec_score = predict_section(sec_pipe, line)

        spans = [
            s for s in predict_entities(ner_pipe, line)
            if s.score >= min_score and len(s.text) >= min_len
        ]

        results.append(
            {
                "line_no": i,
                "line": line,
                "section_pred": sec_label,
                "section_conf": sec_score,
                "spans": spans,
            }
        )

        for s in spans:
            all_entity_rows.append(
                {
                    "line_no": i,
                    "entity": s.entity_group,
                    "text": s.text,
                    "score": round(s.score, 3),
                }
            )

    col_left, col_right = st.columns([1.2, 1.0], gap="large")

    with col_left:
        st.subheader("Line-by-line predictions")
        for r in results:
            st.markdown(
                f"**Line {r['line_no']} — Section:** `{r['section_pred']}` "
                f"(conf {r['section_conf']:.2f})"
            )

            highlighted_html = render_highlighted_html(r["line"], r["spans"])
            # Render HTML directly (no markdown parsing)
            components.html(highlighted_html, height=50, scrolling=False)

            st.divider()

    with col_right:
        st.subheader("Summary")
        st.subheader("Legend")
        components.html(render_entity_legend(), height=220, scrolling=False)

        section_counts: Dict[str, int] = {}
        for r in results:
            section_counts[r["section_pred"]] = section_counts.get(r["section_pred"], 0) + 1

        st.write("**Predicted section counts:**")
        st.json(section_counts)

        if show_entity_table:
            st.subheader("Extracted entities")
            if all_entity_rows:
                st.dataframe(all_entity_rows, use_container_width=True, hide_index=True)
            else:
                st.write("No entities above the confidence/length thresholds.")

        st.subheader("How to read this")
        st.markdown(
            "- Left: section prediction per line + highlighted entity spans (hover highlight to see label/score)\n"
            "- Right: structured entity table (what you’d export downstream)\n"
            "- If entities are still noisy, raise **Min entity confidence**."
        )


if __name__ == "__main__":
    main()
