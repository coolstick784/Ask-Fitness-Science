import json
import random
from pathlib import Path
from typing import Dict, Optional

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_CANDIDATES = [
    BASE_DIR / "pipeline-data" / "pmc_corpus.jsonl",
    BASE_DIR / "pipeline-data" / "pmc_chunks.jsonl",
]
OUT_JSONL = Path(__file__).resolve().parent / "test_questions.jsonl"


def extract_abstract(record: Dict) -> str:
    # Corpus format: sections[]. Fallback to direct abstract/text fields.
    sections = record.get("sections") or []
    if isinstance(sections, list):
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            heading = str(sec.get("heading", "")).lower()
            text = str(sec.get("text", "")).strip()
            if text and "abstract" in heading:
                return text
        for sec in sections:
            if isinstance(sec, dict):
                text = str(sec.get("text", "")).strip()
                if text:
                    return text
    for key in ("abstract", "text"):
        val = str(record.get(key, "")).strip()
        if val:
            return val
    return ""


def find_source_jsonl() -> Optional[Path]:
    for p in CORPUS_CANDIDATES:
        if p.exists():
            return p
    return None


def pick_random_study(path: Path) -> Optional[Dict]:
    # Reservoir sampling to pick one random line without loading whole file.
    chosen = None
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            seen += 1
            if random.randint(1, seen) == 1:
                chosen = obj
    return chosen


def study_id(study: Dict) -> str:
    for key in ("pmcid", "pmid", "id"):
        val = str(study.get(key, "")).strip()
        if val:
            return val
    return "UNKNOWN"


def main() -> None:
    st.set_page_config(page_title="Create Test Questions", layout="wide")
    st.title("Create Test Questions")

    source = find_source_jsonl()
    if not source:
        st.error("No source file found. Expected pipeline-data/pmc_corpus.jsonl or pmc_chunks.jsonl.")
        return

    if "current_study" not in st.session_state:
        st.session_state["current_study"] = pick_random_study(source)

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Pick Random Study"):
            st.session_state["current_study"] = pick_random_study(source)
            st.rerun()
    with col2:
        st.caption(f"Source: {source}")

    study = st.session_state.get("current_study")
    if not study:
        st.error("Could not load a study from source file.")
        return

    sid = study_id(study)
    title = str(study.get("title", "")).strip() or "(No title)"
    abstract = extract_abstract(study) or "(No abstract)"

    st.subheader(f"Study ID: {sid}")
    st.markdown(f"**Title:** {title}")
    st.markdown("**Abstract:**")
    st.write(abstract)

    with st.form("question_form", clear_on_submit=True):
        q1 = st.text_area("Question 1")
        q2 = st.text_area("Question 2")
        submitted = st.form_submit_button("Save 2 Questions")

    if submitted:
        q1s = q1.strip()
        q2s = q2.strip()
        if not q1s or not q2s:
            st.error("Please enter both questions before saving.")
            return

        OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with OUT_JSONL.open("a", encoding="utf-8") as f:
            for q in (q1s, q2s):
                rec = {"study_id": sid, "question": q}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        st.success(f"Saved 2 questions to {OUT_JSONL}")


if __name__ == "__main__":
    main()
