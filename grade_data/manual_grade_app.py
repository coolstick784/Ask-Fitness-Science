import json
from pathlib import Path
from typing import Dict, List

import streamlit as st

from config import DEFAULT_LLM_EVAL_RESULTS, DEFAULT_MANUAL_GRADES
from io_utils import read_jsonl, write_jsonl


INPUT_JSONL = DEFAULT_LLM_EVAL_RESULTS
OUT_JSONL = DEFAULT_MANUAL_GRADES

SUPPORT_OPTIONS = ["supported", "partially supported", "not supported"]


def load_manual_grades(path: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for row in read_jsonl(path):
        qid = str(row.get("question_id", "")).strip()
        if qid:
            out[qid] = row
    return out


def save_manual_grades(path: Path, grades_by_id: Dict[str, Dict]) -> None:
    rows = [grades_by_id[qid] for qid in sorted(grades_by_id.keys(), key=lambda x: int(x) if x.isdigit() else x)]
    write_jsonl(path, rows)


def main() -> None:
    st.set_page_config(page_title="Manual LLM Grading", layout="wide")
    st.title("Manual LLM Grading")

    rows = read_jsonl(INPUT_JSONL)
    if not rows:
        st.error(f"No input rows found at {INPUT_JSONL}. Run auto_grade_llm.py first.")
        return

    manual = load_manual_grades(OUT_JSONL)

    if "idx" not in st.session_state:
        st.session_state["idx"] = 0
    max_idx = len(rows) - 1
    st.session_state["idx"] = max(0, min(st.session_state["idx"], max_idx))
    row = rows[st.session_state["idx"]]

    qid = str(row.get("question_id", st.session_state["idx"] + 1))
    pre = manual.get(qid, {})

    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("Prev", disabled=st.session_state["idx"] == 0):
            st.session_state["idx"] -= 1
            st.rerun()
    with col2:
        if st.button("Next", disabled=st.session_state["idx"] >= max_idx):
            st.session_state["idx"] += 1
            st.rerun()
    with col3:
        st.caption(f"Item {st.session_state['idx'] + 1}/{len(rows)} | question_id={qid}")

    st.markdown("### Question")
    st.write(str(row.get("question", "")))
    st.markdown(f"**Study ID:** `{row.get('study_id', '')}`")
    st.markdown(
        f"**Ranks:** fused={row.get('target_rank_fused', -1)}, "
        f"dense={row.get('target_rank_dense', -1)}, sparse={row.get('target_rank_sparse', -1)}"
    )
    st.markdown(
        f"**Timing:** answer={row.get('answer_time_s', 'NA')}s, grader={row.get('grader_time_s', 'NA')}s"
    )

    st.markdown("### LLM Answer")
    st.write(str(row.get("answer", "")))

    with st.expander("Retrieved Contexts", expanded=False):
        for i, c in enumerate(row.get("retrieved_contexts", []), start=1):
            st.markdown(f"**Study {i}** `{c.get('pmcid', '')}`")
            st.write(c.get("title", ""))
            st.write(c.get("text", ""))
            st.markdown("---")

    st.markdown("### Grade")
    with st.form("grade_form"):
        chunks_sources_good = st.radio(
            "A. Chunks/formatting/sources are good",
            options=[True, False],
            index=0 if pre.get("chunks_sources_good", True) else 1,
            format_func=lambda x: "True" if x else "False",
        )
        support = st.selectbox(
            "B. Support level",
            options=SUPPORT_OPTIONS,
            index=SUPPORT_OPTIONS.index(pre.get("support", "partially supported"))
            if pre.get("support", "partially supported") in SUPPORT_OPTIONS
            else 1,
        )
        answer_quality = st.selectbox(
            "B. Answer quality (1-3)",
            options=[1, 2, 3],
            index=max(0, min(2, int(pre.get("answer_quality", 2)) - 1)),
        )
        notes = st.text_area("Notes", value=str(pre.get("notes", "")))
        submitted = st.form_submit_button("Save Grade")

    if submitted:
        rec = {
            "question_id": int(qid) if str(qid).isdigit() else qid,
            "study_id": row.get("study_id", ""),
            "question": row.get("question", ""),
            "chunks_sources_good": bool(chunks_sources_good),
            "support": support,
            "answer_quality": int(answer_quality),
            "notes": notes.strip(),
        }
        manual[str(qid)] = rec
        save_manual_grades(OUT_JSONL, manual)
        st.success(f"Saved manual grade to {OUT_JSONL}")


if __name__ == "__main__":
    main()
