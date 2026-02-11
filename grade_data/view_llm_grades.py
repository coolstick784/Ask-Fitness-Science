"""
view_llm_grades.py
Allow a user to view LLM grading and provide input on whether they agree
"""
from pathlib import Path
from typing import Dict

import streamlit as st

from config import DEFAULT_AGREEMENT_RESULTS, DEFAULT_LLM_EVAL_RESULTS
from io_utils import read_jsonl, write_jsonl


RESULTS_PATH = DEFAULT_LLM_EVAL_RESULTS
AGREEMENT_PATH = DEFAULT_AGREEMENT_RESULTS


def load_agreements(path: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for row in read_jsonl(path):
        qid = str(row.get("question_id", "")).strip()
        if qid:
            out[qid] = row
    return out


def save_agreements(path: Path, agreements: Dict[str, Dict]) -> None:
    rows = [agreements[qid] for qid in sorted(agreements.keys(), key=lambda x: int(x) if x.isdigit() else x)]
    write_jsonl(path, rows)


def main() -> None:
    st.set_page_config(page_title="LLM Grade Viewer", layout="wide")
    st.title("LLM Grade Viewer")

    rows = read_jsonl(RESULTS_PATH)
    if not rows:
        st.error(f"No rows found in {RESULTS_PATH}. Run auto_grade_llm.py first.")
        return
    agreements = load_agreements(AGREEMENT_PATH)

    if "idx" not in st.session_state:
        st.session_state["idx"] = 0
    total = len(rows)
    idx = max(0, min(st.session_state["idx"], total - 1))
    st.session_state["idx"] = idx
    row = rows[idx]

    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        if st.button("Prev", disabled=idx == 0):
            st.session_state["idx"] = idx - 1
            st.rerun()
    with c2:
        if st.button("Next", disabled=idx >= total - 1):
            st.session_state["idx"] = idx + 1
            st.rerun()
    with c3:
        st.caption(f"Item {idx + 1}/{total} | question_id={row.get('question_id', idx + 1)}")

    graded = len(agreements)
    agreed_count = sum(1 for r in agreements.values() if bool(r.get("agree", False)))
    disagreed_count = sum(1 for r in agreements.values() if "agree" in r and not bool(r.get("agree", False)))
    st.caption(f"Agreement progress: graded {graded}/{total} | agree {agreed_count} | disagree {disagreed_count}")

    st.markdown("### Question")
    st.write(str(row.get("question", "")))
    st.markdown(f"**Study ID:** `{row.get('study_id', '')}`")
    st.markdown(f"**Models:** answer=`{row.get('answer_model', '')}`, grader=`{row.get('grader_model', '')}`")
    st.markdown(f"**Timing:** answer={row.get('answer_time_s', 'NA')}s, grader={row.get('grader_time_s', 'NA')}s")

    st.markdown("### Qwen3-Coder Answer")
    st.write(str(row.get("answer", "")))

    st.markdown("### Grading")
    auto_grade = row.get("auto_grade", {})
    if isinstance(auto_grade, dict) and auto_grade:
        st.json(auto_grade)
    else:
        st.warning("No parsed grading JSON found. Showing raw grader output.")
        st.code(str(row.get("auto_grade_raw", "")))

    qid = str(row.get("question_id", idx + 1))
    current_vote = agreements.get(qid, {})
    default_idx = 0 if bool(current_vote.get("agree", True)) else 1
    with st.form("agreement_form"):
        agree_choice = st.radio("Do you agree with this auto grade?", options=["Agree", "Disagree"], index=default_idx)
        note = st.text_input("Optional note", value=str(current_vote.get("note", "")))
        submitted = st.form_submit_button("Save Agreement")
    if submitted:
        agreements[qid] = {
            "question_id": int(qid) if qid.isdigit() else qid,
            "study_id": row.get("study_id", ""),
            "agree": agree_choice == "Agree",
            "note": note.strip(),
        }
        save_agreements(AGREEMENT_PATH, agreements)
        st.success(f"Saved to {AGREEMENT_PATH}")


if __name__ == "__main__":
    main()
