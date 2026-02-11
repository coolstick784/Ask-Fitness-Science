"""
compare_agreement.py
Generate summary statistics of the LLM eval results 
"""

import argparse
from pathlib import Path
from typing import Dict, List

from config import DEFAULT_AGREEMENT_RESULTS, DEFAULT_LLM_EVAL_RESULTS
from io_utils import read_jsonl


AUTO_PATH = DEFAULT_LLM_EVAL_RESULTS
AGREE_PATH = DEFAULT_AGREEMENT_RESULTS


def to_id(v) -> str:
    return str(v).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare manual agreement decisions with auto-grade outputs.")
    parser.add_argument("--auto", default=str(AUTO_PATH), help="Path to llm_eval_results.jsonl")
    parser.add_argument("--agree", default=str(AGREE_PATH), help="Path to llm_grade_agreement.jsonl")
    args = parser.parse_args()

    auto_rows = read_jsonl(Path(args.auto))
    agree_rows = read_jsonl(Path(args.agree))

    if not auto_rows:
        raise FileNotFoundError(f"No auto-grade rows found: {args.auto}")
    if not agree_rows:
        raise FileNotFoundError(f"No agreement rows found: {args.agree}")

    auto_by_id: Dict[str, Dict] = {}
    for r in auto_rows:
        qid = to_id(r.get("question_id", ""))
        if qid:
            auto_by_id[qid] = r

    agree_by_id: Dict[str, Dict] = {}
    for r in agree_rows:
        qid = to_id(r.get("question_id", ""))
        if qid:
            agree_by_id[qid] = r

    matched_ids = [qid for qid in agree_by_id.keys() if qid in auto_by_id]
    if not matched_ids:
        raise RuntimeError("No overlapping question_id values between auto and agreement files.")

    total_auto = len(auto_by_id)
    total_manual = len(agree_by_id)
    total_matched = len(matched_ids)

    agree_true = 0
    agree_false = 0
    disagreement_ids: List[str] = []

    for qid in matched_ids:
        agree_val = bool(agree_by_id[qid].get("agree", False))
        if agree_val:
            agree_true += 1
        else:
            agree_false += 1
            disagreement_ids.append(qid)

    def pct(n: int, d: int) -> float:
        return (100.0 * n / d) if d > 0 else 0.0

    print("Agreement Summary")
    print(f"Auto rows: {total_auto}")
    print(f"Manual rows: {total_manual}")
    print(f"Matched rows: {total_matched}")
    print(f"Agree: {agree_true}/{total_matched} ({pct(agree_true, total_matched):.2f}%)")
    print(f"Disagree: {agree_false}/{total_matched} ({pct(agree_false, total_matched):.2f}%)")
    print("")
    print("Disagreement Question IDs")
    if disagreement_ids:
        print(", ".join(disagreement_ids))
    else:
        print("None")


if __name__ == "__main__":
    main()
