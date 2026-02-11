"""
ones_not_found.py
For studies that were not found in the top 5 of the fused rankings from the question bank, check if there were more relevant studies for questions
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from config import (
    DEFAULT_CHUNKS,
    DEFAULT_DENSE_K,
    DEFAULT_GRADER_MODEL,
    DEFAULT_INDEX,
    DEFAULT_META,
    DEFAULT_PMID_MAP,
    DEFAULT_QS,
    DEFAULT_SPARSE_K,
    DEFAULT_TOP_K,
)
from io_utils import read_jsonl
from llm_clients import call_ollama
from prompt_templates import top10_judge_prompt
from retrieval_utils import (
    build_or_load_sparse_cache,
    hybrid_fuse_scores,
    load_index_meta,
    load_pmid_map,
    load_questions,
    rank_study_in_list,
    sparse_retrieve,
)


def extract_json(text: str) -> Dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(DEFAULT_QS))
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    parser.add_argument("--meta", default=str(DEFAULT_META))
    parser.add_argument("--pmid-map", default=str(DEFAULT_PMID_MAP))
    parser.add_argument("--dense-k", type=int, default=DEFAULT_DENSE_K)
    parser.add_argument("--sparse-k", type=int, default=DEFAULT_SPARSE_K)
    parser.add_argument("--fused-top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
    parser.add_argument("--save-details", default="")
    args = parser.parse_args()

    # Get everything needed to get dense, sparse, and fused rankings
    questions = load_questions(Path(args.questions))
    chunks = read_jsonl(Path(args.chunks))
    index = faiss.read_index(str(Path(args.index)))
    model_name = str(load_index_meta(Path(args.meta)).get("model", "BAAI/bge-base-en-v1.5")).strip() or "BAAI/bge-base-en-v1.5"
    embedder = SentenceTransformer(model_name)
    pmid_map = load_pmid_map(Path(args.pmid_map))
    sparse_tf, sparse_doc_lens, sparse_avg_dl, sparse_idf, sparse_postings = build_or_load_sparse_cache(Path(args.chunks), chunks)

    total = len(questions)
    found = 0
    not_found = 0
    answered = 0
    not_answered = 0
    details: List[Dict] = []

    for i, row in enumerate(questions, start=1):
        # Get the dense, sparse, and fused scores
        q = row["question"]
        sid = row["study_id"]
        q_emb = embedder.encode([q], batch_size=16, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        dense_scores, dense_idxs = index.search(q_emb, args.dense_k)
        dense_ranked: List[Tuple[int, float]] = [(int(idx), float(score)) for idx, score in zip(dense_idxs[0].tolist(), dense_scores[0].tolist()) if idx >= 0]
        sparse_ranked = sparse_retrieve(q, sparse_tf, sparse_doc_lens, sparse_avg_dl, sparse_idf, sparse_postings, args.sparse_k)
        fused = hybrid_fuse_scores(dense_ranked, sparse_ranked, args.fused_top_k)

        # If it's in the top 10, say so and move on
        if 0 < rank_study_in_list(fused[:10], sid, chunks, pmid_map) <= 10:
            found += 1
            details.append({"question_id": i, "study_id": sid, "question": q, "fused_top10_found": True})
            print(f"[{i}/{total}] FOUND in fused top-10 | study_id={sid}")
            continue

        # Otherwise, ask a LLM if the top 10 studies accurately answer the question
        # If they do, then there may have been more relevant studies for the question
        not_found += 1
        top10_contexts = [chunks[idx] for idx in fused[:10] if 0 <= idx < len(chunks)]
        raw = call_ollama(args.grader_model, top10_judge_prompt(q, top10_contexts), 180)
        parsed = extract_json(raw)
        ok = bool(parsed.get("top10_answers_question", False))
        if ok:
            answered += 1
        else:
            not_answered += 1
        details.append(
            {
                "question_id": i,
                "study_id": sid,
                "question": q,
                "fused_top10_found": False,
                "top10_answers_question": ok,
                "reason": str(parsed.get("reason", "")).strip(),
                "grader_raw": raw,
            }
        )
        print(f"[{i}/{total}] NOT FOUND in fused top-10 | top10_answers_question={ok} | study_id={sid}")

    print("")
    print("Summary")
    print(f"Total questions: {total}")
    print(f"Found in fused top-10: {found}/{total}")
    print(f"Not found in fused top-10: {not_found}/{total}")
    if not_found > 0:
        print(f"Among NOT FOUND: top-10 DOES answer question: {answered}/{not_found}")
        print(f"Among NOT FOUND: top-10 does NOT answer question: {not_answered}/{not_found}")

    if args.save_details:
        out = Path(args.save_details)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"Saved details: {out}")


if __name__ == "__main__":
    main()
