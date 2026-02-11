"""
auto_grade_llm.py
Provide answers and provide grading on those answers for a set of questions
"""
import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from config import (
    DEFAULT_ANSWER_MODEL,
    DEFAULT_CHUNKS,
    DEFAULT_DENSE_K,
    DEFAULT_GRADER_MODEL,
    DEFAULT_INDEX,
    DEFAULT_LLM_EVAL_RESULTS,
    DEFAULT_META,
    DEFAULT_PMID_MAP,
    DEFAULT_QS,
    DEFAULT_SPARSE_K,
    DEFAULT_TOP_K,
)
from io_utils import read_jsonl
from llm_clients import call_ollama
from prompt_templates import answer_prompt, grader_prompt
from retrieval_utils import (
    build_or_load_sparse_cache,
    hybrid_fuse_scores,
    load_index_meta,
    load_pmid_map,
    rank_study_in_list,
    sparse_retrieve,
)


def extract_json_object(text: str) -> Dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(DEFAULT_QS))
    parser.add_argument("--output", default=str(DEFAULT_LLM_EVAL_RESULTS))
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    parser.add_argument("--meta", default=str(DEFAULT_META))
    parser.add_argument("--pmid-map", default=str(DEFAULT_PMID_MAP))
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
    parser.add_argument("--dense-k", type=int, default=DEFAULT_DENSE_K)
    parser.add_argument("--sparse-k", type=int, default=DEFAULT_SPARSE_K)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--contexts", type=int, default=5)
    parser.add_argument("--answer-max-tokens", type=int, default=350)
    parser.add_argument("--grader-max-tokens", type=int, default=220)
    parser.add_argument("--sleep-seconds", type=float, default=30.0)
    args = parser.parse_args()

    q_path = Path(args.questions)
    out_path = Path(args.output)
    chunks_path = Path(args.chunks)
    idx_path = Path(args.index)
    meta_path = Path(args.meta)
    pmid_map_path = Path(args.pmid_map)

    # Read the data needed for dense/sparse lookup

    questions = read_jsonl(q_path)
    chunks = read_jsonl(chunks_path)
    index = faiss.read_index(str(idx_path))
    index_meta = load_index_meta(meta_path)
    embed_model_name = str(index_meta.get("model", "BAAI/bge-base-en-v1.5")).strip() or "BAAI/bge-base-en-v1.5"
    embedder = SentenceTransformer(embed_model_name)
    pmid_map = load_pmid_map(pmid_map_path)
    sparse_tf, sparse_doc_lens, sparse_avg_dl, sparse_idf, sparse_postings = build_or_load_sparse_cache(chunks_path, chunks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for i, row in enumerate(questions, start=1):
            q = str(row.get("question", "")).strip()
            sid = str(row.get("study_id", "")).strip()
            if not q or not sid:
                continue
            # start timer at the start of each question
            t0 = time.perf_counter()
            
            # Calculated dense, sparse, and hybrid rankings
            q_emb = embedder.encode([q], batch_size=16, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

            dense_scores, dense_idxs = index.search(q_emb, args.dense_k)
            dense_ranked: List[Tuple[int, float]] = []
            dense_only: List[int] = []
            for idx, score in zip(dense_idxs[0].tolist(), dense_scores[0].tolist()):
                if idx >= 0:
                    dense_ranked.append((int(idx), float(score)))
                    dense_only.append(int(idx))
            sparse_ranked = sparse_retrieve(q, sparse_tf, sparse_doc_lens, sparse_avg_dl, sparse_idf, sparse_postings, args.sparse_k)
            sparse_only = [idx for idx, _ in sparse_ranked]
            fused_idxs = hybrid_fuse_scores(dense_ranked, sparse_ranked, args.top_k)
            contexts = [chunks[idx] for idx in fused_idxs[: max(1, args.contexts)] if 0 <= idx < len(chunks)]

            # Get the answer + grade
            answer = call_ollama(args.answer_model, answer_prompt(q, contexts), args.answer_max_tokens)
            answer_t = time.perf_counter() - t0
            t1 = time.perf_counter()
            grader_raw = call_ollama(args.grader_model, grader_prompt(q, answer, contexts), args.grader_max_tokens)
            grader_t = time.perf_counter() - t1

            # Save the data
            rec = {
                "question_id": i,
                "study_id": sid,
                "question": q,
                "answer_model": args.answer_model,
                "grader_model": args.grader_model,
                "answer": answer,
                "answer_time_s": round(answer_t, 4),
                "grader_time_s": round(grader_t, 4),
                "auto_grade": extract_json_object(grader_raw),
                "auto_grade_raw": grader_raw,
                "target_rank_fused": rank_study_in_list(fused_idxs, sid, chunks, pmid_map),
                "target_rank_dense": rank_study_in_list(dense_only[: args.top_k], sid, chunks, pmid_map),
                "target_rank_sparse": rank_study_in_list(sparse_only[: args.top_k], sid, chunks, pmid_map),
                "retrieved_contexts": [
                    {"pmcid": str(c.get("pmcid", "")).strip(), "title": str(c.get("title", "")).strip(), "text": str(c.get("text", "")).strip()}
                    for c in contexts
                ],
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i}] done | answer_time={answer_t:.2f}s grader_time={grader_t:.2f}s")
            if i < len(questions) and args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
