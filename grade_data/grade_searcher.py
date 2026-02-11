"""
grade_searcher.py
Evaluate dense/sparse/fused retrieval ranks for test questions.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from config import (
    DEFAULT_CHUNKS,
    DEFAULT_DENSE_K,
    DEFAULT_INDEX,
    DEFAULT_META,
    DEFAULT_PMID_MAP,
    DEFAULT_QS,
    DEFAULT_SPARSE_K,
    DEFAULT_TOP_K,
)
from io_utils import read_jsonl
from retrieval_utils import (
    build_or_load_sparse_cache,
    hybrid_fuse_scores,
    load_index_meta,
    load_pmid_map,
    load_questions,
    rank_study_in_list,
    sparse_retrieve,
)


def summarize_hits(name: str, ranks: List[int], total: int) -> None:
    def count_at(k: int) -> int:
        return sum(1 for r in ranks if 0 < r <= k)

    def pct(n: int) -> float:
        return (100.0 * n / total) if total > 0 else 0.0

    top5 = count_at(5)
    top10 = count_at(10)
    top20 = count_at(20)
    print(f"{name} Top-5  : {top5}/{total} ({pct(top5):.2f}%)")
    print(f"{name} Top-10 : {top10}/{total} ({pct(top10):.2f}%)")
    print(f"{name} Top-20 : {top20}/{total} ({pct(top20):.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade retrieval ranking on test questions.")
    parser.add_argument("--questions", default=str(DEFAULT_QS), help="Path to test_questions.jsonl")
    parser.add_argument("--index", default=str(DEFAULT_INDEX), help="Path to FAISS index")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS), help="Path to chunks JSONL")
    parser.add_argument("--meta", default=str(DEFAULT_META), help="Path to index meta JSON")
    parser.add_argument("--pmid-map", default=str(DEFAULT_PMID_MAP), help="Path to pmid_to_pmcid JSONL")
    parser.add_argument("--dense-k", type=int, default=DEFAULT_DENSE_K, help="Dense retrieval depth")
    parser.add_argument("--sparse-k", type=int, default=DEFAULT_SPARSE_K, help="Sparse retrieval depth")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top results to evaluate rank against")
    args = parser.parse_args()

    q_path = Path(args.questions)
    idx_path = Path(args.index)
    chunks_path = Path(args.chunks)
    meta_path = Path(args.meta)
    pmid_map_path = Path(args.pmid_map)

    if not q_path.exists():
        raise FileNotFoundError(f"Missing questions file: {q_path}")
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing index file: {idx_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    questions = load_questions(q_path)
    if not questions:
        raise RuntimeError("No valid question rows found.")
    
    # Get the data needed for ranking and searching

    chunks = read_jsonl(chunks_path)
    index = faiss.read_index(str(idx_path))
    index_meta = load_index_meta(meta_path)
    model_name = str(index_meta.get("model", "BAAI/bge-base-en-v1.5")).strip() or "BAAI/bge-base-en-v1.5"
    model = SentenceTransformer(model_name)
    pmid_map = load_pmid_map(pmid_map_path)
    sparse_tf, sparse_doc_lens, sparse_avg_dl, sparse_idf, sparse_postings = build_or_load_sparse_cache(chunks_path, chunks)

    fused_ranks: List[int] = []
    dense_ranks: List[int] = []
    sparse_ranks: List[int] = []

    total = len(questions)
    for i, row in enumerate(questions, start=1):
        # For each question, get the dense, sparse, and fused scores, then get its ranking in the list. 
        question = row["question"]
        study_id = row["study_id"]

        q_emb = model.encode(
            [question],
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        dense_scores, dense_idxs = index.search(q_emb, args.dense_k)
        dense_ranked: List[Tuple[int, float]] = []
        dense_only_idxs: List[int] = []
        for idx, score in zip(dense_idxs[0].tolist(), dense_scores[0].tolist()):
            if idx < 0:
                continue
            dense_ranked.append((int(idx), float(score)))
            dense_only_idxs.append(int(idx))

        sparse_ranked = sparse_retrieve(
            question,
            term_freqs=sparse_tf,
            doc_lens=sparse_doc_lens,
            avg_dl=sparse_avg_dl,
            idf=sparse_idf,
            postings=sparse_postings,
            top_k=args.sparse_k,
        )
        sparse_only_idxs = [idx for idx, _ in sparse_ranked]

        fused_idxs = hybrid_fuse_scores(
            dense=dense_ranked,
            sparse=sparse_ranked,
            top_k=args.top_k,
            w_dense=0.7,
            w_sparse=0.3,
            fuse_limit=max(args.dense_k, args.sparse_k),
        )

        fused_rank = rank_study_in_list(fused_idxs, study_id, chunks, pmid_map)
        dense_rank = rank_study_in_list(dense_only_idxs[: args.top_k], study_id, chunks, pmid_map)
        sparse_rank = rank_study_in_list(sparse_only_idxs[: args.top_k], study_id, chunks, pmid_map)

        fused_ranks.append(fused_rank)
        dense_ranks.append(dense_rank)
        sparse_ranks.append(sparse_rank)


        def to_text(r: int) -> str:
            return str(r) if r > 0 else f"> {args.top_k} / not found"

        print(
            f"[{i}/{total}] study_id={study_id} "
            f"fused={to_text(fused_rank)} dense={to_text(dense_rank)} sparse={to_text(sparse_rank)}"
        )
    # Generate summary statistics
    print("")
    print("Metrics")
    summarize_hits("Fused ", fused_ranks, total)
    summarize_hits("Dense ", dense_ranks, total)
    summarize_hits("Sparse", sparse_ranks, total)


if __name__ == "__main__":
    main()
