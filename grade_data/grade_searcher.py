import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "pipeline-data"
DEFAULT_INDEX = DATA_DIR / "pmc_faiss.index"
DEFAULT_CHUNKS = DATA_DIR / "pmc_chunks.jsonl"
DEFAULT_META = DATA_DIR / "pmc_index_meta.json"
DEFAULT_QS = Path(__file__).resolve().parent / "test_questions.jsonl"
DEFAULT_PMID_MAP = DATA_DIR / "pmid_to_pmcid.jsonl"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of",
    "on", "or", "that", "the", "to", "was", "were", "will", "with", "this", "these", "those",
    "about", "after", "before", "between", "during", "each", "every", "however", "including",
    "may", "might", "often", "per", "regarding", "thus", "via", "also",
}


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def load_questions(path: Path) -> List[Dict]:
    out: List[Dict] = []
    for row in read_jsonl(path):
        q = str(row.get("question", "")).strip()
        sid = str(row.get("study_id", "")).strip()
        if q and sid:
            out.append({"question": q, "study_id": sid})
    return out


def load_index_meta(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_pmid_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    mapping: Dict[str, str] = {}
    for row in read_jsonl(path):
        pmcid = str(row.get("pmcid", "")).strip()
        pmid = str(row.get("pmid", "")).strip()
        if pmcid.startswith("PMC"):
            mapping[pmcid] = pmcid
            if pmid:
                mapping[pmid] = pmcid
    return mapping


def normalize_study_id(raw_id: str, pmid_map: Dict[str, str]) -> str:
    rid = str(raw_id).strip()
    if not rid:
        return ""
    if rid in pmid_map:
        return pmid_map[rid]
    return rid


def sparse_tokenize(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    out: List[str] = []
    for t in toks:
        if t in STOPWORDS:
            continue
        if len(t) < 2 and not t.isdigit():
            continue
        out.append(t)
    return out


def build_or_load_sparse_cache(
    chunks_path: Path, chunks: List[Dict]
) -> Tuple[List[Dict[str, int]], List[int], float, Dict[str, float], Dict[str, List[int]]]:
    cache_path = chunks_path.with_suffix(".sparse.pkl")
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and all(k in obj for k in ("term_freqs", "doc_lens", "avg_dl", "idf", "postings")):
                return obj["term_freqs"], obj["doc_lens"], obj["avg_dl"], obj["idf"], obj["postings"]
        except Exception:
            pass

    term_freqs: List[Dict[str, int]] = []
    doc_lens: List[int] = []
    df: Dict[str, int] = {}
    postings: Dict[str, List[int]] = {}

    for i, c in enumerate(chunks):
        toks = sparse_tokenize(str(c.get("text", "")))
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        term_freqs.append(tf)
        doc_lens.append(len(toks))
        for tok in tf:
            df[tok] = df.get(tok, 0) + 1
            postings.setdefault(tok, []).append(i)

    n_docs = max(1, len(term_freqs))
    avg_dl = float(sum(doc_lens) / max(1, len(doc_lens)))
    idf = {tok: float(np.log((n_docs - d + 0.5) / (d + 0.5) + 1.0)) for tok, d in df.items()}

    try:
        with cache_path.open("wb") as f:
            pickle.dump(
                {
                    "term_freqs": term_freqs,
                    "doc_lens": doc_lens,
                    "avg_dl": avg_dl,
                    "idf": idf,
                    "postings": postings,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception:
        pass

    return term_freqs, doc_lens, avg_dl, idf, postings


def sparse_retrieve(
    query: str,
    term_freqs: List[Dict[str, int]],
    doc_lens: List[int],
    avg_dl: float,
    idf: Dict[str, float],
    postings: Dict[str, List[int]],
    top_k: int,
) -> List[Tuple[int, float]]:
    q_tokens = list(dict.fromkeys(sparse_tokenize(query)))
    if not q_tokens:
        return []

    candidate_docs = set()
    for t in q_tokens:
        for doc_id in postings.get(t, []):
            candidate_docs.add(doc_id)
    if not candidate_docs:
        return []

    k1 = 1.5
    b = 0.75
    scored: List[Tuple[int, float]] = []
    for i in candidate_docs:
        tf = term_freqs[i]
        dl = max(1, doc_lens[i] if i < len(doc_lens) else 1)
        score = 0.0
        for t in q_tokens:
            f = tf.get(t, 0)
            if f <= 0:
                continue
            idf_t = idf.get(t, 0.0)
            denom = f + k1 * (1.0 - b + b * (dl / max(1e-9, avg_dl)))
            score += idf_t * ((f * (k1 + 1.0)) / max(1e-9, denom))
        if score > 0.0:
            scored.append((i, float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def hybrid_fuse_scores(
    dense: List[Tuple[int, float]],
    sparse: List[Tuple[int, float]],
    top_k: int,
    w_dense: float = 0.7,
    w_sparse: float = 0.3,
    fuse_limit: int = 200,
) -> List[int]:
    def minmax_norm(items: List[Tuple[int, float]]) -> Dict[int, float]:
        if not items:
            return {}
        items = items[:fuse_limit]
        idxs = [i for i, _ in items]
        vals = np.array([s for _, s in items], dtype=np.float64)
        lo = float(vals.min())
        hi = float(vals.max())
        if hi == lo:
            return {idx: 1.0 for idx in idxs}
        norm = (vals - lo) / (hi - lo)
        return {idxs[j]: float(norm[j]) for j in range(len(idxs))}

    d = minmax_norm(dense)
    s = minmax_norm(sparse)
    fused: Dict[int, float] = {}
    for idx, v in d.items():
        fused[idx] = fused.get(idx, 0.0) + w_dense * v
    for idx, v in s.items():
        fused[idx] = fused.get(idx, 0.0) + w_sparse * v

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [idx for idx, _ in ranked[:top_k]]


def rank_study_in_list(
    candidate_idxs: List[int], target_study_id: str, chunks: List[Dict], pmid_map: Dict[str, str]
) -> int:
    target_norm = normalize_study_id(target_study_id, pmid_map)
    for pos, chunk_idx in enumerate(candidate_idxs, start=1):
        if chunk_idx < 0 or chunk_idx >= len(chunks):
            continue
        chunk_sid = normalize_study_id(str(chunks[chunk_idx].get("pmcid", "")).strip(), pmid_map)
        if chunk_sid and chunk_sid == target_norm:
            return pos
    return -1


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
    parser.add_argument("--dense-k", type=int, default=80, help="Dense retrieval depth")
    parser.add_argument("--sparse-k", type=int, default=200, help="Sparse retrieval depth")
    parser.add_argument("--top-k", type=int, default=100, help="Top results to evaluate rank against")
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
        dense_ranked = []
        dense_only_idxs = []
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

    print("")
    print("Metrics")
    summarize_hits("Fused ", fused_ranks, total)
    summarize_hits("Dense ", dense_ranks, total)
    summarize_hits("Sparse", sparse_ranks, total)


if __name__ == "__main__":
    main()

