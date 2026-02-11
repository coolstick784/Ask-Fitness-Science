"""
retrieval_utils.py
A collection of functions for assistance with the project
"""
import json
import pickle
import re
from typing import Dict, List, Tuple

import numpy as np

# If importing from app.py, we want to use grade_data as a base.
try:
    from grade_data.config import DEFAULT_CHUNKS, DEFAULT_INDEX, DEFAULT_META, DEFAULT_PMID_MAP, DEFAULT_QS
    from grade_data.io_utils import read_jsonl
except ImportError:
    from config import DEFAULT_CHUNKS, DEFAULT_INDEX, DEFAULT_META, DEFAULT_PMID_MAP, DEFAULT_QS
    from io_utils import read_jsonl


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of",
    "on", "or", "that", "the", "to", "was", "were", "will", "with", "this", "these", "those",
    "about", "after", "before", "between", "during", "each", "every", "however", "including",
    "may", "might", "often", "per", "regarding", "thus", "via", "also",
}

def load_sparse_nlp_tools() -> Tuple[set, object, object]:
    # Load the lemmatizer and stemmer
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer, WordNetLemmatizer

    for resource_path, download_name in (
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ):
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(download_name, quiet=True)
            except Exception:
                pass

    try:
        stop_set = set(stopwords.words("english"))
    except Exception:
        stop_set = set()
    # Remove excess filler words
    stop_set.update(
        {
            "also", "about", "across", "after", "before", "between", "during", "each",
            "either", "every", "however", "including", "may", "might", "often",
            "per", "regarding", "thus", "via",
        }
    )

    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")
    return stop_set, lemmatizer, stemmer

def load_questions(path) -> List[Dict]:
    out: List[Dict] = []
    for row in read_jsonl(path):
        q = str(row.get("question", "")).strip()
        sid = str(row.get("study_id", "")).strip()
        if q and sid:
            out.append({"question": q, "study_id": sid})
    return out


def load_index_meta(path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_pmid_map(path) -> Dict[str, str]:
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


# Lemmatize and stem words if identified as a candidate 
def sparse_tokenize(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    stop_set, lemmatizer, stemmer = load_sparse_nlp_tools()
    out: List[str] = []
    for t in toks:
        if t in stop_set:
            continue
        
        try:
            t = lemmatizer.lemmatize(t, pos="n")
            t = lemmatizer.lemmatize(t, pos="v")
        except Exception:
            pass
        try:
            t = stemmer.stem(t)
        except Exception:
            pass
        if not t or t in stop_set:
            continue
        if len(t) < 2 and not t.isdigit():
            continue
        out.append(t)
    return out

# Load a cache of sparse data at the document level, including applying the BM25 algorithm across all chunks
def build_or_load_sparse_cache(chunks_path, chunks: List[Dict]):
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
    return term_freqs, doc_lens, avg_dl, idf, postings


# Calculate sparse scores based on the BM25 algorithm
def sparse_retrieve(query, term_freqs, doc_lens, avg_dl, idf, postings, top_k):
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
    scored = []
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

# Fuse dense and sparse scores according to minmax scaling and weighting.
# Optionally rerank fused candidates with bge-reranker-base when query/chunks are provided.
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


def rank_study_in_list(candidate_idxs: List[int], target_study_id: str, chunks: List[Dict], pmid_map: Dict[str, str]) -> int:
    target_norm = normalize_study_id(target_study_id, pmid_map)
    for pos, chunk_idx in enumerate(candidate_idxs, start=1):
        if chunk_idx < 0 or chunk_idx >= len(chunks):
            continue
        chunk_sid = normalize_study_id(str(chunks[chunk_idx].get("pmcid", "")).strip(), pmid_map)
        if chunk_sid and chunk_sid == target_norm:
            return pos
    return -1
