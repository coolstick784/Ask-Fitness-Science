"""
app.py

This produces a Streamlit UI as well as takes in an input, processes it, and provides a response to the question
"""

import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer


# Use GROQ for the API for the LLM
GROQ_API_BASE = "https://api.groq.com/openai/v1"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "pipeline-data"

# Set token levels
TOKEN_LEVELS = {"Low": 160, "Medium": 320, "High": 640}
SUMMARY_TOKEN_LEVELS = {"Low": 96, "Medium": 192, "High": 320}
CHUNK_LEVELS = {"Low": 4, "Medium": 8, "High": 12}
RESPONSE_LEVEL_MAP = {"Short": "Low", "Medium": "Medium", "Long": "High"}


# Set speed profiles
SPEED_PROFILES = {
    "Fast": {
        "model": "llama-3.1-8b-instant",
        "embed_model": "all-MiniLM-L6-v2",
        "top_k": 4,
        "num_predict": 192,
        "summary_predict": 96,
        "context_chars": 420,
        "max_studies": 2,
        "index_path": str(DATA_DIR / "pmc_faiss_fast.index"),
        "chunks_path": str(DATA_DIR / "pmc_chunks_fast.jsonl"),
        "index_meta_path": str(DATA_DIR / "pmc_index_meta_fast.json"),
    },
    "Balanced": {
        "model": "llama-3.3-70b-versatile",
        "embed_model": "BAAI/bge-large-en-v1.5",
        "top_k": 6,
        "num_predict": 320,
        "summary_predict": 160,
        "context_chars": 700,
        "max_studies": 3,
        "index_path": str(DATA_DIR / "pmc_faiss.index"),
        "chunks_path": str(DATA_DIR / "pmc_chunks.jsonl"),
        "index_meta_path": str(DATA_DIR / "pmc_index_meta.json"),
    },
    "Quality": {
        "model": "groq/compound",
        "embed_model": "all-MiniLM-L6-v2",
        "top_k": 10,
        "num_predict": 512,
        "summary_predict": 256,
        "context_chars": 900,
        "max_studies": 3,
        "index_path": str(DATA_DIR / "pmc_faiss.index"),
        "chunks_path": str(DATA_DIR / "pmc_chunks.jsonl"),
        "index_meta_path": str(DATA_DIR / "pmc_index_meta.json"),
    },
}



@st.cache_resource(show_spinner=False)
def load_index(index_path: Path) -> faiss.Index:
    return faiss.read_index(str(index_path))


# Cache the data for the models
@st.cache_data(show_spinner=False, ttl=300)
def list_groq_models() -> List[str]:
    defaults: List[str] = []
    if not GROQ_API_KEY:
        return defaults
    try:
        resp = requests.get(
            f"{GROQ_API_BASE}/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=4,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data", []) if isinstance(data, dict) else []
        names = [str(m.get("id", "")).strip() for m in items if isinstance(m, dict) and m.get("id")]
        names = sorted(set(n for n in names if n and n.lower().startswith("groq")))
        return names or defaults
    except Exception:
        return defaults

# Load the chunks and cache that data. Try to load a preloaded file if possible
@st.cache_resource(show_spinner=False)
def load_chunks(chunks_path: Path) -> List[Dict]:
    cache_path = chunks_path.with_suffix(".chunks.pkl")
    try:
        src_stat = chunks_path.stat()
    except Exception:
        src_stat = None

    if src_stat and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                obj = pickle.load(f)
            if (
                isinstance(obj, dict)
                and int(obj.get("source_size", -1)) == int(src_stat.st_size)
                and int(obj.get("source_mtime_ns", -1)) == int(src_stat.st_mtime_ns)
                and isinstance(obj.get("chunks"), list)
            ):
                return obj["chunks"]
        except Exception:
            pass

    chunks: List[Dict] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except Exception:
                continue

    if src_stat:
        try:
            with cache_path.open("wb") as f:
                pickle.dump(
                    {
                        "source_size": int(src_stat.st_size),
                        "source_mtime_ns": int(src_stat.st_mtime_ns),
                        "chunks": chunks,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception:
            pass
    return chunks



@st.cache_resource(show_spinner=False)
def load_index_meta(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# Load the PMID mapping if possible, otherwise create it
@st.cache_resource(show_spinner=False)
def load_pmid_map(path: Path) -> Dict[str, str]:
    cache_path = path.with_suffix(".pmidmap.pkl")
    try:
        src_stat = path.stat()
    except Exception:
        src_stat = None

    if src_stat and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                obj = pickle.load(f)
            if (
                isinstance(obj, dict)
                and int(obj.get("source_size", -1)) == int(src_stat.st_size)
                and int(obj.get("source_mtime_ns", -1)) == int(src_stat.st_mtime_ns)
                and isinstance(obj.get("mapping"), dict)
            ):
                return obj["mapping"]
        except Exception:
            pass

    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            pmcid = rec.get("pmcid")
            pmid = str(rec.get("pmid", "")).strip()
            if pmcid and pmcid.startswith("PMC"):
                mapping[pmcid] = pmcid
                if pmid:
                    mapping[pmid] = pmcid
    if src_stat:
        try:
            with cache_path.open("wb") as f:
                pickle.dump(
                    {
                        "source_size": int(src_stat.st_size),
                        "source_mtime_ns": int(src_stat.st_mtime_ns),
                        "mapping": mapping,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception:
            pass
    return mapping


# Load the embedder for the vectorization
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


# Return the embedding of a series of texts
def embed(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    embs = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embs.astype("float32")


# Return the searching based on the embedding and FAISS index for dense searching
def retrieve(q_emb: np.ndarray, index: faiss.Index, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    return index.search(q_emb, top_k)


# load tools for sparse searching. Download them if not found
@st.cache_resource(show_spinner=False)
def load_sparse_nlp_tools() -> Tuple[set, object, object]:
    # NLTK-based normalization (stopwords + lemmatizer + stemmer).
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
    # Extra filler words to suppress conversational noise in sparse retrieval.
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

# Load the sparse index if possible, otherwise create it
@st.cache_resource(show_spinner=False)
def load_sparse_index(
    chunks_path: Path,
) -> Tuple[List[Dict[str, int]], List[int], float, Dict[str, float], Dict[str, List[int]]]:
    cache_path = chunks_path.with_suffix(".sparse.pkl")
    # On Streamlit Cloud, file mtimes can differ from local builds.
    # If a cache file exists and has the expected keys, trust it.
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and all(
                k in obj for k in ("term_freqs", "doc_lens", "avg_dl", "idf", "postings")
            ):
                return (
                    obj["term_freqs"],
                    obj["doc_lens"],
                    obj["avg_dl"],
                    obj["idf"],
                    obj["postings"],
                )
        except Exception:
            pass

    try:
        src_stat = chunks_path.stat()
    except Exception:
        src_stat = None

    chunks = load_chunks(chunks_path)
    term_freqs: List[Dict[str, int]] = []
    doc_lens: List[int] = []
    df: Dict[str, int] = {}
    postings: Dict[str, List[int]] = {}
    for c in chunks:
        toks = sparse_tokenize(str(c.get("text", "")))
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        term_freqs.append(tf)
        doc_lens.append(len(toks))
        for tok in tf.keys():
            df[tok] = df.get(tok, 0) + 1
            if tok not in postings:
                postings[tok] = []
            postings[tok].append(len(term_freqs) - 1)
    n_docs = max(1, len(term_freqs))
    avg_dl = float(sum(doc_lens) / max(1, len(doc_lens)))
    # BM25 idf
    idf = {tok: float(np.log((n_docs - d + 0.5) / (d + 0.5) + 1.0)) for tok, d in df.items()}

    try:
        with cache_path.open("wb") as f:
            pickle.dump(
                {
                    "source_size": int(src_stat.st_size) if src_stat else -1,
                    "source_mtime_ns": int(src_stat.st_mtime_ns) if src_stat else -1,
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


# Apply the BM25 algorithm for sparse retrieval
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

    # higher k = repeating a word helps more
    k1 = 1.5
    # higher b = longer chunks get penalized more. Not a big issue here since abstracts are generally the same length
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
        if score <= 0.0:
            continue
        scored.append((i, float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]



# Combine dense and sparse scores
def hybrid_fuse_scores(
    dense: List[Tuple[int, float]],   
    sparse: List[Tuple[int, float]],  
    top_k: int,
    w_dense: float = 0.7,
    w_sparse: float = 0.3,
    fuse_limit: int = 200,
    dense_higher_is_better: bool = True, # higher is better due to cosine similarity being applied for Dense
) -> List[int]:


    # Normalize each score according to the minmax algorithm
    def minmax_norm(items: List[Tuple[int, float]], higher_is_better: bool) -> Dict[int, float]:
        if not items:
            return {}
        
        items = items[:fuse_limit]

        idxs = [i for i, _ in items]
        vals = np.array([s for _, s in items], dtype=np.float64)

        
        if not higher_is_better:
            vals = -vals

        lo = float(vals.min())
        hi = float(vals.max())
        if hi == lo:
            
            return {idx: 1.0 for idx in idxs}

        norm = (vals - lo) / (hi - lo)
        return {idxs[j]: float(norm[j]) for j in range(len(idxs))}

    d = minmax_norm(dense, higher_is_better=dense_higher_is_better)
    s = minmax_norm(sparse, higher_is_better=True)

    fused: Dict[int, float] = {}



    # Apply a weighting of dense and sparse to get the final score, then return the top scores
    for idx, v in d.items():
        fused[idx] = fused.get(idx, 0.0) + w_dense * v
    for idx, v in s.items():
        fused[idx] = fused.get(idx, 0.0) + w_sparse * v

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    top_pairs = ranked[:top_k]
    top_ids = [idx for idx, _ in top_pairs]
    top_scores = {idx: float(score) for idx, score in top_pairs}
    return top_ids, top_scores

# Call the groq API given a model, prompt, and the number of output tokens
def call_groq(model: str, prompt: str, num_predict: int) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")

    # Set the temperature to 0 and top_p to 1 to reduce randomness
    # The max_tokens argument is the # of output tokens
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "If the answer is not in the quotes, say \"Not in corpus.\""
                    "Answer at a 8th-grade reading level "
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": int(num_predict),
        "stream": False,
    }
    resp = requests.post(
        f"{GROQ_API_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    if resp.status_code == 404:
        raise RuntimeError("This Groq model is currently down. Please try another model.")
    if resp.status_code == 429:
        raise RuntimeError("Groq rate limit reached (429). Please try another model.")
    resp.raise_for_status()
    data = resp.json()
    content = ""
    if isinstance(data, dict):
        choices = data.get("choices", [])
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message", {})
            content = str(msg.get("content", "")).strip()
    return content

# Return the PubMed URL from a study ID
def canonical_pubmed_url(study_id: str) -> str:
    sid = str(study_id or "").strip()
    sid = re.sub(r"^PMC", "", sid, flags=re.IGNORECASE)
    return f"https://pubmed.ncbi.nlm.nih.gov/{sid}/" if sid else "https://pubmed.ncbi.nlm.nih.gov/"


# Check if a question compares two things
def is_comparative_question(question: str) -> bool:
    q = question.lower()
    if any(token in q for token in (" vs ", " vs. ", " versus ", "compared to", "better than", "worse than")):
        return True
    # "or" often implies options/comparison in this QA workflow.
    return " or " in q

# Define the prompt for summary length based on the number of tokens desired
def summary_length_rules(token_budget: int) -> str:
    # Keep summary short enough to fit the generation cap.
    if token_budget <= 96:
        return "Keep total output <= 70 words. Use 1 short sentence in Conclusion."
    if token_budget <= 160:
        return "Keep total output <= 110 words. Use 1-2 sentences in Conclusion."
    if token_budget <= 256:
        return "Keep total output <= 170 words. Use up to 2 sentences in Conclusion."
    return "Return as many words as necessary to fully answer the question."

# Based on a list of studies, get the title and create a link with the title as the link name
def format_referenced_studies_llm(
    contexts: List[Dict],
) -> str:
    if not contexts:
        return "Not in corpus."
    lines: List[str] = []
    for c in contexts:
        pmid = str(c.get("pmcid", "UNKNOWN")).strip() or "UNKNOWN"
        title = str(c.get("title", "")).strip() or pmid
        url = canonical_pubmed_url(pmid)
        lines.append(f"- [{title}]({url}) ({pmid})")
    return "\n".join(lines)


# This formats the studies into a prompt to ask the LLM
def format_full_abstract_context(contexts: List[Dict], max_studies: int = 5) -> str:
    if not contexts:
        return "Not in corpus."
    blocks: List[str] = []
    for idx, c in enumerate(contexts[:max_studies], start=1):
        pmid = str(c.get("pmcid", "UNKNOWN")).strip() or "UNKNOWN"
        title = str(c.get("title", "")).strip() or pmid
        abstract = str(c.get("text", "")).strip() or "(No abstract text)"
        blocks.append(
            f"Study {idx}\n"
            f"PMID/PMCID: {pmid}\n"
            f"Title: {title}\n"
            f"Abstract:\n{abstract}"
        )
    return "\n\n---\n\n".join(blocks)

# Each sentence is supposed to come with a study number
# This function converts study numbers to links
def convert_study_number_citations_to_links(text: str, contexts: List[Dict]) -> str:
    if not text:
        return text
    refs: Dict[int, Tuple[str, str]] = {}
    # Get the PMID and URL
    for idx, c in enumerate(contexts, start=1):
        pmid = str(c.get("pmcid", "UNKNOWN")).strip() or "UNKNOWN"
        url = canonical_pubmed_url(pmid)
        refs[idx] = (pmid, url)

    # Format the link so the name is the ID and it links to the study

    def study_link(num: int) -> str:
        if num not in refs:
            return f"Study {num}"
        pmid, url = refs[num]
        return f"[{pmid}]({url})"

    # return the study link given a study
    def repl_single(match: re.Match) -> str:
        return study_link(int(match.group(1)))

    # return multiple study links if there are multiple links in a grouping, e.g. <Study 1 Study 2>
    def repl_multi(match: re.Match) -> str:
        nums = [int(n) for n in re.findall(r"Study\s*(\d+)", match.group(0), flags=re.IGNORECASE)]
        if not nums:
            return match.group(0)
        return " ".join(study_link(n) for n in nums)

    # Do a bunch of formatting to ensure that no extra characters are returned and format <Study N> is always replace with a link
    out = re.sub(
        r"[\[<]\s*(?:Study\s*\d+\s*(?:[,;/]\s*Study\s*\d+\s*)+)[\]>]",
        repl_multi,
        text,
        flags=re.IGNORECASE,
    )

    out = re.sub(r"\[\s*Study\s*(\d+)[^\]]*\]", repl_single, out, flags=re.IGNORECASE)

    out = re.sub(r"<\s*Study\s*(\d+)\s*>", repl_single, out, flags=re.IGNORECASE)

    out = re.sub(r"(?<![A-Za-z0-9])Study\s*(\d+)(?![A-Za-z0-9])", repl_single, out, flags=re.IGNORECASE)

    out = re.sub(r"<\s*(\[[^\]]+\]\([^)]+\))\s*>", r"\1", out)
    out = re.sub(r"\[\s*(\[[^\]]+\]\([^)]+\)(?:\s+\[[^\]]+\]\([^)]+\))*)\s*\]", r"\1", out)
    return out


# If the LLM returned extra information, like links, remove them
def strip_redundant_inline_pmc_links(text: str) -> str:
    if not text:
        return text
    out = text
    # Remove HTML line break tags often inserted by model output.
    out = re.sub(r"<\s*br\s*/?\s*>", " ", out, flags=re.IGNORECASE)
    # Remove inline PMC markdown links; study-citation links already cover references.
    out = re.sub(
        r"\[PMC\d+\]\(\s*https?://(?:www\.)?ncbi\.nlm\.nih\.gov/pmc/articles/PMC\d+/?\s*\)",
        "",
        out,
        flags=re.IGNORECASE,
    )
    # Remove inline PMCID parentheticals, e.g. "(PMCID: PMC1234567)" or "(PMCID: 1234567)".
    out = re.sub(r"\(\s*PMCID\s*:\s*[^)]+\)", "", out, flags=re.IGNORECASE)
    # Ensure adjacent study markers/links are separated by a space.
    out = re.sub(r"(</?Study\s*\d+>|(?:\[[^\]]+\]\([^)]+\)))\s*(?=(</?Study\s*\d+>|(?:\[[^\]]+\]\([^)]+\))))", r"\1 ", out, flags=re.IGNORECASE)
    out = re.sub(r"(\[[^\]]+\]\([^)]+\))(?=\[)", r"\1 ", out)
    # Normalize spacing after removals.
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()

# This is the prompt we feed the LLM to elicit a response
def format_summary_prompt(question: str, studies_answer: str, is_comparative: bool, token_budget: int) -> str:
    scope_rule = (
        "If this is not a comparison question, do not introduce unasked alternatives in the conclusion.\n"
        if not is_comparative
        else "If this is a comparison question, compare only the asked options.\n"
    )
    return (
        "STRICT OUTPUT CONTRACT.\n"
        "You must internally check compliance and rewrite until ALL rules pass.\n"
        "Output ONLY the final compliant answer.\n\n"

        "OUTPUT (EXACTLY these 2 sections, in order; no extra text before/after; no blank line after labels):\n"
        "Conclusion: <exactly 3-5 sentences on this same line>\n"
        "Gaps:\n"
        "<gap>\n"
        "<gap>\n\n"
        "Do NOT include anything after this, including citations.  \n\n"

        "CITATIONS (mandatory):\n"
        "1) EVERY Conclusion sentence MUST end with one or more citations immediately BEFORE the final period.\n"
        "   Example: This is a sentence <Study 1>.\n"
        "2) The ONLY allowed citation format is: <Study N> (use provided N).\n"
        "3) Do NOT include any other identifiers (NO PMCID/PMC/DOI/URLs/markdown links).\n\n"

        "CONTENT RULES:\n"
        "1) Use ONLY the studies provided below.\n"
        "2) Cite ONLY relevant studies.\n"
        "3) If any part of the question is missing/indirect/one-sided or evidence is inconclusive: "
        "explicitly say 'partial/incomplete' in the Conclusion, "
        "and list specific missing items in Gaps, one per line.\n"
        "4) If no gaps: in Gaps write exactly:\n"
        "None noted\n"
        "5) In Gaps, do NOT include '-' or bullet characters.\n"
        "6) After the last Gaps line, STOP (do not add anything).\n\n"

        + scope_rule +
        summary_length_rules(token_budget) + "\n"
        f"Question: {question}\n\n"
        f"Studies (do not repeat or list these in output):\n{studies_answer}\n"
    )

# Given the summary, format it into the desired output
def normalize_summary_sections(summary_text: str) -> str:
    # Remove excess asterisks
    text = summary_text.replace("**", "").strip()

    # Return the first match -- otherwise the fallback
    def pick(label_pattern: str, fallback: str = "", flags: int = re.IGNORECASE | re.DOTALL) -> str:
        m = re.search(label_pattern, text, flags=flags)
        if not m:
            return fallback
        return m.group(1).strip()

    # Remove repeated sections, normalize spaces/tabs, keep newlines
    def clean_section(section_text: str) -> str:

        section_text = re.sub(
            r"\n\s*(Conclusion|Gaps)\s*:\s*.*$",
            "",
            section_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        section_text = section_text.replace("*", "")

        section_text = re.sub(r"[ \t]+", " ", section_text)
        section_text = re.sub(r"\n{3,}", "\n\n", section_text)
        return section_text.strip()

    # Format gaps so newlines or hyphens are treated as separate listings
    def format_gaps_bullets(gaps_text: str) -> str:
        raw = gaps_text.replace("*", "").strip()
        if not raw:
            return "- None noted"

        parts = re.split(r"(?:\r?\n|;)+", raw)
        items: List[str] = []
        for ln in parts:
            ln = re.sub(r"^\s*(?:[-•]\s+)", "", ln).strip()  
            if ln:
                items.append(ln)

        if not items:
            return "- None noted"
        if len(items) == 1 and items[0].lower() in ("none", "none noted", "n/a", "na"):
            return "- None noted"
        return "\n".join(f"- {it}" for it in items)

    # Get the conclusion and gaps if possible, then format and return them
    conclusion = pick(
        r"Conclusion\s*:?\s*(.*?)(?=\n\s*Gaps\s*:?\s*|$)",
        fallback="Not clearly stated.",
    )
    gaps = pick(r"Gaps\s*:?\s*(.*)$", fallback="None noted", flags=re.DOTALL)

    conclusion = clean_section(conclusion)
    gaps = format_gaps_bullets(clean_section(gaps))

    return (
        f"**Conclusion:** {conclusion}\n\n"
        f"**Gaps:**\n{gaps}"
    ).strip()


def main():
    st.set_page_config(page_title="AskFit", page_icon="🏋️", layout="wide")
    # HTML/CSS to make the page look good
    st.markdown(
        """
<style>
.stApp {
  background:
    radial-gradient(1200px 500px at 20% -10%, #dcecff 0%, transparent 60%),
    radial-gradient(900px 400px at 100% 0%, #d9f8ed 0%, transparent 60%),
    linear-gradient(180deg, #f5f8fc 0%, #eef3f8 100%);
}
.block-container {padding-top: 1.2rem;}
[data-testid="stMainBlockContainer"] {padding-top: 5.6rem;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #0f2238 0%, #0c1b2f 100%);}
[data-testid="stSidebar"] * {color: #eef5ff;}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: #132a45 !important;
  border: 1px solid #2a4f78 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] input {
  color: #eef5ff !important;
}
div[role="listbox"] ul li,
div[role="listbox"] ul li * {
  color: #10243e !important;
}
[data-testid="stChatMessageContent"] {
  border-radius: 14px; padding: 0.45rem 0.65rem;
}
.top-header-wrap {
  position: fixed;
  top: 5rem;
  left: 0;
  right: 0;
  z-index: 1000;
  pointer-events: none;
}
.top-header-card {
  max-width: 46rem;
  margin: 0 auto;
  padding: 0.6rem 0.9rem;
  border-radius: 14px;
  background: rgba(245, 248, 252, 0.9);
  backdrop-filter: blur(6px);
  box-shadow: 0 6px 18px rgba(16,36,62,0.08);
}
.top-header-title {
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: 0.01em;
  color: #10243e;
  line-height: 1.15;
}
.top-header-subtitle {
  color: #3a5778;
  margin-top: 0.2rem;
}
@media (max-width: 768px) {
  [data-testid="stMainBlockContainer"] {padding-top: 6.5rem;}
  .top-header-wrap {top: 0.35rem;}
  .top-header-card {
    max-width: calc(100vw - 1rem);
    margin: 0 0.5rem;
    padding: 0.55rem 0.65rem;
    border-radius: 12px;
  }
  .top-header-title {font-size: 1.22rem;}
  .top-header-subtitle {font-size: 0.88rem; margin-top: 0.1rem;}
  .block-container {padding-left: 0.65rem; padding-right: 0.65rem;}
}
</style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            '<div class="top-header-wrap">'
            '<div class="top-header-card">'
            '<div class="top-header-title">Search Open Access Resistance Training Studies</div>'
            '<div class="top-header-subtitle">'
            'Ask a training question and get deterministic evidence-backed answers.</div>'
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )

    # Add a sidebar to choose the model
    with st.sidebar:
        st.markdown("### Query Settings")
        profile = SPEED_PROFILES["Quality"]

        available_models = list_groq_models()
        if not available_models:
            st.error("No Groq models found that start with 'groq'.")
            return
        model = st.selectbox("Groq model", options=available_models, index=0)
        response_length = st.selectbox("Response length", options=["Short", "Medium", "Long"], index=1)

        embed_model_name = profile["embed_model"]
        token_level = RESPONSE_LEVEL_MAP[response_length]
        num_predict = TOKEN_LEVELS[token_level]
        summary_predict = SUMMARY_TOKEN_LEVELS[token_level]


        index_path = Path(profile["index_path"])
        chunks_path = Path(profile["chunks_path"])
        index_meta_path = Path(profile["index_meta_path"])
    
    # Define the PMID mapping path
    pmid_map_path = DATA_DIR / "pmid_to_pmcid.jsonl"
    if not index_path.exists() or not chunks_path.exists():
        st.error("Index or chunks missing for the quality profile. Build the corresponding files first.")
        return

    # One-time warmup so first query does not pay full cold-start costs.
    if not st.session_state.get("_warmup_done", False):
        with st.spinner("Warming up retrieval..."):
            load_index(index_path)
            load_index_meta(index_meta_path)
            load_chunks(chunks_path)
            load_sparse_index(chunks_path)
            load_embedder(embed_model_name)
            load_pmid_map(pmid_map_path)
        st.session_state["_warmup_done"] = True
    
    # Defint the chat history and prompt the user

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.chat_input("Ask about resistance training evidence...")
    is_loading_new_query = bool(question and question.strip())

    # Output the history of the chat
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown("**Answer Summary**")
            st.markdown(turn["summary"])
            if "response_time_s" in turn:
                st.caption(f"Response time: {turn['response_time_s']:.2f}s")
            if not is_loading_new_query:
                st.markdown("**Referenced Studies**")
                st.markdown(turn["references"])

    # If there's a question, load the data, check the validity of the model, get the best matches, prompt the LLM, get the answer, and output the answer
    if question and question.strip():
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            query_start = time.perf_counter()
            with st.spinner("Searching and generating..."):
                # Load the data
                comparative = is_comparative_question(question)
                index = load_index(index_path)
                index_meta = load_index_meta(index_meta_path)
                chunks = load_chunks(chunks_path)
                sparse_tf, sparse_doc_lens, sparse_avg_dl, sparse_idf, sparse_postings = load_sparse_index(chunks_path)
                embedder = load_embedder(embed_model_name)
                pmid_map = load_pmid_map(pmid_map_path)

                # Ensure the model is valid and embed the question
                q_emb = embed(embedder, [question])
                if q_emb.shape[1] != index.d:
                    fallback_model = str(index_meta.get("model", "")).strip()
                    if fallback_model and fallback_model != embed_model_name:
                        st.warning(
                            f"Embedding mismatch ({embed_model_name} -> dim {q_emb.shape[1]}, "
                            f"index dim {index.d}). Falling back to index model: {fallback_model}."
                        )
                        embedder = load_embedder(fallback_model)
                        q_emb = embed(embedder, [question])
                    if q_emb.shape[1] != index.d:
                        st.error(
                            f"Embedding dimension mismatch: query dim {q_emb.shape[1]} vs index dim {index.d}. "
                            "Use the same embedding model used to build the index or rebuild the index."
                        )
                        return
                # Get the top k1 dense rankings
                dense_k = 80
                scores, idxs = retrieve(q_emb, index, dense_k)
                dense_ranked: List[Tuple[int, float]] = []
                for i, s in zip(idxs[0].tolist(), scores[0].tolist()):
                    if i < 0:
                        continue
                    dense_ranked.append((int(i), float(s)))

                # Get the top k2 sparse rankings and fuse them with the dense rankings
                sparse_k = 200
                sparse_ranked = sparse_retrieve(
                    question,
                    term_freqs=sparse_tf,
                    doc_lens=sparse_doc_lens,
                    avg_dl=sparse_avg_dl,
                    idf=sparse_idf,
                    postings=sparse_postings,
                    top_k=sparse_k,
                )
                idxs_list, fused_score_map = hybrid_fuse_scores(
                    dense=dense_ranked,
                    sparse=sparse_ranked,
                    top_k=100,
                    fuse_limit=200,
                    dense_higher_is_better=True,
                )
                dense_score_map = {i: s for i, s in dense_ranked}
                sparse_score_map = {i: s for i, s in sparse_ranked}

                # Get the score for each study, get the URL, and save them to a list
                contexts = []
                for i in idxs_list:
                    if i < 0 or i >= len(chunks):
                        continue
                    ctx = dict(chunks[i])
                    ctx["dense_score"] = float(dense_score_map.get(i, 0.0))
                    ctx["sparse_score"] = float(sparse_score_map.get(i, 0.0))
                    ctx["score"] = float(fused_score_map.get(i, 0.0))
                    pmcid_raw = ctx.get("pmcid", "")
                    url = ""
                    resolved = ""
                    if pmcid_raw.startswith("PMC"):
                        resolved = pmcid_raw
                        url = canonical_pubmed_url(resolved)
                    elif pmcid_raw and pmcid_raw in pmid_map and pmid_map[pmcid_raw].startswith("PMC"):
                        resolved = pmid_map[pmcid_raw]
                        url = canonical_pubmed_url(resolved)
                    elif pmcid_raw:
                        url = canonical_pubmed_url(pmcid_raw)
                    ctx["pmcid_resolved"] = resolved
                    ctx["pmcid"] = resolved or pmcid_raw
                    ctx["url"] = url
                    contexts.append(ctx)

                # Get the top 5 studies, formatted, and sort them
                pmc_best: Dict[str, Dict] = {}
                for ctx in contexts:
                    pmcid = ctx.get("pmcid", "")
                    if pmcid not in pmc_best:
                        pmc_best[pmcid] = ctx
                grouped = sorted(
                    pmc_best.values(),
                    key=lambda c: float(c.get("score", 0.0)),
                    reverse=True,
                )
                grouped = grouped[:5]

                # Format the studies, get the LLM response, time the response, and write the reuslts
                answer = format_referenced_studies_llm(
                    contexts=grouped,
                )
                full_abstract_context = format_full_abstract_context(grouped, max_studies=5)
                summary_budget = min(num_predict, summary_predict)
                summary_prompt = format_summary_prompt(
                    question,
                    full_abstract_context,
                    is_comparative=comparative,
                    token_budget=summary_budget,
                )
                
                try:
                    answer_summary = call_groq(model, summary_prompt, num_predict=summary_budget)
                except RuntimeError as e:
                    st.error(str(e))
                    return

                answer_summary = normalize_summary_sections(answer_summary)
                answer_summary = convert_study_number_citations_to_links(answer_summary, grouped)
                answer_summary = strip_redundant_inline_pmc_links(answer_summary)
                response_time_s = time.perf_counter() - query_start
            st.markdown("**Answer Summary**")
            st.markdown(answer_summary)
            st.caption(f"Response time: {response_time_s:.2f}s")
            st.markdown("**Referenced Studies**")
            st.markdown(answer)
        # Add the data to the chat history
        st.session_state.chat_history.append(
            {
                "question": question,
                "summary": answer_summary,
                "references": answer,
                "response_time_s": response_time_s,
            }
        )


if __name__ == "__main__":
    main()
