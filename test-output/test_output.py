"""
test_output.py
Run this script on every push to ensure that function calls work correctly
"""


import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Get the root directory
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Import functions and variables from app.py
from app import (
    GROQ_API_KEY,
    QUALITY_PROFILE,
    SUMMARY_TOKEN_LEVELS,
    TOKEN_LEVELS,
    call_groq,
    canonical_pubmed_url,
    convert_study_number_citations_to_links,
    embed,
    format_full_abstract_context,
    format_referenced_studies_llm,
    format_summary_prompt,
    hybrid_fuse_scores,
    is_comparative_question,
    list_groq_models,
    load_chunks,
    load_embedder,
    load_index,
    load_index_meta,
    load_pmid_map,
    load_sparse_index,
    normalize_summary_sections,
    retrieve,
    sparse_retrieve,
    strip_redundant_inline_pmc_links,
)

# For each study, grab the dense score and sparse score
# Combine them using a weighting of dense and sparse scores
# Create a list of dictionaries, contexts, that has information for each study, as well as its scores
# Return the 5 most relevant studies
def resolve_contexts(
    question: str,
    chunks: List[Dict],
    pmid_map: Dict[str, str],
    dense_ranked: List[Tuple[int, float]],
    sparse_ranked: List[Tuple[int, float]],
) -> List[Dict]:
    idxs_list, fused_score_map = hybrid_fuse_scores(
        dense=dense_ranked,
        sparse=sparse_ranked,
        top_k=100,
        fuse_limit=200,
        dense_higher_is_better=True,
    )
    dense_score_map = {i: s for i, s in dense_ranked}
    sparse_score_map = {i: s for i, s in sparse_ranked}

    contexts: List[Dict] = []
    for i in idxs_list:
        if i < 0 or i >= len(chunks):
            continue
        ctx = dict(chunks[i])
        ctx["dense_score"] = float(dense_score_map.get(i, 0.0))
        ctx["sparse_score"] = float(sparse_score_map.get(i, 0.0))
        ctx["score"] = float(fused_score_map.get(i, 0.0))
        pmcid_raw = str(ctx.get("pmcid", "")).strip()
        resolved = ""
        if pmcid_raw.startswith("PMC"):
            resolved = pmcid_raw
        elif pmcid_raw and pmcid_raw in pmid_map and str(pmid_map[pmcid_raw]).startswith("PMC"):
            resolved = str(pmid_map[pmcid_raw]).strip()
        ctx["pmcid_resolved"] = resolved
        ctx["pmcid"] = resolved or pmcid_raw
        ctx["url"] = canonical_pubmed_url(ctx["pmcid"])
        contexts.append(ctx)

    pmc_best: Dict[str, Dict] = {}
    for ctx in contexts:
        pmcid = str(ctx.get("pmcid", "")).strip()
        if pmcid and pmcid not in pmc_best:
            pmc_best[pmcid] = ctx
    grouped = sorted(pmc_best.values(), key=lambda c: float(c.get("score", 0.0)), reverse=True)
    return grouped[:5]



def main() -> None:
    # Pass arguments, like question and model
    parser = argparse.ArgumentParser(description="Smoke-test app retrieval + LLM output.")
    parser.add_argument("--question", default="what depth is best for squats")
    parser.add_argument("--model", default="", help="Optional Groq model override.")
    args = parser.parse_args()

    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")

    cwd = ROOT
    # Set the profile, model name, token levels, and various paths
    profile = QUALITY_PROFILE
    embed_model_name = profile["embed_model"]
    num_predict = TOKEN_LEVELS["High"]
    summary_predict = SUMMARY_TOKEN_LEVELS["High"]
    index_path = cwd / profile["index_path"]
    chunks_path = cwd / profile["chunks_path"]
    index_meta_path = cwd / profile["index_meta_path"]
    pmid_map_path = cwd / "pipeline-data" / "pmid_to_pmcid.jsonl"
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(f"Missing required files: {index_path} and/or {chunks_path}")

    # Set the model to the first model if it isn't passed in
    model = args.model.strip()
    if not model:
        models = list_groq_models()
        if not models:
            raise RuntimeError("No Groq models found (filtered to names starting with 'groq').")
        model = models[0]

    
    # Load the necessary inputs and start a counter

    t0 = time.perf_counter()
    index = load_index(index_path)
    index_meta = load_index_meta(index_meta_path)
    chunks = load_chunks(chunks_path)
    sparse_tf, sparse_doc_lens, sparse_avg_dl, sparse_idf, sparse_postings = load_sparse_index(chunks_path)
    embedder = load_embedder(embed_model_name)
    pmid_map = load_pmid_map(pmid_map_path)

    # Ensure that the model embedder size is valid
    q_emb = embed(embedder, [args.question])
    if q_emb.shape[1] != index.d:
        fallback_model = str(index_meta.get("model", "")).strip()
        if fallback_model and fallback_model != embed_model_name:
            embedder = load_embedder(fallback_model)
            q_emb = embed(embedder, [args.question])
        if q_emb.shape[1] != index.d:
            raise RuntimeError(f"Embedding dimension mismatch: query={q_emb.shape[1]} index={index.d}")
    
    # Return the top 200 dense and 500 sparse studies

    dense_k = 200
    scores, idxs = retrieve(q_emb, index, dense_k)
    dense_ranked: List[Tuple[int, float]] = [
        (int(i), float(s))
        for i, s in zip(idxs[0].tolist(), scores[0].tolist())
        if i >= 0
    ]
    sparse_k = 500
    sparse_ranked = sparse_retrieve(
        args.question,
        term_freqs=sparse_tf,
        doc_lens=sparse_doc_lens,
        avg_dl=sparse_avg_dl,
        idf=sparse_idf,
        postings=sparse_postings,
        top_k=sparse_k,
    )
    # Get the combined top 5 of the sparse/dense rankings
    grouped = resolve_contexts(args.question, chunks, pmid_map, dense_ranked, sparse_ranked)

    # Format the studies, get the LLM output, and format the response

    refs = format_referenced_studies_llm(grouped)
    summary_budget = min(num_predict, summary_predict)
    summary_prompt = format_summary_prompt(
        args.question,
        format_full_abstract_context(grouped, max_studies=5),
        is_comparative=is_comparative_question(args.question),
        token_budget=summary_budget,
    )
    summary = call_groq(model, summary_prompt, num_predict=summary_budget)
    summary = normalize_summary_sections(summary)
    summary = convert_study_number_citations_to_links(summary, grouped)
    summary = strip_redundant_inline_pmc_links(summary)
    elapsed = time.perf_counter() - t0

    print(f"Question: {args.question}")
    print(f"Model: {model}")
    print(f"Elapsed: {elapsed:.2f}s")
    print("\n=== Answer Summary ===")
    print(summary)
    print("\n=== Referenced Studies ===")
    print(refs)


if __name__ == "__main__":
    main()
