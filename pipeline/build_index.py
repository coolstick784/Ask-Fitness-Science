"""

build_index.py

Build a FAISS index for a dense search from the abstracts
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

#load the corpus
def load_corpus(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]



# Add one chunk for each article into JSON format
def chunk_one_per_article(record: Dict) -> List[Dict]:
    pmcid = record.get("pmcid", "")
    title = record.get("title", "")
    year = record.get("year")
    sections = record.get("sections", []) or []
    abstract_text = ""
    for sec in sections:
        heading = str(sec.get("heading", "")).strip().lower()
        text = str(sec.get("text", "")).strip()
        if not text:
            continue
        if "abstract" in heading:
            abstract_text = text
            break
        if not abstract_text:
            abstract_text = text
    if not abstract_text:
        return []
    return [
        {
            "pmcid": pmcid,
            "pmcid_resolved": pmcid if str(pmcid).startswith("PMC") else "",
            "title": title,
            "year": year,
            "heading": "Abstract",
            "sec_idx": 0,
            "start_word": 0,
            "text": abstract_text,
            "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if str(pmcid).startswith("PMC") else "",
        }
    ]

#Convert the text into an embedding vector
def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embs.astype("float32")


# Build a FAISS index with GPU if possible (otherwise CPU) with cosine similarity
# FAISS allows searches to be more efficient
def build_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)
    try:
        # Try FAISS GPU if available
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(vectors)
        return gpu_index
    except Exception:
        cpu_index.add(vectors)
        return cpu_index


# Save the index to the path
def save_index(index: faiss.Index, path: Path) -> None:
    try:
        index = faiss.index_gpu_to_cpu(index)
    except Exception:
        pass
    faiss.write_index(index, str(path))


def main():
    # Add arguments with paths, target words, model names, and the batch size
    parser = argparse.ArgumentParser(description="Build FAISS index for PMC corpus")
    parser.add_argument("--corpus", default="pmc_corpus.jsonl", help="Path to corpus JSONL")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5", help="SentenceTransformer model name")
    parser.add_argument("--target_words", type=int, default=320, help="Words per chunk")
    parser.add_argument("--overlap", type=int, default=80, help="Word overlap between chunks")

    parser.add_argument("--out_index", default="pmc_faiss.index", help="Output FAISS index file")
    parser.add_argument("--out_chunks", default="pmc_chunks.jsonl", help="Output chunks JSONL")
    parser.add_argument("--out_meta", default="pmc_index_meta.json", help="Output index meta JSON")
    parser.add_argument("--batch", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    # Load the corpus
    corpus_path = Path(args.corpus)
    records = load_corpus(corpus_path)
    print(f"Loaded {len(records)} records from {corpus_path}")

    all_chunks: List[Dict] = []
    # Create the chunks from the corpus and save them
    for rec in records:

        all_chunks.extend(chunk_one_per_article(rec))
    print(f"Created {len(all_chunks)} chunks.")
    if not all_chunks:
        raise RuntimeError("No chunks created. Check corpus content (missing abstract text).")

    chunks_path = Path(args.out_chunks)
    with chunks_path.open("w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"Wrote chunks to {chunks_path}")

    # Convert the chunks to vectors and build the index

    model = SentenceTransformer(args.model)
    vectors = embed_texts(model, [c["text"] for c in all_chunks], batch_size=args.batch)
    print(f"Embeddings shape: {vectors.shape}")

    index = build_index(vectors)
    index_path = Path(args.out_index)
    save_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")

    # Save the index
    meta = {
        "model": args.model,
        "dim": int(vectors.shape[1]),
        "chunks": len(all_chunks),
        "target_words": args.target_words,
        "overlap": args.overlap,
        "one_chunk_per_article": True,
        "corpus": str(corpus_path),
        "chunks_path": str(chunks_path),
        "index_path": str(index_path),
    }
    Path(args.out_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Meta saved to {args.out_meta}")


if __name__ == "__main__":
    main()
