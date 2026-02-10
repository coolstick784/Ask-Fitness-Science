"""
build_sparse_cache.py

Build the sparse cache once, so the app initial loading time decreases
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import load_sparse_index


def main() -> None:
    data_dir = ROOT / "pipeline-data"
    for chunks_name in ("pmc_chunks.jsonl",):
        chunks_path = data_dir / chunks_name
        print(f"Building sparse cache for {chunks_path} ...")
        try:
            term_freqs, doc_lens, avg_dl, idf, postings = load_sparse_index(chunks_path)
            cache_path = chunks_path.with_suffix(".sparse.pkl")
            print(
                f"Done: {cache_path} | docs={len(term_freqs)} | "
                f"avg_dl={avg_dl:.2f} | vocab={len(idf)} | postings_terms={len(postings)}"
            )
        except FileNotFoundError:
            print(f"Skipped: missing {chunks_path} and split parts.")


if __name__ == "__main__":
    main()
