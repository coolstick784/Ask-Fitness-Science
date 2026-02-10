"""
End-to-end pipeline.
This scrapes the data, builds a corpus, and builds the index
"""



import argparse
import gzip
import shutil
import subprocess
import sys
import time
from pathlib import Path


# This function runs a step in the pipeline, printing the process and time
def run_step(name: str, cmd: list[str], cwd: Path) -> None:
    print(f"\n=== {name} ===")
    print(" ".join(cmd))
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.time() - start
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode} after {elapsed:.1f}s")
    print(f"{name} completed in {elapsed:.1f}s")


def gzip_file(path: Path) -> Path:
    gz_path = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as src, gzip.open(gz_path, "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)
    return gz_path


def main() -> None:
    
    parser = argparse.ArgumentParser(description="Run ask_science pipeline")


    parser.add_argument("--input-jsonl", default="pubmed_resistance_training_abstracts_edirect.jsonl", help="Input abstracts JSONL")
    parser.add_argument("--corpus", default="pmc_corpus.jsonl", help="Output corpus JSONL")
    args = parser.parse_args()

    # The pipeline directory for data is pipeline-data
    pipeline_dir = Path(__file__).resolve().parent
    root_dir = pipeline_dir.parent
    data_dir = root_dir / "pipeline-data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Scrape data from PubMed's API using WSL
    # Use WSL because PubMed's scraper is not available for Windows
    py = sys.executable
    scrape_wsl_path = "/mnt/c/Users/cools/ask_science/pipeline/scrape_abstract.py"
    run_step(
        "Scrape abstracts",
        [
            "wsl",
            "python3",
            scrape_wsl_path,
        ],
        cwd=pipeline_dir,
    )


    input_jsonl = data_dir / args.input_jsonl
    corpus_path = data_dir / args.corpus
    std_index = data_dir / "pmc_faiss.index"
    std_chunks = data_dir / "pmc_chunks.jsonl"
    std_meta = data_dir / "pmc_index_meta.json"


    if not input_jsonl.exists():
        raise FileNotFoundError(f"Missing input JSONL after scrape step: {input_jsonl}")

    # Build the corpus
    run_step(
        "Build corpus",
        [
            py,
            "build_pubmed_corpus.py",
            "--in-jsonl",
            str(input_jsonl),
            "--out-corpus",
            str(corpus_path),
        ],
        cwd=pipeline_dir,
    )

    # Build standard index only.
    run_step(
        "Build standard index (Balanced/Quality)",
        [
            py,
            "build_index.py",
            "--corpus",
            str(corpus_path),
            "--model",
            "BAAI/bge-large-en-v1.5",
            "--out_index",
            str(std_index),
            "--out_chunks",
            str(std_chunks),
            "--out_meta",
            str(std_meta),
        ],
        cwd=pipeline_dir,
    )
    # Write compressed copies for easier upload/archival.
    compressed_input = gzip_file(input_jsonl)
    compressed_corpus = gzip_file(corpus_path)
    compressed_chunks = gzip_file(std_chunks)

    # Remove stale fast-profile artifacts if they exist.
    for stale in (
        data_dir / "pmc_faiss_fast.index",
        data_dir / "pmc_chunks_fast.jsonl",
        data_dir / "pmc_index_meta_fast.json",
        data_dir / "pmc_chunks_fast.jsonl.gz",
    ):
        if stale.exists():
            stale.unlink()

    print("\nPipeline completed.")
    print(f"Compressed: {compressed_input.name}, {compressed_corpus.name}, {compressed_chunks.name}")
    print("Run the app with: streamlit run app/app.py")


if __name__ == "__main__":
    main()
