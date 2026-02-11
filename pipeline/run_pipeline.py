"""
End-to-end pipeline.
This scrapes the data, builds a corpus, and builds the index
"""



import argparse
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


def to_wsl_path(path: Path) -> str:
    raw = str(path.resolve())
    # Convert Windows absolute path like C:\x\y to /mnt/c/x/y for WSL.
    if len(raw) >= 3 and raw[1] == ":" and raw[2] in ("\\", "/"):
        drive = raw[0].lower()
        rest = raw[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return raw.replace("\\", "/")


def main() -> None:
    
    parser = argparse.ArgumentParser(description="Run ask_science pipeline")


    parser.add_argument("--input-jsonl", default="pubmed_resistance_training_abstracts_edirect.jsonl", help="Input abstracts JSONL")
    parser.add_argument("--corpus", default="pmc_corpus.jsonl", help="Output corpus JSONL")
    parser.add_argument("--publish-to-hf", action="store_true", help="Upload generated artifacts to HF dataset")
    parser.add_argument("--hf-repo-id", default="coolstick/Ask-Fitness-Science", help="HF dataset repo id, e.g. user/repo")
    parser.add_argument("--hf-token-env", default="HF_TOKEN", help="Environment variable name for HF token")
    args = parser.parse_args()

    # The pipeline directory for data is pipeline-data
    pipeline_dir = Path(__file__).resolve().parent
    root_dir = pipeline_dir.parent
    data_dir = root_dir / "pipeline-data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Scrape data from PubMed's API.
    # On Windows, use WSL for EDirect. On Linux/macOS (including Docker), run directly.
    py = sys.executable
    scrape_script = pipeline_dir / "scrape_abstract.py"
    if sys.platform.startswith("win"):
        scrape_cmd = ["wsl", "python3", to_wsl_path(scrape_script)]
    else:
        scrape_cmd = [py, str(scrape_script)]
    run_step(
        "Scrape abstracts",
        scrape_cmd,
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
            "BAAI/bge-base-en-v1.5",
            "--out_index",
            str(std_index),
            "--out_chunks",
            str(std_chunks),
            "--out_meta",
            str(std_meta),
        ],
        cwd=pipeline_dir,
    )
    # Remove stale split/compressed artifacts if they exist.
    for stale in data_dir.glob("*.part*"):
        stale.unlink()
    for stale in data_dir.glob("*.gz"):
        stale.unlink()
    for stale in (
        data_dir / "pmc_faiss_fast.index",
        data_dir / "pmc_chunks_fast.jsonl",
        data_dir / "pmc_index_meta_fast.json",
    ):
        if stale.exists():
            stale.unlink()

    if args.publish_to_hf:
        run_step(
            "Publish artifacts to Hugging Face",
            [
                py,
                "publish_to_hf.py",
                "--repo-id",
                args.hf_repo_id.strip(),
                "--data-dir",
                str(data_dir),
                "--token-env",
                args.hf_token_env,
            ],
            cwd=pipeline_dir,
        )

    print("\nPipeline completed.")
    if args.publish_to_hf:
        print(f"Published artifacts to HF dataset: {args.hf_repo_id}")
    print("Run the app with: streamlit run app/app.py")


if __name__ == "__main__":
    main()
