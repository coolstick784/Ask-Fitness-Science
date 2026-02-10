"""
Publish pipeline artifacts to a Hugging Face dataset repo.
"""

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload pipeline-data artifacts to Hugging Face dataset")
    parser.add_argument("--repo-id", required=True, help="HF dataset repo id, e.g. user/repo")
    parser.add_argument("--data-dir", required=True, help="Local pipeline-data directory")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable containing HF token")
    args = parser.parse_args()

    token = os.getenv(args.token_env, "").strip()
    if not token:
        raise RuntimeError(f"Missing {args.token_env} environment variable.")

    from huggingface_hub import HfApi

    data_dir = Path(args.data_dir)
    required = [
        "pmc_faiss.index",
        "pmc_chunks.jsonl",
        "pmc_index_meta.json",
    ]
    optional = [
        "pmid_to_pmcid.jsonl",
        "pmc_corpus.jsonl",
        "pubmed_resistance_training_abstracts_edirect.jsonl",
    ]
    files = [f for f in required if (data_dir / f).exists()]
    missing_required = [f for f in required if f not in files]
    if missing_required:
        raise FileNotFoundError(f"Missing required files for upload: {missing_required}")
    files.extend([f for f in optional if (data_dir / f).exists()])

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    for name in files:
        path = data_dir / name
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded: {name}")
    print(f"Done. Uploaded {len(files)} file(s) to dataset: {args.repo_id}")


if __name__ == "__main__":
    main()

