"""
build_pubmed_corpus.py
Converts the JSONL from PubMed into a corpus we can index

The key difference is that we convert the abstract: <abstract> into sections:[{"heading":"Abstract", "text":<text>]


"""

import argparse
import json
from pathlib import Path

# Iterate through and return json
def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main() -> None:
    # Add in the input abstract path as an argument, as well as the path for the output corpus
    parser = argparse.ArgumentParser(description="Build pmc_corpus.jsonl from PubMed abstract JSONL")
    parser.add_argument(
        "--in-jsonl",
        default="pubmed_resistance_training_abstracts_edirect.jsonl",
        help="Input PubMed abstract JSONL",
    )
    parser.add_argument("--out-corpus", default="pmc_corpus.jsonl", help="Output corpus JSONL")
    args = parser.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_corpus)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    written = 0
    skipped = 0
    seen = set()
    # For each record, get the PMID and abstract, and create a JSON with the new formatting

    with out_path.open("w", encoding="utf-8") as f_out:
        for rec in iter_jsonl(in_path):
            pmid = str(rec.get("pmid", "")).strip()
            abstract = str(rec.get("abstract") or "").strip()
            if not pmid or not abstract:
                skipped += 1
                continue
            if pmid in seen:
                skipped += 1
                continue
            seen.add(pmid)
            row = {
                "pmcid": pmid,
                "pmid": pmid,
                "title": str(rec.get("title", "")).strip(),
                "journal": str(rec.get("journal", "")).strip(),
                "year": str(rec.get("year", "")).strip() or None,
                "sections": [{"heading": "Abstract", "text": abstract}],
            }
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} corpus rows to {out_path} (skipped {skipped})")


if __name__ == "__main__":
    main()
