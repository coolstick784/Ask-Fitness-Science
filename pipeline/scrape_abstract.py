"""
scrape_abstract.py
This scrapes the information for the project from PubMed
"""

import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional



# These are all the terms we want
# We did not only include MeSH Terms because some studies weren't indexed
TERM = (
    "("
    '"Resistance Training"[MeSH Terms] OR '
    '"Weight Lifting"[MeSH Terms] OR '
    '"Exercise Therapy"[MeSH Terms] OR '
    '"resistance training"[Title/Abstract] OR '
    '"weight training"[Title/Abstract] OR '
    '"strength training"[Title/Abstract] OR '
    '"resistance exercise"[Title/Abstract] OR '
    '"strength exercise"[Title/Abstract] OR '
    '"weight lifting"[Title/Abstract] OR '
    '"weightlifting"[Title/Abstract] OR '
    '"progressive resistance"[Title/Abstract] OR '
    '"progressive overload"[Title/Abstract] OR '
    '"muscle strengthening"[Title/Abstract] OR '
    '"neuromuscular training"[Title/Abstract] OR '
    '"power training"[Title/Abstract] OR '
    '"hypertrophy training"[Title/Abstract] OR '
    '"free weights"[Title/Abstract] OR '
    '"resistance band*"[Title/Abstract] OR '
    '"strength and conditioning"[Title/Abstract] OR '
    '"functional training"[Title/Abstract] OR '
    '"squat"[Title/Abstract] OR '
    '"deadlift"[Title/Abstract] OR '
    '"bench press"[Title/Abstract]'
    ")"
)


# This is the output file
OUT_JSONL = Path(__file__).resolve().parent.parent / "pipeline-data" / "pubmed_resistance_training_abstracts_edirect.jsonl"

# Read the data in batches of 1,000 to avoid memory issues
BATCH_SIZE = 1000

# Matches any &...; that is NOT one of the 5 predefined XML entities
# and NOT a numeric entity like &#123; or &#x1A;
_BAD_ENTITY = re.compile(r"&(?!(amp|lt|gt|quot|apos);|#\d+;|#x[0-9A-Fa-f]+;)")



def sanitize_xml_entities(xml_text: str) -> str:
    # Turn "&nbsp;" into "&amp;nbsp;" so XML parser won't choke.
    return _BAD_ENTITY.sub("&amp;", xml_text)



# Run the scraping command
def run_cmd(cmd: str) -> str:
    res = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            "EDirect pipeline failed.\n"
            f"CMD:\n{cmd}\n"
            f"STDERR:\n{res.stderr}\n"
            f"STDOUT:\n{res.stdout}\n"
        )
    return res.stdout


# Run the command to get the IDs for the studies we want
def fetch_pmids(term: str) -> List[str]:
    out = run_cmd(f"esearch -db pubmed -query '{term}' | efetch -format uid")
    pmids: List[str] = []
    for line in out.splitlines():
        p = line.strip()
        if p.isdigit():
            pmids.append(p)
    return pmids

# Fetch the XML with the abstract for each ID we want
def fetch_batch_xml(pmids: List[str]) -> str:
    ids = ",".join(pmids)
    xml_text = run_cmd(f"efetch -db pubmed -id '{ids}' -format xml")
    return sanitize_xml_entities(xml_text)


def write_articles_from_xml(xml_text: str, out_file, written: int) -> int:
    root = ET.fromstring(xml_text)
    for pm_article in root.findall(".//PubmedArticle"):
        pmid = (pm_article.findtext(".//MedlineCitation/PMID") or "").strip()
        if not pmid:
            continue

        title = (pm_article.findtext(".//Article/ArticleTitle") or "").strip()
        journal = (pm_article.findtext(".//Article/Journal/Title") or "").strip()
        year = extract_year(pm_article)
        abstract = extract_abstract(pm_article)

        out_file.write(
            json.dumps(
                {
                    "pmid": pmid,
                    "year": year,
                    "journal": journal,
                    "title": title,
                    "abstract": abstract,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        written += 1
    return written

# Convert the text to a string
def _text(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()

# Get the year from the XML
def extract_year(article: ET.Element) -> Optional[str]:
    for path in [
        ".//JournalIssue/PubDate/Year",
        ".//ArticleDate/Year",
        ".//PubDate/Year",
    ]:
        val = article.findtext(path)
        if val:
            return val.strip()

    medline = article.findtext(".//JournalIssue/PubDate/MedlineDate")
    if medline:
        for tok in medline.split():
            if tok.isdigit() and len(tok) == 4:
                return tok
    return None


# Get the abstract from the XML
def extract_abstract(article: ET.Element) -> Optional[str]:
    elems = article.findall(".//Abstract/AbstractText")
    if not elems:
        return None

    parts = []
    for el in elems:
        seg = _text(el)
        if not seg:
            continue
        label = el.attrib.get("Label")
        parts.append(f"{label}: {seg}" if label else seg)

    return "\n".join(parts) if parts else None


def main():
    # Get the IDs
    pmids = fetch_pmids(TERM)
    total = len(pmids)
    print(f"Found {total} PMIDs")

    written = 0
    # Query the IDs, save the information, and dump it to the JSON
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for start in range(0, total, BATCH_SIZE):
            batch_pmids = pmids[start : start + BATCH_SIZE]
            try:
                xml_text = fetch_batch_xml(batch_pmids)
                written = write_articles_from_xml(xml_text, f, written)
            except ET.ParseError:
                print(
                    "Warning: malformed batch XML; retrying PMIDs individually "
                    f"for batch starting at index {start}."
                )
                for one_pmid in batch_pmids:
                    try:
                        one_xml = fetch_batch_xml([one_pmid])
                        written = write_articles_from_xml(one_xml, f, written)
                    except Exception:
                        print(f"Warning: skipping PMID {one_pmid} (failed to fetch/parse).")
            except Exception:
                print(
                    "Warning: batch fetch failed; retrying PMIDs individually "
                    f"for batch starting at index {start}."
                )
                for one_pmid in batch_pmids:
                    try:
                        one_xml = fetch_batch_xml([one_pmid])
                        written = write_articles_from_xml(one_xml, f, written)
                    except Exception:
                        print(f"Warning: skipping PMID {one_pmid} (failed to fetch/parse).")

            print(f"Processed {min(start + BATCH_SIZE, total)}/{total} PMIDs | wrote {written}")

    print(f"Wrote {written} records to {OUT_JSONL}")


if __name__ == "__main__":
    main()
