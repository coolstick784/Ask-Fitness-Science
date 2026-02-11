"""
prompt_templates.py
"""
from typing import Dict, List


def answer_prompt(question: str, contexts: List[Dict]) -> str:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        sid = str(c.get("pmcid", "")).strip()
        title = str(c.get("title", "")).strip()
        abstract = str(c.get("text", "")).strip()
        blocks.append(f"Study {i}\nStudy ID: {sid}\nTitle: {title}\nAbstract:\n{abstract}")
    studies = "\n\n---\n\n".join(blocks) if blocks else "No studies."
    return (
        "Use only the provided studies to answer.\n"
        "Return concise evidence-grounded text with citations like <Study 1>.\n\n"
        f"Question: {question}\n\nStudies:\n{studies}\n"
    )


def grader_prompt(question: str, answer: str, contexts: List[Dict]) -> str:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        sid = str(c.get("pmcid", "")).strip()
        title = str(c.get("title", "")).strip()
        abstract = str(c.get("text", "")).strip()
        blocks.append(f"Study {i}\nStudy ID: {sid}\nTitle: {title}\nAbstract:\n{abstract}")
    studies = "\n\n---\n\n".join(blocks) if blocks else "No studies."
    return (
        "You are grading an LLM answer against provided studies.\n"
        "Output JSON only with keys: chunks_sources_good, support, answer_quality, notes.\n"
        "support must be one of: supported, partially supported, not supported.\n"
        "answer_quality must be 1, 2, or 3.\n\n"
        f"Question:\n{question}\n\nAnswer:\n{answer}\n\nStudies:\n{studies}\n"
    )


def top10_judge_prompt(question: str, top_contexts: List[Dict]) -> str:
    blocks = []
    for i, c in enumerate(top_contexts, start=1):
        sid = str(c.get("pmcid", "")).strip()
        title = str(c.get("title", "")).strip()
        abstract = str(c.get("text", "")).strip()
        blocks.append(f"Study {i}\nStudy ID: {sid}\nTitle: {title}\nAbstract:\n{abstract}")
    studies = "\n\n---\n\n".join(blocks) if blocks else "No studies."
    return (
        "Determine if any one of these 10 studies are relevant to the question. They do not need to directly answer the question,"
        "but their information should be helpful to the user asking it. Only 1 of the 10 (or more) needs to match the criteria \n"
        "Return JSON only: {\"top10_answers_question\": true/false, \"reason\": \"...\"}\n\n"
        f"Question:\n{question}\n\nTop 10 studies:\n{studies}\n"
    )
