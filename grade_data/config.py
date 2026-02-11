"""
config.py
Generate hardcoded paths, models, the Ollama API, and top K values
"""
import os
from pathlib import Path

GRADE_DIR = Path(__file__).resolve().parent
ROOT_DIR = GRADE_DIR.parent
DATA_DIR = ROOT_DIR / "pipeline-data"

DEFAULT_INDEX = DATA_DIR / "pmc_faiss.index"
DEFAULT_CHUNKS = DATA_DIR / "pmc_chunks.jsonl"
DEFAULT_META = DATA_DIR / "pmc_index_meta.json"
DEFAULT_PMID_MAP = DATA_DIR / "pmid_to_pmcid.jsonl"

DEFAULT_QS = GRADE_DIR / "test_questions.jsonl"
DEFAULT_LLM_EVAL_RESULTS = GRADE_DIR / "llm_eval_results.jsonl"
DEFAULT_AGREEMENT_RESULTS = GRADE_DIR / "llm_grade_agreement.jsonl"
DEFAULT_MANUAL_GRADES = GRADE_DIR / "manual_grades.jsonl"

DEFAULT_ANSWER_MODEL = "qwen3-coder:latest"
DEFAULT_GRADER_MODEL = "qwen3-coder:latest"
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434").rstrip("/")

DEFAULT_DENSE_K = 80
DEFAULT_SPARSE_K = 200
DEFAULT_TOP_K = 100
