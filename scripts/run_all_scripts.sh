#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== Running pipeline ==="
python pipeline/run_pipeline.py

echo "=== Building sparse cache ==="
python pipeline/build_sparse_cache.py

if [[ "${RUN_PUBLISH_HF:-0}" == "1" ]]; then
  : "${HF_REPO_ID:?HF_REPO_ID is required when RUN_PUBLISH_HF=1}"
  echo "=== Publishing artifacts to Hugging Face ==="
  python pipeline/publish_to_hf.py \
    --repo-id "${HF_REPO_ID}" \
    --data-dir "${ROOT_DIR}/pipeline-data" \
    --token-env HF_TOKEN
fi

if [[ -f "grade_data/test_questions.jsonl" ]]; then
  echo "=== Running retrieval grading ==="
  python grade_data/grade_searcher.py
else
  echo "Skipping grade_searcher.py (missing grade_data/test_questions.jsonl)"
fi

if [[ "${RUN_ONES_NOT_FOUND:-0}" == "1" ]]; then
  echo "=== Running ones_not_found ==="
  python grade_data/ones_not_found.py --save-details grade_data/ones_not_found_details.jsonl
else
  echo "Skipping ones_not_found.py (set RUN_ONES_NOT_FOUND=1 to enable)"
fi

if [[ "${RUN_AUTO_GRADE:-0}" == "1" ]]; then
  echo "=== Running auto LLM grading ==="
  python grade_data/auto_grade_llm.py
else
  echo "Skipping auto_grade_llm.py (set RUN_AUTO_GRADE=1 to enable)"
fi

if [[ "${RUN_COMPARE_AGREEMENT:-0}" == "1" ]]; then
  echo "=== Running agreement comparison ==="
  python grade_data/compare_agreement.py
else
  echo "Skipping compare_agreement.py (set RUN_COMPARE_AGREEMENT=1 to enable)"
fi

echo "Batch run complete."

if [[ "${START_APP:-1}" == "1" ]]; then
  PORT="${PORT:-8501}"
  echo "=== Starting Streamlit app on 0.0.0.0:${PORT} ==="
  exec streamlit run app/app.py --server.address=0.0.0.0 --server.port="${PORT}"
else
  echo "START_APP=0, exiting after batch run."
fi
