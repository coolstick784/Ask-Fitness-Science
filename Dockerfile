FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System packages needed for build/runtime and PubMed EDirect.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    perl \
    && rm -rf /var/lib/apt/lists/*

# Install NCBI EDirect (esearch/efetch used by scrape_abstract.py).
# Use pipefail so curl/install failures fail the build.
# Do not symlink esearch/efetch to /usr/local/bin: EDirect resolves helper scripts
# (like ecommon.sh) relative to its own install directory.
RUN bash -o pipefail -lc "curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh | bash" \
    && test -x /root/edirect/esearch \
    && test -f /root/edirect/ecommon.sh
ENV PATH="/root/edirect:${PATH}"

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/scripts/run_all_scripts.sh

EXPOSE 8501

# Optional env toggles:
# - RUN_PUBLISH_HF=1 requires HF_TOKEN + HF_REPO_ID
# - RUN_ONES_NOT_FOUND=1 requires reachable Ollama endpoint
# - RUN_AUTO_GRADE=1 requires reachable Ollama endpoint
# - RUN_COMPARE_AGREEMENT=1 requires agreement file generated beforehand
# - START_APP=0 disables launching Streamlit at the end
CMD ["/app/scripts/run_all_scripts.sh"]
