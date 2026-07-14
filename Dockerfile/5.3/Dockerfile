FROM quay.io/jupyter/scipy-notebook:python-3.11

ARG JUPYTERHUB_VERSION=5.4.4

# ─────────────────────────────────────────────────────────────────────────────
# Lecture 05 — Edge Computing in AI. CPU-ONLY workshop (no GPU / CUDA). Datasets
# are tiny and ship in the repo (delivered via nbgitpuller); the only heavy pulls
# are the two models NB01 uses, which we precache below.
# ─────────────────────────────────────────────────────────────────────────────

# wget: NB03 fetches a few small cluster-trace files at runtime.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends wget \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /opt/hf-cache && chown -R ${NB_UID}:${NB_GID} /opt/hf-cache
USER ${NB_UID}

# Install CPU-only PyTorch first so the (huge, unused) CUDA build isn't pulled — this
# is an edge/CPU workshop. `2.5.1+cpu` satisfies the `torch==2.5.1` pin in requirements.
RUN pip3 install --no-cache torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip3 install --no-cache -r requirements.txt

RUN pip3 install --no-cache \
    jupyterhub==$JUPYTERHUB_VERSION \
    nbgitpuller

# Precache the models NB01 pulls at runtime so students don't each download ~1.7 GB
# mid-class: Gemma-2-2b GGUF (llama.cpp, CPU) + all-MiniLM-L6-v2 embeddings. Both are
# public (bartowski re-upload) — no HF token/gating. Cache lives under /opt so a future
# home-volume mount can't shadow it.
ENV HF_HOME=/opt/hf-cache
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('bartowski/gemma-2-2b-it-GGUF', 'gemma-2-2b-it-Q4_K_M.gguf')" \
 && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
