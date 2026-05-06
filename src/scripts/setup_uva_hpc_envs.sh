#!/usr/bin/env bash
set -euo pipefail

# Bootstraps two isolated Python envs using uv.
# Tested on SLURM clusters with CUDA 12.x. Adjust module names for your system.
#
# Usage:
#   module load cuda/12.4.1 gcc/11.4.0
#   bash scripts/setup_uva_hpc_envs.sh

CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.1}"
GCC_MODULE="${GCC_MODULE:-gcc/11.4.0}"

GEN_ENV="${GEN_ENV:-odpo-gen}"
TRAIN_ENV="${TRAIN_ENV:-odpo-train}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

ENVS_DIR="${ENVS_DIR:-${HOME}/conda_envs}"
GEN_DIR="${ENVS_DIR}/${GEN_ENV}"
TRAIN_DIR="${ENVS_DIR}/${TRAIN_ENV}"
GEN_PY="${GEN_DIR}/bin/python"
TRAIN_PY="${TRAIN_DIR}/bin/python"

module purge
module load "${CUDA_MODULE}"
module load "${GCC_MODULE}"

# Install uv if not already available
if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> uv version: $(uv --version)"

# ── odpo-gen (generation with vLLM) ──────────────────────────────────────────
echo "==> Creating ${GEN_ENV}..."
uv venv "${GEN_DIR}" --python "${PYTHON_VERSION}"

echo "==> Installing packages into ${GEN_ENV}..."
uv pip install --python "${GEN_PY}" setuptools wheel
uv pip install --python "${GEN_PY}" datasets
uv pip install --python "${GEN_PY}" vllm==0.5.4
uv pip install --python "${GEN_PY}" accelerate==0.33.0
uv pip install --python "${GEN_PY}" deepspeed==0.14.5
uv pip install --python "${GEN_PY}" transformers==4.48.1
uv pip install --python "${GEN_PY}" numpy==1.26.4
uv pip install --python "${GEN_PY}" \
    antlr4-python3-runtime==4.7.2 sympy==1.12 latex2sympy2==1.9.1 word2number==1.1
uv pip install --python "${GEN_PY}" regex pebble

# The 'pyairports' on PyPI (0.0.1) is a stub with no actual module.
# outlines (vllm dep) needs pyairports.airports.AIRPORT_LIST, but we never
# do airport structured generation, so an empty list satisfies the import.
PYAIRPORTS_DIR="${GEN_DIR}/lib/python${PYTHON_VERSION}/site-packages/pyairports"
mkdir -p "${PYAIRPORTS_DIR}"
touch "${PYAIRPORTS_DIR}/__init__.py"
echo "AIRPORT_LIST = []" > "${PYAIRPORTS_DIR}/airports.py"

# ── odpo-train (training with DeepSpeed + TRL) ───────────────────────────────
echo "==> Creating ${TRAIN_ENV}..."
uv venv "${TRAIN_DIR}" --python "${PYTHON_VERSION}"

echo "==> Installing packages into ${TRAIN_ENV}..."
uv pip install --python "${TRAIN_PY}" "setuptools<70" wheel
# Pin numpy<2 — torch 2.1.2 was compiled against NumPy 1.x
uv pip install --python "${TRAIN_PY}" "numpy<2"
# psutil is an undeclared build dep of flash-attn; must be present before --no-build-isolation
uv pip install --python "${TRAIN_PY}" psutil
uv pip install --python "${TRAIN_PY}" \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121
# --no-build-isolation lets flash-attn see the already-installed torch + setuptools
uv pip install --python "${TRAIN_PY}" flash-attn==2.6.3 --no-build-isolation
uv pip install --python "${TRAIN_PY}" accelerate==0.33.0
uv pip install --python "${TRAIN_PY}" huggingface-hub==0.24.7
uv pip install --python "${TRAIN_PY}" transformers==4.42.2
uv pip install --python "${TRAIN_PY}" peft==0.7.1
uv pip install --python "${TRAIN_PY}" deepspeed==0.15.4
uv pip install --python "${TRAIN_PY}" trl==0.9.6
uv pip install --python "${TRAIN_PY}" wandb datasets rich
# alignment-handbook has dep conflicts with our peft/accelerate pins; --no-deps is intentional
uv pip install --python "${TRAIN_PY}" alignment-handbook --no-deps
uv pip install --python "${TRAIN_PY}" regex pebble

echo ""
echo "==> Done. Sanity checks:"
echo "  ${GEN_PY} -c 'import vllm, transformers; print(vllm.__version__, transformers.__version__)'"
echo "  ${TRAIN_PY} -c 'import torch, transformers, trl; print(torch.__version__, transformers.__version__, trl.__version__)'"
