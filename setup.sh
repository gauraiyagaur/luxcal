#!/usr/bin/env bash
set -euo pipefail

# LUXCAL environment setup. Run once from an empty project directory.

echo "==> Installing uv (fast Python package manager)"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

echo "==> Creating virtual environment (Python 3.11)"
uv venv --python 3.11
source .venv/bin/activate

echo "==> Installing dependencies"
uv pip install \
  anthropic \
  "pydantic>=2.7" pydantic-settings \
  langgraph langgraph-checkpoint-sqlite \
  pyyaml tenacity python-dotenv \
  pandas pyarrow scipy statsmodels krippendorff \
  matplotlib seaborn \
  pytest

uv pip freeze > requirements.txt

echo "==> Scaffolding directories"
mkdir -p luxcal/{agents,core,retrieval,logging_,rubric/dimensions} \
         configs data/cases scripts tests runs analysis
find luxcal -type d -exec touch {}/__init__.py \;

echo "==> Writing .gitignore"
cat > .gitignore <<'GIT'
.venv/
__pycache__/
*.pyc
.env
runs/
analysis/*.parquet
.pytest_cache/
GIT

echo "==> Writing .env template"
cat > .env.example <<'ENV'
ANTHROPIC_API_KEY=sk-ant-...
ENV
cp .env.example .env

echo "==> Initialising git"
git init -q 2>/dev/null || true
git add -A 2>/dev/null || true

echo
echo "Done. Next:"
echo "  1. Put your API key in .env"
echo "  2. Copy CLAUDE.md and SPEC.md into this directory"
echo "  3. Open in VS Code:  code ."
echo "  4. Start Claude Code: claude"
