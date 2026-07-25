#!/usr/bin/env bash
# One-time environment setup. Works on a GCP Deep Learning VM (PyTorch pre-installed)
# and on a bare Ubuntu/Debian box. Creates a venv that INHERITS any system PyTorch
# (so we don't reinstall it), installs the project deps, and verifies the GPU.
#   bash setup.sh   &&   source ~/brats-env/bin/activate
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV="${HOME}/brats-env"

echo "=== venv at ${ENV} (inherits system site-packages, incl. pre-installed torch) ==="
python3 -m venv "${ENV}" --system-site-packages
# shellcheck disable=SC1091
source "${ENV}/bin/activate"
pip install --upgrade pip wheel

if python -c "import torch" 2>/dev/null; then
  echo "=== PyTorch already present ($(python -c 'import torch;print(torch.__version__)')) — not reinstalling ==="
else
  echo "=== Installing PyTorch (CUDA 12.4; switch to cu121/cu126 to match the host) ==="
  pip install torch --index-url https://download.pytorch.org/whl/cu124
fi

echo "=== Installing project requirements + synapseclient ==="
pip install -r "${PROJ}/requirements.txt" synapseclient

echo "=== Verifying ==="
python - <<'PY'
import torch, monai
print("torch :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
print("monai :", monai.__version__)
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo
echo "Done. Activate in every new shell with:  source ${ENV}/bin/activate"
