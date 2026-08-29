#!/usr/bin/env bash
# Runs ON the pod. Idempotent: builds a venv on the persistent /workspace
# volume (inheriting the image's CUDA torch via --system-site-packages) and
# installs the training deps. Re-runs are fast no-ops.
set -euo pipefail

# GPU preflight FIRST: on some community hosts nvidia-smi works but every
# CUDA app fails (cuInit 999, host needs reboot). Exit 42 so pod.py knows to
# TERMINATE this pod (a stopped pod restarts on the same broken host).
python3 - <<'EOF'
import ctypes, sys
rc = ctypes.CDLL("libcuda.so.1").cuInit(0)
if rc != 0:
    print(f"FATAL: broken GPU on this host (cuInit -> {rc})", file=sys.stderr)
    sys.exit(42)
print("GPU preflight ok (cuInit 0)")
EOF

VENV=/workspace/venv
if [ ! -f "$VENV/bin/activate" ]; then
  echo "creating venv at $VENV (system-site-packages for CUDA torch)"
  python -m venv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"

# Pinned to versions compatible with the image's torch 2.4 — transformers
# releases past ~4.5x assume torch >= 2.6 and fail at import.
pip install --quiet "transformers==4.46.3" "peft==0.14.0" "accelerate==1.2.1" matplotlib

mkdir -p /workspace/hf
python - <<'EOF'
import torch
print(f"bootstrap ok: torch {torch.__version__}, cuda={torch.cuda.is_available()}",
      f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu'})")
import transformers, peft
print(f"transformers {transformers.__version__}, peft {peft.__version__}")
EOF
