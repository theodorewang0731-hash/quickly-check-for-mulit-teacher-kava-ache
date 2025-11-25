#!/bin/bash
# Quick test script to verify setup
# Usage: bash test_setup.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_DIR"

echo "Testing KaVa setup..."
echo ""

# Activate venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "✗ Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Test imports
echo ""
echo "Testing Python imports..."
python3 << 'PYEOF'
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✓ torch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"✗ torch import failed: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ transformers import failed: {e}")
    sys.exit(1)

try:
    import datasets
    print(f"✓ datasets {datasets.__version__}")
except ImportError as e:
    print(f"✗ datasets import failed: {e}")
    sys.exit(1)

try:
    import accelerate
    print(f"✓ accelerate {accelerate.__version__}")
except ImportError as e:
    print(f"✗ accelerate import failed: {e}")
    sys.exit(1)

try:
    import numpy
    print(f"✓ numpy {numpy.__version__}")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")
    sys.exit(1)

print("\n✓ All required packages are installed")
PYEOF

# Test project modules
echo ""
echo "Testing project modules..."
python3 -c "from experiments.kv_utils import full_kv; print('✓ kv_utils')"
python3 -c "from experiments.kv_loss import compute_kv_loss; print('✓ kv_loss')"
python3 -c "from experiments.projector import StudentToTeacherProjector; print('✓ projector')"

echo ""
echo "=========================================="
echo "✓ Setup test passed!"
echo "=========================================="
echo ""
echo "You can now submit training jobs:"
echo "  sbatch scripts/run_hpc_training.sh"
echo "  sbatch scripts/run_multi_teacher.sh"
echo ""
