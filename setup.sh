#!/bin/bash
# KaVa Project Setup Script for HPC
# Run this script after uploading to HPC: bash setup.sh

set -e  # Exit on error

echo "=========================================="
echo "KaVa Project Setup"
echo "=========================================="
echo ""

# Get project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

echo "Project directory: $PROJECT_DIR"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p outputs
mkdir -p cache
mkdir -p .huggingface
mkdir -p data
echo "✓ Directories created"
echo ""

# Check Python version
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "✗ Python not found. Please load Python module first:"
    echo "  module load python/3.11"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version)
echo "✓ Found: $PYTHON_VERSION"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Check if GPU is available
echo "Checking GPU availability..."
$PYTHON_CMD -c "import torch; print('✓ PyTorch installed'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count()}')" || echo "⚠ Could not check GPU status"
echo ""

# Check Hugging Face login status
echo "Checking Hugging Face authentication..."
if command -v hf &> /dev/null; then
    if hf auth whoami &> /dev/null; then
        echo "✓ Already logged in to Hugging Face"
        hf auth whoami
    else
        echo "⚠ Not logged in to Hugging Face"
        echo "Please run: hf auth login"
    fi
else
    echo "⚠ Hugging Face CLI not found"
    echo "Install with: pip install huggingface_hub"
fi
echo ""

# Make scripts executable
echo "Setting script permissions..."
chmod +x scripts/*.sh 2>/dev/null || true
chmod +x scripts/*.py 2>/dev/null || true
echo "✓ Scripts are executable"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Login to Hugging Face (if not already):"
echo "   hf auth login"
echo ""
echo "2. Test the installation:"
echo "   python -c 'from transformers import AutoTokenizer; print(\"✓ Setup OK\")'"
echo ""
echo "3. Submit a training job:"
echo "   sbatch scripts/run_hpc_training.sh"
echo ""
echo "4. Or run all experiments:"
echo "   sbatch scripts/run_all_experiments.sh"
echo ""
echo "5. Monitor jobs:"
echo "   squeue -u \$USER"
echo "   tail -f logs/train_*.out"
echo ""
