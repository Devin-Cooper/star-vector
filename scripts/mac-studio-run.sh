#!/bin/bash
# Mac Studio StarVector Runner
# This script makes it easy to run StarVector with MPS acceleration on Mac Studio

# Make executable with: chmod +x scripts/mac-studio-run.sh
# Run with: ./scripts/mac-studio-run.sh [mode]

# Check if script is running in a conda environment
if [[ -z $CONDA_DEFAULT_ENV ]]; then
  echo "ERROR: This script must be run within the conda environment."
  echo "Please run:"
  echo "  conda activate starvector"
  echo "  ./scripts/mac-studio-run.sh [mode]"
  exit 1
fi

echo "Running in conda environment: $CONDA_DEFAULT_ENV"

# Disable Flash Attention for Mac Studio
export STARVECTOR_DISABLE_FLASH_ATTN=1
# This environment variable will be checked by the model to disable flash attention

# Check if MPS is available
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Get the mode from command line argument
MODE=${1:-"inference"}

case $MODE in
  "inference")
    echo "===== Running inference with MPS on Mac Studio ====="
    python scripts/quickstart.py
    ;;
    
  "inference-hf")
    echo "===== Running HuggingFace inference with MPS on Mac Studio ====="
    python scripts/quickstart-hf.py
    ;;
    
  "train")
    echo "===== Training with MPS on Mac Studio ====="
    accelerate launch --config_file configs/accelerate/mac-studio.yaml \
      starvector/train/train.py \
      config=configs/models/starvector-1b/im2svg-stack-mac.yaml
    ;;
    
  "validate")
    echo "===== Validating with MPS on Mac Studio ====="
    python starvector/validation/validate.py \
      config=configs/generation/hf/mac-studio/im2svg.yaml \
      dataset.name=starvector/svg-stack
    ;;
    
  "demo")
    echo "===== Running demo with MPS on Mac Studio ====="
    echo "Step 1: Starting controller..."
    python -m starvector.serve.controller --host 0.0.0.0 --port 10000 &
    CONTROLLER_PID=$!
    sleep 2
    
    echo "Step 2: Starting web server..."
    python -m starvector.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload --port 7000 &
    WEBSERVER_PID=$!
    sleep 2
    
    echo "Step 3: Starting model worker with MPS..."
    python -m starvector.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path starvector/starvector-1b-im2svg --device mps
    
    # Cleanup when the model worker exits
    kill $CONTROLLER_PID
    kill $WEBSERVER_PID
    ;;
    
  "app")
    echo "===== Running desktop app with MPS on Mac Studio ====="
    python starvector_app.py
    ;;
    
  *)
    echo "Unknown mode: $MODE"
    echo "Available modes: inference, inference-hf, train, validate, demo, app"
    exit 1
    ;;
esac 