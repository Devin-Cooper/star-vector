#!/bin/bash
# Mac Studio StarVector Installation Script

# Check if running in conda environment
if [[ -z $CONDA_DEFAULT_ENV ]]; then
  echo "ERROR: This script must be run within the conda environment."
  echo "Please run:"
  echo "  conda activate starvector"
  echo "  ./mac_install.sh"
  exit 1
fi

echo "==== Installing StarVector for Mac Studio ===="
echo "This script will install StarVector with MPS support for your Mac Studio."

# Disable Flash Attention for Mac
export STARVECTOR_DISABLE_FLASH_ATTN=1
echo "Flash Attention disabled for Mac compatibility"

# Login to HuggingFace
echo "Authenticating with HuggingFace..."
export HF_TOKEN=""
# Install the CLI if not already installed
pip install -q huggingface_hub
# Login with token
echo -e "y\n$HF_TOKEN" | huggingface-cli login --token $HF_TOKEN

# Add conda-forge channel and install dependencies 
echo "Installing conda-forge dependencies..."
conda config --add channels conda-forge
conda config --set channel_priority flexible
conda install -y cairo pango pkg-config

# Install dependencies from mac-specific requirements
echo "Installing Python dependencies..."
pip install -r mac_requirements.txt

# Ensure cairocffi is installed with conda packages
pip uninstall -y cairocffi cairosvg
pip install cairocffi cairosvg

# Backup original pyproject.toml
echo "Backing up original configuration..."
if [ -f "pyproject.toml.original" ]; then
  echo "Backup already exists, using it"
else
  cp pyproject.toml pyproject.toml.original
fi

# Replace with Mac-specific pyproject.toml
echo "Configuring for Mac..."
cp pyproject_mac.toml pyproject.toml

# Install package in development mode
echo "Installing StarVector package..."
pip install -e .

# Create necessary directories
echo "Creating required directories..."
mkdir -p configs/generation/hf/mac-studio

# Restore original pyproject.toml
echo "Restoring original configuration..."
mv pyproject.toml.original pyproject.toml

# Verify MPS is available
echo "Checking MPS availability..."
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Test CairoSVG
echo "Testing CairoSVG..."
python -c "import cairosvg; print('CairoSVG import successful!')"

# Test SVG processing
echo "Testing SVG processing..."
cat > test.svg << EOF
<svg width="100" height="100">
  <circle cx="50" cy="50" r="40" stroke="black" stroke-width="3" fill="red" />
</svg>
EOF
python -c "import cairosvg; cairosvg.svg2png(url='test.svg', write_to='test.png'); print('SVG processing successful!')"
rm test.svg test.png

# Test StarVector import
echo "Testing starvector import..."
python -c "from starvector.model.starvector_arch import StarVectorForCausalLM; print('Import successful!')"

# Set permanent environment variables for this conda env
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo '#!/bin/bash
export STARVECTOR_DISABLE_FLASH_ATTN=1
' > $CONDA_PREFIX/etc/conda/activate.d/starvector-env.sh
chmod +x $CONDA_PREFIX/etc/conda/activate.d/starvector-env.sh

echo "==== Installation Complete ===="
echo "To run StarVector on your Mac Studio:"
echo "  ./scripts/mac-studio-run.sh [mode]"
echo "Available modes: inference, inference-hf, train, validate, demo, app"

# Make the run script executable
chmod +x scripts/mac-studio-run.sh 