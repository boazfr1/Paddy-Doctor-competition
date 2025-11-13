#!/bin/bash

# Paddy Doctor Disease Classification - Setup and Run Script
# This script sets up the virtual environment, installs dependencies, and runs the application

set -e  # Exit on any error

echo "🌾 Paddy Doctor Disease Classification - Setup and Run Script"
echo "============================================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install required packages from requirements.txt
echo "📥 Installing required packages from requirements.txt..."
pip install -r requirements.txt

echo "✅ All packages installed successfully"

# Check if Kaggle credentials exist
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "⚠️  Warning: Kaggle credentials not found at ~/.kaggle/kaggle.json"
    echo "   If you need to download data from Kaggle, please:"
    echo "   1. Go to https://www.kaggle.com/account"
    echo "   2. Create a new API token"
    echo "   3. Place the kaggle.json file in ~/.kaggle/"
    echo "   4. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
else
    echo "✅ Kaggle credentials found"
fi

# Check if paddy dataset directory exists
if [ -d "paddy-disease-classification" ] && [ -d "paddy-disease-classification/paddy-disease-classification/train_images" ]; then
    echo "✅ Paddy disease dataset already downloaded"
else
    echo "📥 Paddy disease dataset will be downloaded automatically when running the script"
fi

# Check if models directory exists, create if not
if [ ! -d "models" ]; then
    echo "📁 Creating models directory..."
    mkdir models
    echo "✅ Models directory created"
else
    echo "✅ Models directory already exists"
fi

# Check for GPU/MPS availability
echo "🔍 Checking for GPU/MPS availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print('✅ CUDA GPU available:', torch.cuda.get_device_name(0))
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✅ Apple Silicon MPS available')
else:
    print('⚠️  No GPU acceleration available, using CPU')
"

# Run the application
echo "🚀 Running the Paddy Doctor Disease Classification model..."
echo "=========================================================="
python paddy.py

echo ""
echo "✅ Script completed successfully!"
echo "🎯 Check the output above to see the model performance:"
echo "   - Training progress with error rates"
echo "   - Learning rate finder results"
echo "   - Final validation error rate"
echo "   - TTA (Test Time Augmentation) results"
echo ""
echo "💡 The model uses ConvNeXt architecture with the following features:"
echo "   - 3 disease classes: hispa, normal, tungro"
echo "   - Image size: 224x224 pixels"
echo "   - Data augmentation with transforms"
echo "   - Mixed precision training (FP16)"
echo "   - Automatic learning rate finding"
echo ""
echo "📊 Model files are saved in the 'models/' directory"
