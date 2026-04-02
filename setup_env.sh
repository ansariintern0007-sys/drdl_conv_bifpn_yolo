#!/bin/bash
# ============================================================
# setup_env.sh — Create isolated Python 3.10 venv for weld AI
# Run with: bash setup_env.sh
# ============================================================
set -e

PROJECT_DIR="/media/aid-pc/My1TB_2/Swin Yolo Model"

# --- 1. Deactivate any active conda environment ---------------
echo "[1/5] Deactivating conda environments..."
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda deactivate 2>/dev/null || true
    conda deactivate 2>/dev/null || true
fi
echo "       Done."

# --- 2. Create Python 3.10 venv ------------------------------
echo "[2/5] Creating Python 3.10 virtual environment..."
cd "$PROJECT_DIR"
python3.10 -m venv weld_ai_env
source weld_ai_env/bin/activate
echo "       Python: $(python --version)  |  Pip: $(pip --version | awk '{print $2}')"

# --- 3. Upgrade pip ------------------------------------------
echo "[3/5] Upgrading pip..."
pip install --upgrade pip setuptools wheel -q

# --- 4. Install core packages --------------------------------
echo "[4/5] Installing core ML packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
pip install ultralytics -q
pip install opencv-python-headless -q
pip install numpy pandas scikit-learn tqdm -q
pip install matplotlib seaborn -q
pip install pylabel -q
pip install openmim -q
pip install Pillow -q
pip install pyyaml -q
pip install timm einops -q

# --- 5. Install MMDet stack -----------------------------------
echo "[5/5] Installing MMEngine / MMCV / MMDet..."
mim install mmengine -q
mim install "mmcv>=2.0.0" -q
mim install mmdet -q

echo ""
echo "============================================"
echo " Environment setup complete!"
echo " Activate with: source '$PROJECT_DIR/weld_ai_env/bin/activate'"
echo "============================================"
