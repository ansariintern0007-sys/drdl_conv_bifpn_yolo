#!/bin/bash
# ============================================================
# run_pipeline.sh — Master script to run the full pipeline
# ============================================================
set -e

PROJECT="/media/aid-pc/My1TB_2/Swin Yolo Model"
SCRIPTS="$PROJECT/scripts"
VENV="$PROJECT/weld_ai_env"

# --- Activate environment ---
echo "Activating virtual environment..."
source "$VENV/bin/activate"
echo "Python: $(python --version)"

# --- Step 2-9: Process datasets ---
echo ""
echo "════════════════════════════════════════════════════════"
echo "  PHASE 1: Dataset Processing (Steps 2-9)"
echo "════════════════════════════════════════════════════════"
python "$SCRIPTS/process_datasets.py"

# --- Step 10-12: Training ---
echo ""
echo "════════════════════════════════════════════════════════"
echo "  PHASE 2: Training (Steps 10-12)"
echo "════════════════════════════════════════════════════════"
python "$SCRIPTS/train_weld_detector.py"

# --- Evaluation ---
echo ""
echo "════════════════════════════════════════════════════════"
echo "  PHASE 3: Evaluation"
echo "════════════════════════════════════════════════════════"
python "$SCRIPTS/evaluate_model.py"

echo ""
echo "✅ Pipeline complete!"
