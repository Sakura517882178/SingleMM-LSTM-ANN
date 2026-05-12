#!/bin/bash
set -e
cd "$(dirname "$0")"

# ---- defaults ----
TRAIN_LR=0.003
TRAIN_MLR=0.001
FT_LR=0.0001
FT_MLR=0.001
BATCH_SIZES="16 32 64"
FREEZE=""
# -------------------

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --train_lr LR        Training AdamW lr       (default: 0.003)"
    echo "  --train_mlr LR       Training memristor lr   (default: 0.001)"
    echo "  --ft_lr LR           Finetune AdamW lr       (default: 1e-4)"
    echo "  --ft_mlr LR          Finetune memristor lr   (default: 0.001)"
    echo "  --bs 'SZ1 SZ2 ...'   Batch sizes to sweep    (default: '16 32 64')"
    echo "  --freeze             Freeze non-memristor in finetune"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --train_lr)  TRAIN_LR="$2"; shift 2 ;;
        --train_mlr) TRAIN_MLR="$2"; shift 2 ;;
        --ft_lr)     FT_LR="$2";    shift 2 ;;
        --ft_mlr)    FT_MLR="$2";   shift 2 ;;
        --bs)        BATCH_SIZES="$2"; shift 2 ;;
        --freeze)    FREEZE="--freeze"; shift ;;
        -h|--help)   usage ;;
        *) echo "Unknown: $1"; usage ;;
    esac
done

echo "============================================"
echo "Step 1/4: Training"
echo "  lr=$TRAIN_LR  memristor_lr=$TRAIN_MLR  bs=$BATCH_SIZES"
echo "============================================"
python training-ANN-M.py --lr "$TRAIN_LR" --memristor_lr "$TRAIN_MLR" --batch_sizes $BATCH_SIZES

echo ""
echo "============================================"
echo "Step 2/4: Finetuning"
echo "  lr=$FT_LR  memristor_lr=$FT_MLR  bs=$BATCH_SIZES  freeze=${FREEZE:-no}"
echo "============================================"
python finetune-ANN-M.py --lr "$FT_LR" --memristor_lr "$FT_MLR" --batch_sizes $BATCH_SIZES $FREEZE

echo ""
echo "============================================"
echo "Step 3/4: Inference"
echo "============================================"
python infer-ANN-M.py

echo ""
echo "============================================"
echo "Step 4/4: Confusion Matrix"
echo "============================================"
python CMX.py

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "  Model:            checkpoints-AM-Uni/best_memristor_lstm_attn_ft.pth"
echo "  Predictions:      predictions-ANN-M.csv"
echo "  Confusion Matrix: confusion_matrix-ANN-M.csv"
echo "============================================"
