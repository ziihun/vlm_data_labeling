#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

RAW_ROOT="${1:-raw_datasets/SteelBench}"
SAM_ROOT="${SAM_ROOT:-outputs/SteelBench_sam2}"
SELECTION_FILE="${SELECTION_FILE:-outputs/SteelBench_category5_selection.json}"
NAMING_ROOT="${NAMING_ROOT:-outputs/SteelBench_category5_qwen_naming}"
OUTPUT_DIR="${OUTPUT_DIR:-vqa_datasets/SteelBench_category5}"
JOBS_TSV="$NAMING_ROOT/pending_jobs.tsv"
PER_CATEGORY="${PER_CATEGORY:-5}"
SELECTION_SEED="${SELECTION_SEED:-20261106}"
QWEN_MODEL="${QWEN_MODEL:-mlx-community/Qwen3.5-9B-MLX-4bit}"
OPENAI_MODEL_VALUE="${OPENAI_MODEL:-gpt-6.1-terra}"

[[ -d "$RAW_ROOT/data/frames" ]] || { echo "Missing frames: $RAW_ROOT/data/frames" >&2; exit 1; }
[[ -d "$RAW_ROOT/data/annotations" ]] || { echo "Missing GT: $RAW_ROOT/data/annotations" >&2; exit 1; }
[[ -x .venv/bin/python ]] || { echo "Missing .venv" >&2; exit 1; }
[[ -x .venv-sam2/bin/python ]] || { echo "Missing .venv-sam2" >&2; exit 1; }
[[ -x .venv-qwen/bin/python ]] || { echo "Missing .venv-qwen" >&2; exit 1; }
[[ "$PER_CATEGORY" =~ ^[1-9][0-9]*$ ]] || { echo "PER_CATEGORY must be >= 1" >&2; exit 2; }

mkdir -p "$SAM_ROOT" "$NAMING_ROOT" "$OUTPUT_DIR" logs "$(dirname "$SELECTION_FILE")"

echo "SteelBench category sampling · category x $PER_CATEGORY clips"
echo "========== PHASE 1/5 · FIXED RANDOM CATEGORY SELECTION =========="
if [[ ! -f "$SELECTION_FILE" ]]; then
  .venv/bin/python scripts/select_steelbench_category_clips.py \
    --raw-root "$RAW_ROOT" \
    --sam-root "$SAM_ROOT" \
    --output "$SELECTION_FILE" \
    --per-category "$PER_CATEGORY" \
    --seed "$SELECTION_SEED"
else
  echo "Selection resume · $SELECTION_FILE"
fi

echo "========== PHASE 2/5 · SAM2 SELECTED CLIPS ONLY =========="
.venv-sam2/bin/python workers/sam2_frame_batch.py \
  --raw-root "$RAW_ROOT" \
  --output-root "$SAM_ROOT" \
  --selection-file "$SELECTION_FILE" < /dev/null

echo "========== PHASE 3/5 · QWEN NAMING =========="
.venv/bin/python scripts/prepare_steelbench_naming.py \
  --raw-root "$RAW_ROOT" \
  --sam-root "$SAM_ROOT" \
  --naming-root "$NAMING_ROOT" \
  --jobs-tsv "$JOBS_TSV" \
  --selection-file "$SELECTION_FILE"
if [[ -s "$JOBS_TSV" ]]; then
  .venv-qwen/bin/python workers/steelbench_vlm_name_batch.py \
    --jobs-tsv "$JOBS_TSV" \
    --model "$QWEN_MODEL" \
    --cleanup-inputs < /dev/null
else
  echo "Naming resume · all selected clips completed"
fi

echo "========== PHASE 4/5 · OPENAI VQA FROM QWEN + GT =========="
.venv/bin/python scripts/build_steelbench_vqa.py \
  --raw-root "$RAW_ROOT" \
  --sam-root "$SAM_ROOT" \
  --naming-root "$NAMING_ROOT" \
  --selection-file "$SELECTION_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --model "$OPENAI_MODEL_VALUE" < /dev/null

echo "========== PHASE 5/5 · HUMAN REVIEW =========="
.venv/bin/python scripts/build_steelbench_review.py "$OUTPUT_DIR"

echo "Selection: $SELECTION_FILE"
echo "Naming: $NAMING_ROOT"
echo "VQA JSON: $OUTPUT_DIR"
echo "Review: $OUTPUT_DIR/human_review.html"
