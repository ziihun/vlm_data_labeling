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
NAMING_ROOT="${NAMING_ROOT:-outputs/SteelBench_qwen_naming}"
OUTPUT_DIR="${OUTPUT_DIR:-vqa_datasets/SteelBench}"
JOBS_TSV="$NAMING_ROOT/pending_jobs.tsv"
QWEN_MODEL="${QWEN_MODEL:-mlx-community/Qwen3.5-9B-MLX-4bit}"
OPENAI_MODEL_VALUE="${OPENAI_MODEL:-gpt-6.1-terra}"
MAX_CLIPS="${MAX_CLIPS:-0}"
START_AFTER="${START_AFTER:-}"

[[ -d "$RAW_ROOT/data/frames" ]] || { echo "Missing frames: $RAW_ROOT/data/frames" >&2; exit 1; }
[[ -d "$RAW_ROOT/data/annotations" ]] || { echo "Missing GT: $RAW_ROOT/data/annotations" >&2; exit 1; }
[[ -d "$SAM_ROOT" ]] || { echo "Missing completed SAM outputs: $SAM_ROOT" >&2; exit 1; }
[[ -x .venv/bin/python ]] || { echo "Missing .venv" >&2; exit 1; }
[[ -x .venv-qwen/bin/python ]] || { echo "Missing .venv-qwen" >&2; exit 1; }
[[ "$MAX_CLIPS" =~ ^[0-9]+$ ]] || { echo "MAX_CLIPS must be >= 0" >&2; exit 2; }

mkdir -p "$NAMING_ROOT" "$OUTPUT_DIR" logs

echo "SteelBench completed-SAM pipeline · Qwen naming → GT fusion → OpenAI VQA"
echo "========== PHASE 1/4 · PREPARE COMPLETED SAM CLIPS =========="
.venv/bin/python scripts/prepare_steelbench_naming.py \
  --raw-root "$RAW_ROOT" \
  --sam-root "$SAM_ROOT" \
  --naming-root "$NAMING_ROOT" \
  --jobs-tsv "$JOBS_TSV" \
  --max-clips "$MAX_CLIPS" \
  --start-after "$START_AFTER"

echo "========== PHASE 2/4 · QWEN TRACK NAMING =========="
if [[ -s "$JOBS_TSV" ]]; then
  .venv-qwen/bin/python workers/steelbench_vlm_name_batch.py \
    --jobs-tsv "$JOBS_TSV" \
    --model "$QWEN_MODEL" \
    --cleanup-inputs < /dev/null
else
  echo "Naming resume · all completed"
fi

echo "========== PHASE 3/4 · OPENAI VQA FROM QWEN + GT =========="
.venv/bin/python scripts/build_steelbench_vqa.py \
  --raw-root "$RAW_ROOT" \
  --sam-root "$SAM_ROOT" \
  --naming-root "$NAMING_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --model "$OPENAI_MODEL_VALUE" \
  --max-clips "$MAX_CLIPS" \
  --start-after "$START_AFTER" < /dev/null

echo "========== PHASE 4/4 · HUMAN REVIEW =========="
.venv/bin/python scripts/build_steelbench_review.py "$OUTPUT_DIR"

echo "Naming: $NAMING_ROOT"
echo "VQA JSON: $OUTPUT_DIR"
echo "Review: $OUTPUT_DIR/human_review.html"
