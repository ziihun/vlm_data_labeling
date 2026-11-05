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
OUTPUT_DIR="${OUTPUT_DIR:-vqa_datasets/SteelBench}"
SAM_OUTPUT_DIR="${SAM_OUTPUT_DIR:-outputs/SteelBench_sam2}"
MAX_CLIPS="${MAX_CLIPS:-0}"
START_AFTER="${START_AFTER:-}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_SAM_MASKS="${KEEP_SAM_MASKS:-0}"

[[ -d "$RAW_ROOT/data/frames" ]] || { echo "Missing frames: $RAW_ROOT/data/frames" >&2; exit 1; }
[[ -d "$RAW_ROOT/data/annotations" ]] || { echo "Missing annotations: $RAW_ROOT/data/annotations" >&2; exit 1; }
[[ "$MAX_CLIPS" =~ ^[0-9]+$ ]] || { echo "MAX_CLIPS must be >= 0" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR" "$SAM_OUTPUT_DIR" logs
SAM_ARGS=(
  --raw-root "$RAW_ROOT"
  --output-root "$SAM_OUTPUT_DIR"
  --max-clips "$MAX_CLIPS"
  --start-after "$START_AFTER"
)
VQA_ARGS=(
  --raw-root "$RAW_ROOT"
  --sam-root "$SAM_OUTPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --model "${OPENAI_MODEL:-gpt-6.1-terra}"
  --max-clips "$MAX_CLIPS"
  --start-after "$START_AFTER"
)
if [[ "$DRY_RUN" == "1" ]]; then
  SAM_ARGS+=(--dry-run)
  VQA_ARGS+=(--dry-run)
fi
if [[ "$KEEP_SAM_MASKS" == "1" ]]; then
  SAM_ARGS+=(--keep-masks)
fi

echo "SteelBench automation · SAM2 → OpenAI VQA (8 frames + GT) · resume enabled"
echo "========== PHASE 1/2 · SAM2 ON 8 ORDERED FRAMES =========="
.venv-sam2/bin/python workers/sam2_frame_batch.py "${SAM_ARGS[@]}" < /dev/null

echo "========== PHASE 2/2 · OPENAI VQA FROM SAM2 + HUMAN GT =========="
.venv/bin/python scripts/build_steelbench_vqa.py "${VQA_ARGS[@]}" < /dev/null

if [[ "$DRY_RUN" != "1" ]]; then
  .venv/bin/python scripts/build_steelbench_review.py "$OUTPUT_DIR"
  echo "JSON: $OUTPUT_DIR"
  echo "Review: $OUTPUT_DIR/human_review.html"
fi
