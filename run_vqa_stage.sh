#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

usage() {
  echo "Usage: $0 {sam|naming|vqa} <video_name_or_mp4_path>" >&2
  echo "Example: $0 sam 0_te1" >&2
  echo "Then:    $0 naming 0_te1" >&2
  echo "Finally: $0 vqa 0_te1" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage
STAGE="$1"
VIDEO_INPUT="$2"
DATA_ROOT="${DATA_ROOT:-raw_datasets/Safe_and_Unsafe_Behaviours/data}"

if [[ "$VIDEO_INPUT" == /* ]]; then
  VIDEO_PATH="$VIDEO_INPUT"
elif [[ -f "$VIDEO_INPUT" ]]; then
  VIDEO_PATH="$(realpath "$VIDEO_INPUT")"
else
  VIDEO_NAME="${VIDEO_INPUT%.mp4}"
  VIDEO_PATH="$DATA_ROOT/$VIDEO_NAME.mp4"
fi

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Video not found: $VIDEO_PATH" >&2
  exit 1
fi
VIDEO_PATH="$(realpath "$VIDEO_PATH")"

VIDEO_FILE="$(basename "$VIDEO_PATH")"
VIDEO_STEM="${VIDEO_FILE%.*}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${VIDEO_STEM}_sam2}"
NAMING_ROOT="$OUTPUT_ROOT/naming_runs/qwen3_5_9b_detailed"
NAMING_MODEL="${NAMING_MODEL:-mlx-community/Qwen3.5-9B-MLX-4bit}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-6.1-terra}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.7}"

case "$STAGE" in
  sam)
    echo "[SAM] $VIDEO_PATH -> $OUTPUT_ROOT"
    LEAN_ARGS=()
    if [[ "${LEAN_BATCH:-0}" == "1" ]]; then
      LEAN_ARGS+=(--lean)
    fi
    .venv/bin/python -m vqa_pipeline.tracked_pipeline \
      --segmenter sam2 \
      --video "$VIDEO_PATH" \
      --output-dir "$OUTPUT_ROOT" \
      --stop-after sam \
      "${LEAN_ARGS[@]}"
    echo "[SAM complete] GPU process exited. Next: $0 naming $VIDEO_INPUT"
    ;;
  naming)
    [[ -f "$OUTPUT_ROOT/01_sam2/tracks.json" ]] || {
      echo "Missing SAM output. Run: $0 sam $VIDEO_INPUT" >&2
      exit 1
    }
    [[ -f "$OUTPUT_ROOT/02_crops/crops.json" ]] || {
      echo "Missing crop output. Run: $0 sam $VIDEO_INPUT" >&2
      exit 1
    }
    echo "[Naming prepare] $OUTPUT_ROOT"
    .venv/bin/python -m vqa_pipeline.naming_experiment prepare \
      --crops-json "$OUTPUT_ROOT/02_crops/crops.json" \
      --tracks-json "$OUTPUT_ROOT/01_sam2/tracks.json" \
      --experiment-dir "$NAMING_ROOT/input"
    echo "[Qwen Naming] $NAMING_MODEL"
    .venv-qwen/bin/python workers/local_vlm_name.py \
      --assets-json "$NAMING_ROOT/input/assets.json" \
      --output-dir "$NAMING_ROOT" \
      --model "$NAMING_MODEL"
    echo "[Naming complete] GPU process exited. Next: $0 vqa $VIDEO_INPUT"
    ;;
  vqa)
    [[ -f "$NAMING_ROOT/naming.json" ]] || {
      echo "Missing detailed Naming output. Run: $0 naming $VIDEO_INPUT" >&2
      exit 1
    }
    echo "[Annotation + Spatial Relations + OpenAI VQA] $OUTPUT_ROOT"
    .venv/bin/python -m vqa_pipeline.finalize_vqa \
      --output-root "$OUTPUT_ROOT" \
      --source-naming "$NAMING_ROOT/naming.json" \
      --model "$OPENAI_MODEL" \
      --confidence-threshold "$CONFIDENCE_THRESHOLD"
    echo "[VQA complete] Review: $OUTPUT_ROOT/final_review.html"
    ;;
  *)
    usage
    ;;
esac
