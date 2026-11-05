#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

DATASET_ROOT="raw_datasets/Safe_and_Unsafe_Behaviours"
DEFAULT_DATA_DIR="$DATASET_ROOT/data"
TARGET="${1:-$DEFAULT_DATA_DIR}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs}"
FINAL_JSON_DIR="${FINAL_JSON_DIR:-vqa_datasets/Safe_and_Unsafe_Behaviours}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat >&2 <<'EOF'
Usage:
  ./run_safe_and_unsafe_behaviours.sh 0_te1
  ./run_safe_and_unsafe_behaviours.sh ./raw_datasets/Safe_and_Unsafe_Behaviours/data/0_te1.mp4
  ./run_safe_and_unsafe_behaviours.sh ./raw_datasets/Safe_and_Unsafe_Behaviours

Environment:
  FORCE=1          Re-run completed stages instead of resuming.
  OUTPUT_BASE=...  Output parent directory; default: outputs.
  FINAL_JSON_DIR=... Final VQA JSON directory.
  DRY_RUN=1        Print phase work without running models or APIs.

Directory mode intentionally selects one test representative per behavior class:
  0_te1.mp4, 1_te1.mp4, ..., 7_te1.mp4
EOF
  exit 2
}

if [[ "$TARGET" == "-h" || "$TARGET" == "--help" ]]; then
  usage
fi

resolve_single_video() {
  local value="$1"
  if [[ -f "$value" ]]; then
    realpath "$value"
    return
  fi
  local name="${value%.mp4}"
  local candidate="$DEFAULT_DATA_DIR/$name.mp4"
  if [[ -f "$candidate" ]]; then
    realpath "$candidate"
    return
  fi
  return 1
}

VIDEO_LIST="$(mktemp -t safe_unsafe_videos.XXXXXX)"
JOBS_TSV="$(mktemp -t safe_unsafe_naming_jobs.XXXXXX)"
trap 'rm -f "$VIDEO_LIST" "$JOBS_TSV"' EXIT

if [[ -d "$TARGET" ]]; then
  if [[ -d "$TARGET/data" ]]; then
    TARGET_DATA_DIR="$TARGET/data"
  else
    TARGET_DATA_DIR="$TARGET"
  fi
  for CLASS_ID in 0 1 2 3 4 5 6 7; do
    REPRESENTATIVE="$TARGET_DATA_DIR/${CLASS_ID}_te1.mp4"
    if [[ ! -f "$REPRESENTATIVE" ]]; then
      echo "Missing class representative: $REPRESENTATIVE" >&2
      exit 1
    fi
    realpath "$REPRESENTATIVE" >> "$VIDEO_LIST"
  done
else
  resolve_single_video "$TARGET" > "$VIDEO_LIST" || {
    echo "Video or directory not found: $TARGET" >&2
    usage
  }
fi

VIDEO_COUNT="$(wc -l < "$VIDEO_LIST" | tr -d ' ')"
if [[ "$VIDEO_COUNT" -eq 0 ]]; then
  echo "No mp4 videos found under: $TARGET" >&2
  exit 1
fi

echo "Safe_and_Unsafe_Behaviours automation"
echo "Videos: $VIDEO_COUNT class representatives | Output: $FINAL_JSON_DIR | JSON-only phase batch"
echo "Selection: one test video per class (0_te1 ... 7_te1)"
mkdir -p "$FINAL_JSON_DIR"

echo
echo "========== PHASE 1/3 · SAM2 ALL VIDEOS =========="
CURRENT=0
while IFS= read -r VIDEO_PATH; do
  CURRENT=$((CURRENT + 1))
  VIDEO_PATH="/${VIDEO_PATH#/}"
  VIDEO_STEM="$(basename "${VIDEO_PATH%.*}")"
  VIDEO_OUTPUT="$OUTPUT_BASE/${VIDEO_STEM}_sam2"
  FINAL_JSON="$FINAL_JSON_DIR/${VIDEO_STEM}.json"
  if [[ "$FORCE" == "1" && "$DRY_RUN" != "1" ]]; then
    rm -rf -- "$VIDEO_OUTPUT"
    rm -f -- "$FINAL_JSON"
  fi
  if [[ -f "$FINAL_JSON" ]]; then
    echo "[SAM $CURRENT/$VIDEO_COUNT] skip final JSON exists · $VIDEO_STEM"
  elif [[ -f "$VIDEO_OUTPUT/01_sam2/tracks.json" && -f "$VIDEO_OUTPUT/02_crops/crops.json" ]]; then
    echo "[SAM $CURRENT/$VIDEO_COUNT] resume output exists · $VIDEO_STEM"
  elif [[ "$DRY_RUN" == "1" ]]; then
    echo "[SAM $CURRENT/$VIDEO_COUNT] would process · $VIDEO_STEM"
  else
    echo "[SAM $CURRENT/$VIDEO_COUNT] processing · $VIDEO_STEM"
    .venv/bin/python -m vqa_pipeline.tracked_pipeline \
      --segmenter sam2 \
      --video "$VIDEO_PATH" \
      --output-dir "$VIDEO_OUTPUT" \
      --stop-after sam \
      --lean < /dev/null
    [[ -f "$VIDEO_OUTPUT/01_sam2/tracks.json" && -f "$VIDEO_OUTPUT/02_crops/crops.json" ]] || {
      echo "SAM stage did not produce required JSON: $VIDEO_STEM" >&2
      exit 1
    }
  fi
done < "$VIDEO_LIST"

echo
echo "========== PHASE 2/3 · QWEN NAMING ALL VIDEOS =========="
: > "$JOBS_TSV"
CURRENT=0
while IFS= read -r VIDEO_PATH; do
  CURRENT=$((CURRENT + 1))
  VIDEO_PATH="/${VIDEO_PATH#/}"
  VIDEO_STEM="$(basename "${VIDEO_PATH%.*}")"
  VIDEO_OUTPUT="$OUTPUT_BASE/${VIDEO_STEM}_sam2"
  NAMING_ROOT="$VIDEO_OUTPUT/naming_runs/qwen3_5_9b_detailed"
  FINAL_JSON="$FINAL_JSON_DIR/${VIDEO_STEM}.json"
  if [[ -f "$FINAL_JSON" ]]; then
    echo "[Naming prepare $CURRENT/$VIDEO_COUNT] skip final JSON exists · $VIDEO_STEM"
  elif [[ -f "$NAMING_ROOT/naming.json" ]]; then
    echo "[Naming prepare $CURRENT/$VIDEO_COUNT] resume output exists · $VIDEO_STEM"
  elif [[ "$DRY_RUN" == "1" ]]; then
    echo "[Naming prepare $CURRENT/$VIDEO_COUNT] would prepare · $VIDEO_STEM"
  else
    [[ -f "$VIDEO_OUTPUT/02_crops/crops.json" ]] || {
      echo "Missing SAM crop output: $VIDEO_OUTPUT/02_crops/crops.json" >&2
      exit 1
    }
    echo "[Naming prepare $CURRENT/$VIDEO_COUNT] $VIDEO_STEM"
    .venv/bin/python -m vqa_pipeline.naming_experiment prepare \
      --crops-json "$VIDEO_OUTPUT/02_crops/crops.json" \
      --tracks-json "$VIDEO_OUTPUT/01_sam2/tracks.json" \
      --experiment-dir "$NAMING_ROOT/input" < /dev/null
    printf '%s\t%s\t%s\n' "$VIDEO_STEM" "$NAMING_ROOT/input/assets.json" "$NAMING_ROOT" >> "$JOBS_TSV"
  fi
done < "$VIDEO_LIST"

JOB_COUNT="$(wc -l < "$JOBS_TSV" | tr -d ' ')"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[Qwen] dry-run; model would be loaded once for pending videos"
elif [[ "$JOB_COUNT" -gt 0 ]]; then
  echo "[Qwen] loading one model for $JOB_COUNT videos"
  .venv-qwen/bin/python workers/local_vlm_name_batch.py \
    --jobs-tsv "$JOBS_TSV" \
    --model "${NAMING_MODEL:-mlx-community/Qwen3.5-9B-MLX-4bit}" \
    --cleanup-images < /dev/null
else
  echo "[Qwen] nothing to do"
fi

echo
echo "========== PHASE 3/3 · OPENAI VQA JSON ALL VIDEOS =========="
CURRENT=0
while IFS= read -r VIDEO_PATH; do
  CURRENT=$((CURRENT + 1))
  VIDEO_PATH="/${VIDEO_PATH#/}"
  VIDEO_STEM="$(basename "${VIDEO_PATH%.*}")"
  VIDEO_OUTPUT="$OUTPUT_BASE/${VIDEO_STEM}_sam2"
  NAMING_JSON="$VIDEO_OUTPUT/naming_runs/qwen3_5_9b_detailed/naming.json"
  FINAL_JSON="$FINAL_JSON_DIR/${VIDEO_STEM}.json"
  if [[ -f "$FINAL_JSON" ]]; then
    echo "[VQA $CURRENT/$VIDEO_COUNT] skip final JSON exists · $VIDEO_STEM"
  elif [[ "$DRY_RUN" == "1" ]]; then
    echo "[VQA $CURRENT/$VIDEO_COUNT] would call API · $VIDEO_STEM"
  elif [[ -f "$VIDEO_OUTPUT/06_vqa/generated_vqa.json" && -f "$VIDEO_OUTPUT/final_review.html" && "$FORCE" != "1" ]]; then
    cp "$VIDEO_OUTPUT/06_vqa/generated_vqa.json" "$FINAL_JSON"
    echo "[VQA $CURRENT/$VIDEO_COUNT] exported existing verified run · $FINAL_JSON"
  else
    [[ -f "$NAMING_JSON" ]] || {
      echo "Missing Naming JSON: $NAMING_JSON" >&2
      exit 1
    }
    echo "[VQA $CURRENT/$VIDEO_COUNT] API generation · $VIDEO_STEM"
    .venv/bin/python -m vqa_pipeline.finalize_vqa \
      --output-root "$VIDEO_OUTPUT" \
      --source-naming "$NAMING_JSON" \
      --model "${OPENAI_MODEL:-gpt-6.1-terra}" \
      --confidence-threshold "${CONFIDENCE_THRESHOLD:-0.7}" \
      --json-only \
      --export-json "$FINAL_JSON" \
      --cleanup-intermediate < /dev/null
    echo "[VQA $CURRENT/$VIDEO_COUNT] saved · $FINAL_JSON"
  fi
done < "$VIDEO_LIST"

echo
echo "Completed JSON files: $FINAL_JSON_DIR"
