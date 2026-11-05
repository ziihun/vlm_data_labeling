# ForgeVLM

### Understanding Manufacturing Workplaces with Vision–Language Models

ForgeVLM is a research pipeline for building grounded video question-answering (VQA)
datasets from manufacturing and industrial-safety footage. It combines object segmentation
and tracking, local vision-language model (VLM) naming, human ground truth, structured VQA
generation, and browser-based human verification.

> Research prototype. Generated annotations and answers remain drafts until human review.
> Risk reasoning is intentionally outside the current scope.

## Overview

```text
Video or ordered frames
        │
        ▼
SAM 2.1 discovery + tracking
        │  Track ID, mask, BBox, timestamp
        ▼
Qwen3.5-9B object naming
        │  name, visible attributes, state, confidence
        ▼
Human GT fusion + spatial/temporal grounding
        │
        ▼
OpenAI VQA generation
        │
        ▼
Human verification HTML
```

The local GPU stages run sequentially. SAM and Qwen are never required to remain loaded at
the same time.

## VQA Tasks

| Group | Capability | Examples |
|---|---|---|
| Scene understanding | Object and attribute recognition | object presence, count, PPE, visible state |
| Scene understanding | Action and spatial understanding | action, interaction, left/right/above/below relations |
| Scene understanding | Temporal understanding | observed order and before/after change |
| Scene understanding | Task state and completion | task phase, completion, restoration |
| Query–scene alignment | Object/attribute alignment | grounded positive and hard-negative queries |
| Query–scene alignment | Action/relation alignment | action and spatial consistency |
| Query–scene alignment | Temporal alignment | consistency with an observed transition |

Questions and answers are generated in English. Frame indices, timestamps, and Track IDs are
stored only as evidence metadata. Temporal questions require an observed transition; negative
questions are never inferred from detector failure alone.

## Models

| Stage | Default model | Runtime |
|---|---|---|
| Segmentation and tracking | `facebook/sam2.1-hiera-tiny` | PyTorch / Transformers |
| Object naming | `mlx-community/Qwen3.5-9B-MLX-4bit` | MLX on Apple Silicon |
| VQA generation | `gpt-6.1-terra` | OpenAI Responses API |

Optional naming baselines include RAM++, SigLIP2, YOLOE Prompt-Free, InternVL3.5, Molmo2,
MiniCPM-V, and OpenAI vision models.

## Installation

Python 3.12 is recommended. Separate environments prevent model dependency conflicts and keep
GPU memory release explicit between stages.

```bash
# Orchestration, OpenAI API, reports, and tests
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# SAM 2.1 / segmentation
python3.12 -m venv .venv-sam2
.venv-sam2/bin/pip install -r requirements-segmentation.txt

# Local Qwen naming on Apple Silicon
python3.12 -m venv .venv-qwen
.venv-qwen/bin/pip install -r requirements-qwen.txt

cp .env.example .env
```

Configure local secrets in `.env`:

```dotenv
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-6.1-terra
HF_TOKEN=...
QWEN_MODEL=mlx-community/Qwen3.5-9B-MLX-4bit
```

`.env`, datasets, checkpoints, logs, and generated artifacts are excluded from Git.

## Data Layout

Data is not distributed with this repository.

```text
raw_datasets/
├── SteelBench/
│   ├── data/
│   │   ├── frames/<clip_id>/frame_0.jpg ... frame_7.jpg
│   │   └── annotations/<clip_id>.json
│   └── manifests/batch_manifest.csv
└── Safe_and_Unsafe_Behaviours/
    └── data/*.mp4
```

## Recommended SteelBench Run

The recommended experiment samples up to five clips from every scene category. A category is
the clip-name prefix before `_YYYYMMDD_HHMMSS_NNNN`. The selection uses a fixed seed, prefers
already completed SAM outputs, and is saved once for reproducibility.

```bash
mkdir -p logs
caffeinate -i ./run_steelbench_category5_vqa.sh \
  ./raw_datasets/SteelBench \
  2>&1 | tee logs/steelbench_category5.log
```

The script performs:

1. fixed random sampling per scene category;
2. SAM 2.1 only for selected clips without reusable outputs;
3. one Qwen naming call per clip for all SAM Tracks;
4. VQA generation from ordered frames, named Tracks, and human GT;
5. generation of a human-review HTML report.

Configuration is exposed through environment variables:

```bash
PER_CATEGORY=5 \
SELECTION_SEED=20261106 \
QWEN_MODEL=mlx-community/Qwen3.5-9B-MLX-4bit \
OPENAI_MODEL=gpt-6.1-terra \
./run_steelbench_category5_vqa.sh ./raw_datasets/SteelBench
```

The saved selection is reused on restart. Delete or change `SELECTION_FILE` only when a new
sample is intentionally required.

## Entry Points

| Command | Purpose |
|---|---|
| `run_steelbench_category5_vqa.sh` | Recommended stratified SteelBench experiment: sample → SAM → Qwen → GT → VQA → review |
| `run_steelbench_completed_sam_vqa.sh` | Run Qwen naming and VQA only for clips with completed SAM outputs |
| `run_steelbench_vqa.sh` | Full-frame baseline using SAM outputs and GT directly with OpenAI |
| `run_safe_and_unsafe_behaviours.sh` | Process one representative test video per behavior class |
| `run_vqa_stage.sh sam <video>` | Run preprocessing, SAM tracking, and crop generation for one video |
| `run_vqa_stage.sh naming <video>` | Name the saved Tracks with the local Qwen model |
| `run_vqa_stage.sh vqa <video>` | Build annotations, VQA JSON, and the final review report |

Single-video staged execution:

```bash
./run_vqa_stage.sh sam 0_te1
./run_vqa_stage.sh naming 0_te1
./run_vqa_stage.sh vqa 0_te1
```

## Outputs

Generated data is written outside the tracked source tree:

```text
outputs/
├── SteelBench_sam2/<clip_id>/
│   ├── tracks.json
│   └── tracking_contact.jpg
├── SteelBench_category5_selection.json
└── SteelBench_category5_qwen_naming/<clip_id>/
    ├── naming.json
    ├── raw_response.json
    └── metrics.json

vqa_datasets/SteelBench_category5/
├── <clip_id>.json
└── human_review.html
```

Every final VQA item remains marked as a draft until a reviewer approves or modifies it in the
HTML interface. The report presents source frames, SAM BBoxes, evidence, and a
rejected-question list; the complete naming records remain beside the report under `outputs/`.

## Grounding Policy

- SAM outputs are region proposals, not semantic truth.
- Local VLM names are candidates and retain confidence plus uncertainty.
- Human GT has priority when it conflicts with an automatic name.
- Stable spatial relations require co-visible objects and consistent BBoxes.
- Sparse ordered frames do not prove an unobserved event.
- Missing detections never provide sufficient evidence for a negative question.
- Risk or intent is not inferred from appearance alone.

## Repository Layout

```text
.
├── run_*.sh                 # Reproducible experiment entry points
├── scripts/                 # Dataset selection, VQA generation, and reports
├── workers/                 # SAM and local VLM inference workers
├── vqa_pipeline/            # Core preprocessing, annotation, and mapping logic
├── tests/                   # Pipeline invariants and schema tests
├── requirements*.txt        # Isolated runtime environments
├── .env.example             # Secret-free configuration template
└── README.md
```

## Verification

```bash
.venv/bin/python -m pytest -q
bash -n run_steelbench_category5_vqa.sh
```

The tests cover clip windows, Track merging, crop limits, naming gates, spatial relations,
evidence metadata, English VQA output, negative grounding, and duplicate prevention.

## Reproducibility Notes

- Sampling seed and selected Clip IDs are stored in the selection JSON.
- Completed stage outputs are detected and reused after interruption.
- Local models are loaded once per phase and released before the next model starts.
- API model names and token usage are stored with generated VQA records.
- Raw datasets and generated results must be obtained or shared separately under their original
  licenses and access terms.

## Citation

A formal paper citation will be added after release. Until then, cite this repository as
**ForgeVLM: Understanding Manufacturing Workplaces with Vision–Language Models**.
