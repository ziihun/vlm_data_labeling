# Industrial CCTV VQA pilot

This project creates a small, human-reviewable VQA pilot from five videos in
`Voxel51/Safe_and_Unsafe_Behaviours`.

## Current pilot scope

- Included: four Scene Understanding categories and three Query-Scene Alignment categories
- Excluded: all Risk Reasoning categories
- Output: exactly seven VQA items per successfully processed video
- Model evidence: eight uniformly sampled frames with timestamps
- Human review: complete MP4 clips plus an HTML page that seeks to evidence timestamps
- Ground truth: stored beside generated output, but hidden from the model to prevent label leakage
- Review: every run writes `human_review.csv`

This is a pipeline smoke test, not the final SAM/DAM annotation implementation. The pilot uses
an OpenAI vision model to create sampled-frame pseudo annotations. It does not claim masks or
true track IDs. The annotation provider can later be replaced with SAM 2 → DAM → naming → SAM 3.

## 현재 실행 프로세스와 사용 모델

현재 파일럿은 아래 흐름으로 실행된다.

```text
FiftyOne samples.json
  → 서로 다른 GT class의 영상 5개 선택
  → ffprobe로 영상 길이 확인
  → ffmpeg로 영상당 8개 프레임 균등 추출
  → OpenAI Responses API에 8개 프레임과 timestamp 전달
  → gpt-6.1-terra가 pseudo annotation과 7개 VQA를 한 번에 생성
  → Pydantic schema 및 timestamp/frame grounding 검증
  → JSON + JSONL + CSV + 전체 MP4 영상 검수 페이지 생성
  → Human Verification
```

| 단계 | 현재 구현 | 모델 / 도구 | 산출물 |
| --- | --- | --- | --- |
| 영상 선택 | 8개 GT class 중 서로 다른 class를 우선하여 5개 선택 | `DatasetCatalog`, FiftyOne `samples.json` | 선택 영상 목록 |
| 영상 전처리 | 영상 길이 확인 후 시작부터 끝까지 8개 프레임 균등 추출 | `ffprobe`, `ffmpeg` | Timestamp가 포함된 JPEG 8장 |
| 공통 Annotation | 샘플 프레임에서 객체·속성·시간 이벤트·작업 상태를 구조화 | **OpenAI `gpt-6.1-terra`** | Sampled-frame pseudo annotation |
| VQA 생성 | Annotation과 시각 근거를 이용해 7개 카테고리 VQA 생성 | **OpenAI `gpt-6.1-terra`**, Responses API, Structured Outputs | 영상당 VQA 7개 |
| 자동 검증 | 카테고리 누락, 잘못된 frame index 및 timestamp 범위 검증 | Pydantic + 로컬 검증 코드 | 검증된 JSON |
| 사람 검수 | 전체 원본 MP4를 재생하고 증거 timestamp로 이동 | HTML5 Video, `review.html` | 승인·수정 대상 확인 |

OpenAI 모델에는 MP4 파일 자체를 보내지 않는다. 영상당 8개의 샘플 프레임을 시간순으로
보내며, 전체 움직임 확인은 `review_videos/`의 MP4와 `review.html`에서 사람이 수행한다.
원본 데이터셋 GT class는 결과 파일에는 보존하지만 모델 입력에서는 숨겨 label leakage를
방지한다.

기본 모델은 `gpt-6.1-terra`이며 다음 순서로 결정된다.

1. 실행 인자 `--model`
2. `.env`의 `OPENAI_MODEL`
3. 코드 기본값 `gpt-6.1-terra`

현재 성공한 파일럿 결과는 영상 5개, 영상당 7개, 총 35개 VQA이다. 각 영상에는 아래
카테고리가 정확히 한 번씩 포함된다.

| 상위 범주 | 하위 카테고리 |
| --- | --- |
| Scene Understanding | `scene.object_attribute_recognition` |
| Scene Understanding | `scene.action_spatial_understanding` |
| Scene Understanding | `scene.temporal_understanding` |
| Scene Understanding | `scene.task_state_completion_understanding` |
| Query-Scene Alignment | `alignment.object_attribute_alignment` |
| Query-Scene Alignment | `alignment.action_relation_alignment` |
| Query-Scene Alignment | `alignment.temporal_alignment` |

Risk Reasoning은 이번 파일럿에서 생성하지 않는다.

## 최종 목표 프로세스와의 차이

노션에 정의한 최종 파이프라인은 다음 구조다.

```text
Raw Video
  → SAM 2: 객체 Mask 생성 및 Tracking
  → DAM-3B-Video: 객체별 시각 특징 설명
  → OpenAI API: 객체 이름·속성 표준화
  → SAM 3: 이름 기반 재탐지 및 Mask 검증
  → Post-processing: Object + Attribute + Track ID + Timestamp + Mask
  → LLM: Scene Understanding + Query-Scene Alignment VQA 생성
  → Human Verification
```

아래 모델은 **현재 파일럿에는 아직 연결되지 않았다**.

- `SAM 2`: 객체 분할 및 실제 tracking
- `DAM-3B-Video`: 객체 단위 video description
- `SAM 3`: 객체 이름 기반 재탐지와 mask 검증
- `YOLO + ByteTrack`: ROI 이탈 및 작업 종료 후보 탐지

현재 `gpt-6.1-terra` 기반 pseudo annotation 단계는 전체 VQA 생성·저장·검수 흐름이
정상 동작하는지 먼저 확인하기 위한 대체 구현이다.

## 질문 단위 Stage Trace MVP

대량 파일럿은 5개 영상의 VQA 생성 여부를 확인하는 경로이고, Stage Trace MVP는 질문
하나가 어떤 중간 결과를 거쳐 구성되는지 확인하는 별도 경로다.

현재 실행된 예시는 다음 질문이다.

```text
Video: 0_te1
Category: scene.object_attribute_recognition
Question: 화면 중앙 설비의 덮개는 무슨 색인가요?
Answer: 주황색입니다.
```

```text
00 기존 VQA 질문 선택
  → 01 질문의 evidence frame 선택
  → 02 gpt-6.1-terra가 질문 관련 ROI box 제안
  → 03 SAM 2.1 Hiera Tiny가 실제 mask·overlay·crop 생성
  → 04 DAM-3B-Video 단계
       ├─ 로컬 실행: 미지원 하드웨어로 차단
       └─ fallback: gpt-6.1-terra가 ROI 상세 설명
  → 05 gpt-6.1-terra가 객체명·속성·상태 표준화
  → 06 SAM 3.1 검증 단계: CUDA/체크포인트 부재로 차단
  → 07 단계 산출물을 사용해 최종 MVP VQA 재구성
```

| 단계 | 요청 모델 | 실제 실행 모델 | 현재 상태 |
| --- | --- | --- | --- |
| 질문 관련 영역 제안 | `gpt-6.1-terra` | `gpt-6.1-terra` | 실행 |
| 객체 mask 및 crop | `facebook/sam2.1-hiera-tiny` | `facebook/sam2.1-hiera-tiny` | 실행 |
| ROI 상세 설명 | `nvidia/DAM-3B-Video` | `gpt-6.1-terra` | 명시적 fallback |
| 객체 정보 표준화 | `gpt-6.1-terra` | `gpt-6.1-terra` | 실행 |
| 이름 기반 재검증 | `facebook/sam3.1` | 없음 | 차단 |
| 최종 VQA 조립 | `gpt-6.1-terra` | `gpt-6.1-terra` | 실행 |

`DAM-3B-Video` 체크포인트는 약 7.12GB다. 공식 데모 코드는 CUDA가 없으면 CPU를
선택하므로 Apple Silicon에서 논리적으로 불가능한 것은 아니지만, 현재 16GB 환경에는
모델·영상 activation까지 포함한 실용적인 실행 여유가 부족하고 DAM 환경도 설치되어 있지
않다. 따라서 fallback 결과를 DAM 출력으로 표기하지 않고
`fallback_is_not_dam_output: true`로 기록한다.

`SAM 3.1` 공식 구현은 CUDA 12.6 GPU와 승인된 체크포인트 접근이 필요하므로 이 Mac에서는
실행하지 않는다. 해당 단계는 `status: blocked`, `executed_model: null`로 기록한다.

### Stage Trace 실행

SAM 2 전용 Python 3.12 환경을 준비한다.

```bash
python3.12 -m venv .venv-sam2
.venv-sam2/bin/pip install -r requirements-sam2.txt
```

질문 하나의 trace를 생성한다.

```bash
.venv/bin/python -m vqa_pipeline.trace \
  --video-id 0_te1 \
  --category scene.object_attribute_recognition
```

### Stage Trace 산출물

```text
vqa_datasets/mvp_trace/0_te1_object/
├── 01_evidence/
│   └── evidence_frame.jpg
├── 02_region_proposal/
│   ├── bbox_overlay.jpg
│   └── result.json
├── 03_sam2/
│   ├── mask.png
│   ├── overlay.png
│   ├── roi_crop.png
│   ├── roi_crop_context.jpg
│   └── result.json
├── 04_dam/result.json
├── 05_normalization/result.json
├── 06_sam3/result.json
├── 07_final_vqa/result.json
├── source_video.mp4
├── trace.json
└── report.html
```

`report.html`에서 원본 영상, 질문, bbox, SAM 2 mask, crop, 설명, 표준화 결과, 검증 상태,
최종 VQA를 순서대로 확인할 수 있다. 초록은 실제 실행, 주황은 fallback, 빨강은 차단된
단계를 의미한다.

## Discovery-first SAM2 MVP

`vqa_pipeline.trace`는 이미 존재하는 질문 하나를 디버깅하기 위한 **query-conditioned 경로**다.
따라서 질문 관련 ROI 하나만 SAM2에 box prompt로 전달한다. 전체 VQA 데이터 생성의 기본
구조는 이 경로가 아니라 아래 `vqa_pipeline.discovery`다.

```text
원본 영상
  → 8개 timestamp 프레임
  → SAM2 automatic mask generation (질문/box prompt 없음)
  → SAM2가 반환한 모든 유지 mask에 region ID 부여
  → 모든 mask bbox + overlay + crop 저장
  → 각 region ID의 객체명·속성·상태 설명
  → 프레임 간 중복 region을 표준 객체로 통합
  → 전체 구조화 annotation에서 VQA 7개 생성
```

SAM2는 사람·설비 같은 의미 이름을 직접 인식하는 detector가 아니라, 이름 없는 mask
proposal을 생성하는 segmentation 모델이다. 의미 이름은 다음 설명 단계가 부여한다.

현재 `0_te1` 실험은 다음과 같이 실제 완료됐다.

| 항목 | 결과 |
| --- | ---: |
| 샘플 프레임 | 8 |
| SAM2가 threshold와 내장 NMS 후 반환한 mask | 177 |
| 추가 cap/custom filter 후 유지한 mask | 177 |
| 프레임별 mask | 18~25개 범위 |
| LLM 설명 ID coverage | 177/177 |
| 프레임 간 표준 객체 | 30 |
| 최종 VQA | 7 |

`raw_proposal_count`는 threshold 적용 전의 모든 점별 후보 수가 아니라, Transformers의
SAM2 automatic mask generator가 score·stability threshold와 내장 crop NMS를 적용한 뒤
반환한 수다. 현재 코드는 이 반환값에 별도 개수 제한이나 필터를 적용하지 않으므로
`raw_proposal_count == retained_proposal_count`다. 기본 임계값은 다음과 같다.

- `points_per_crop=32`
- `pred_iou_thresh=0.82`
- `stability_score_thresh=0.90`
- Transformers pipeline 내장 `crops_nms_thresh=0.7`

실행 명령:

```bash
.venv/bin/python -m vqa_pipeline.discovery
```

전체 결과는 `vqa_datasets/discovery_mvp/0_te1/report.html`에서 확인한다. 각 프레임마다
모든 bbox/ID 이미지, 모든 mask overlay, 177개 개별 crop과 객체 설명, 표준 객체가 참조한
SAM region ID, 최종 VQA를 한 페이지에서 볼 수 있다.

```text
vqa_datasets/discovery_mvp/0_te1/
├── 00_input/manifest.json
├── 01_sam2/
│   ├── result.json
│   └── frame_00 ... frame_07/
│       ├── all_bboxes.jpg
│       ├── all_masks_overlay.png
│       ├── all_crops_contact_sheet.jpg
│       ├── masks/
│       ├── crops/
│       └── result.json
├── 02_region_descriptions/
├── 03_normalization/result.json
├── 04_sam3/result.json
├── 05_vqa/result.json
├── manifest.json
├── source_video.mp4
└── report.html
```

DAM-3B-Video는 현재 Apple Silicon 환경에서 실행하지 못하므로 region 설명은
`gpt-6.1-terra` fallback이다. 결과에는 `fallback_is_not_dam_output: true`가 기록된다.
SAM3도 실행하지 않았으며 `blocked` 상태로 남는다. 프레임 간 객체 통합은 진짜 tracking이
아니라 `llm_visual_association_not_true_tracking`으로 기록한다.

## Track-grounded automation candidate

`vqa_pipeline.tracked_pipeline`은 앞으로 자동화에 사용할 새 경로다. 위의 8장
Discovery MVP와 달리 SAM2 Tracking에 8장 표본을 사용하지 않는다.

```text
원본 영상
  → 원본 24.83 FPS를 유지한 10초 Clip / 5초 Stride
  → Clip별 별도 5 FPS tracking stream
  → 첫 tracking frame에서 SAM2.1 automatic mask discovery
  → 모든 5 FPS frame으로 Track ID와 mask 전파
  → 겹치는 Clip의 동일 Track을 동일 source timestamp mask IoU로 병합
  → Global Track별 대표 masked frame 최대 8장
  → DAM-3B-Video description
  → OpenAI 객체 이름·속성·직접 관찰 변화 구조화
  → SAM3 이름 기반 재탐지 mask와 SAM2 mask IoU 비교
  → 검증 annotation에서만 VQA 생성
```

VQA의 timestamp와 tracking frame index는 정답 대상이 아니라 Evidence metadata다.
질문이나 답변에 `6.399초`, `Frame 7`, `G0001` 같은 값을 넣으면 Pydantic 검증에서
실패한다. 시간·행동·완료·복구 카테고리는 `directly_observed=true`인 before/after change
ID와 서로 다른 두 evidence timestamp가 없으면 생성하지 않고 `skipped_categories`에
기록한다. 따라서 카테고리 7개를 무조건 채우지 않는다.

현재 `0_te1` 실측 결과:

| 항목 | 결과 |
| --- | ---: |
| Source FPS | 24.83 |
| Clip | `[0,10]`, `[5,14.982]` |
| SAM2 Tracking 입력 | Clip당 5 FPS × 50장 |
| DAM 대표 입력 | Global Track당 최대 8장 |
| Local Track | 13 |
| Overlap 병합 후 Global Track | 10 |
| 직접 확인된 before/after change | 0 |
| 생성 VQA | 0 |
| 근거 부족/검증 미완료로 skip | 7 |

이번 실행은 `DAM_BASE_URL`이 없어 `gpt-6.1-terra` fallback을 사용했고,
`SAM3_BASE_URL`이 없어 모든 객체가 `sam3_blocked` 검토 대상으로 기록됐다. 이 상태에서
질문을 억지 생성하지 않은 것이 의도한 통과 조건이다. 실제 자동화 전에는 DAM과 SAM3
서버를 연결하고 아래 strict 실행이 성공해야 한다.

```bash
# 구조 검증 MVP: fallback/blocked 상태를 기록하며 실행
.venv/bin/python -m vqa_pipeline.tracked_pipeline

# 실제 자동화 gate: DAM/SAM3 서버가 없으면 시작 전에 실패
.venv/bin/python -m vqa_pipeline.tracked_pipeline \
  --strict-models \
  --strict-verification
```

새 결과 위치:

```text
vqa_datasets/tracked_mvp/0_te1/
├── clips/                         # 원본 FPS Clip + 별도 5 FPS tracking frames
├── sam2/clip_000, clip_001/
│   ├── tracking.json              # Track ID, frame mask, source timestamp
│   ├── tracks/*/masks/
│   ├── tracks/*/dam_frames/       # Track별 최대 8장
│   └── tracking_overlay_5fps.mp4
├── merged_tracks.json             # overlap IoU Track 병합
├── dam/                           # Track description
├── annotation_before_sam3.json
├── annotation.json                # SAM3 상태 포함
├── vqa.json                       # generated + skipped categories
├── manifest.json
└── report.html
```

## Class architecture

- `DatasetCatalog`: FiftyOne metadata loading and label-diverse selection
- `VideoProcessor`: evidence-frame extraction and full MP4 review copies
- `OpenAIVQAAnalyzer`: OpenAI Responses API calls and grounding validation
- `ArtifactWriter`: JSON/JSONL/CSV and video-first HTML review output
- `VQAPilotPipeline`: orchestration, status, failure handling, and summaries
- `Sam2AutomaticMaskWorker`: 모든 자동 mask proposal, bbox, overlay, crop 생성
- `Sam2DiscoveryRunner`: 별도 SAM2 Python 3.12 환경 실행과 결과 회수
- `OpenAIDiscoveryAnalyzer`: 모든 region 설명, 객체 표준화, annotation 기반 VQA 생성
- `DiscoveryArtifactWriter`: 전체 region contact sheet와 discovery HTML report 생성
- `DiscoveryFirstPipeline`: SAM2 discovery부터 최종 VQA까지 단계별 orchestration
- `OverlappingClipPreprocessor`: source FPS Clip과 별도 5 FPS Tracking stream 생성
- `Sam2ClipTracker`: 초기 automatic masks를 모든 5 FPS frame으로 실제 전파
- `OverlapTrackMerger`: 겹친 Clip의 동일 Track을 timestamp mask IoU로 병합
- `DAMTrackDescriber`: DAM OpenAI-compatible server와 명시적 Terra fallback
- `Sam3NameVerifier`: 이름 기반 재탐지 server 결과와 SAM2 mask IoU 비교
- `OpenAITrackedAnalyzer`: Track 구조화, 변화 근거 검증, 보수적 VQA/skip 생성
- `TrackedVQAPipeline`: 자동화 후보 전체 orchestration과 strict gate

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Add OPENAI_API_KEY to .env
```

`ffmpeg` and `ffprobe` must be available on `PATH`.

## Run

Prepare frames and full review videos without an API call:

```bash
.venv/bin/python -m vqa_pipeline.pipeline --prepare-only
```

Run the five-video pilot:

```bash
.venv/bin/python -m vqa_pipeline.pipeline
```

Use a different model or output directory:

```bash
.venv/bin/python -m vqa_pipeline.pipeline \
  --model gpt-6.1-terra \
  --output-dir vqa_datasets/pilot_5
```

## Output

```text
vqa_datasets/pilot_5/
├── annotations/       # one structured result per video
├── frames/            # timestamped evidence frames
├── review_videos/     # complete MP4 clips for motion review
├── human_review.csv   # review queue
├── manifest.json      # inputs, configuration, errors, limitations
├── review.html        # embedded videos, VQA, and seek-to-evidence buttons
├── summary.json       # coverage counts
└── vqa.jsonl          # flattened VQA records
```

## Test

```bash
.venv/bin/pytest -q
```
