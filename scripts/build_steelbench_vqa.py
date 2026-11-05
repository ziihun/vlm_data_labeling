from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = (
    "scene.object_attribute_recognition",
    "scene.action_spatial_understanding",
    "scene.temporal_understanding",
    "scene.task_state_completion_understanding",
    "alignment.object_attribute_alignment",
    "alignment.action_relation_alignment",
    "alignment.temporal_alignment",
)
METADATA_PATTERN = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:seconds?|s\b))|(?:(?:frame|clip|track)[_ #\-]?\d+)",
    re.IGNORECASE,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(StrictModel):
    frame_indices: list[int] = Field(min_length=1)
    sam_track_ids: list[str]
    annotation_fields: list[str]
    rationale: str


class VQAItem(StrictModel):
    category: Literal[
        "scene.object_attribute_recognition",
        "scene.action_spatial_understanding",
        "scene.temporal_understanding",
        "scene.task_state_completion_understanding",
        "alignment.object_attribute_alignment",
        "alignment.action_relation_alignment",
        "alignment.temporal_alignment",
    ]
    question: str
    answer: str
    answer_type: Literal["yes_no", "short_text"]
    polarity: Literal["positive", "negative", "hard_negative", "not_applicable"]
    evidence: Evidence
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_question(self) -> "VQAItem":
        if METADATA_PATTERN.search(self.question) or METADATA_PATTERN.search(self.answer):
            raise ValueError("question/answer cannot expose frame or timestamp metadata")
        if any(index < 0 or index > 7 for index in self.evidence.frame_indices):
            raise ValueError("frame index must be in 0..7")
        return self


class SkippedCategory(StrictModel):
    category: Literal[
        "scene.object_attribute_recognition",
        "scene.action_spatial_understanding",
        "scene.temporal_understanding",
        "scene.task_state_completion_understanding",
        "alignment.object_attribute_alignment",
        "alignment.action_relation_alignment",
        "alignment.temporal_alignment",
    ]
    reason: str


class VQABatch(StrictModel):
    items: list[VQAItem]
    skipped_categories: list[SkippedCategory]

    @model_validator(mode="after")
    def coverage(self) -> "VQABatch":
        generated = {item.category for item in self.items}
        skipped = {item.category for item in self.skipped_categories}
        if generated & skipped or generated | skipped != set(CATEGORIES):
            raise ValueError("every category must be generated or skipped, never both")
        questions = [item.question.strip().casefold() for item in self.items]
        if len(questions) != len(set(questions)):
            raise ValueError("duplicate questions")
        return self


PROMPT = """Generate grounded English VQA from eight ordered SteelBench frames, named SAM2 region tracks, and the supplied human annotation.
The images are ordered frame_0 through frame_7. They are sparse samples from one clip, not a continuous video.
The final image is a SAM2 BBox contact sheet for the same eight frames. Local VLM names are candidates, not semantic ground truth. Human GT has priority when they conflict.

Use only these categories:
{categories}

Rules:
1. Generate every distinct, useful question supported jointly by the frames and annotation. There is no per-category count cap.
2. Never ask about frame indices, timestamps, clip IDs, annotation fields, action codes, or dataset metadata.
3. Put supporting frame indices, SAM Track IDs, and annotation field paths only in evidence.
4. Do not invent objects, PPE, actions, spatial relations, safety states, or hidden intent.
5. Temporal questions require an actual visible before/after change or an explicit human-annotated action_transition. Sparse ordered samples alone do not prove an event.
6. Task completion/restoration questions require an explicitly observed completion or restoration. Otherwise skip that category.
7. Negative questions require explicit positive evidence in the human annotation; never use non-detection as proof.
8. Spatial questions use the image viewer's left/right and must cite frames where both entities are visible.
9. Do not generate Risk Reasoning questions.
10. Avoid paraphrase duplicates. Skip a category only when no grounded item is possible.
11. A SAM2 region may reveal an object omitted from GT, but use its VLM name only when confidence and visual evidence support it. Do not treat every mask as a real object.

Compact SAM2 tracks (frame index, BBox, mask area):
{sam_tracks}

Human annotation:
{annotation}
"""


def data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def annotation_for_prompt(annotation: dict) -> dict:
    allowed = (
        "annotation_layer", "scene_type", "num_workers", "dominant_actions",
        "overall_ppe_compliance", "visibility", "visibility_conditions",
        "visible_equipment", "persons", "scene_unsafe_act", "action_transition",
        "group_activity", "scene_description", "annotator_confidence",
        "tier", "severity_level",
    )
    return {key: annotation.get(key) for key in allowed if key in annotation}


def tracks_for_prompt(payload: dict, naming: dict | None = None) -> list[dict]:
    names = {
        item["track_id"]: item
        for item in (naming or {}).get("items", [])
        if item.get("track_id")
    }
    return [
        {
            "track_id": track["track_id"],
            "discovered_at_frame_index": track["discovered_at_frame_index"],
            "vlm_naming": names.get(track["track_id"], {
                "canonical_object_name": "unknown",
                "confidence": 0.0,
                "review_status": "needs_review",
            }),
            "observations": [
                {
                    "frame_index": item["frame_index"],
                    "bbox_xyxy": item.get("bbox_xyxy"),
                    "mask_area_pixels": item.get("mask_area_pixels", 0),
                }
                for item in track.get("frames", [])
                if item.get("bbox_xyxy")
            ],
        }
        for track in payload.get("tracks", [])
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--sam-root", type=Path, required=True)
    parser.add_argument("--naming-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--start-after", default="")
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    if args.max_clips < 0:
        parser.error("--max-clips must be >= 0")

    raw_root = args.raw_root.resolve()
    frames_root = raw_root / "data/frames"
    annotations_root = raw_root / "data/annotations"
    sam_root = args.sam_root.resolve()
    naming_root = args.naming_root.resolve() if args.naming_root else None
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = None
    if args.selection_file:
        selection = json.loads(args.selection_file.read_text(encoding="utf-8"))
        selected = set(selection.get("clip_ids", [])) if isinstance(selection, dict) else set(selection)
    jobs = []
    for frame_dir in sorted(path for path in frames_root.iterdir() if path.is_dir()):
        annotation_path = annotations_root / f"{frame_dir.name}.json"
        frames = sorted(frame_dir.glob("frame_*.jpg"))
        sam_tracks = sam_root / frame_dir.name / "tracks.json"
        sam_contact = sam_root / frame_dir.name / "tracking_contact.jpg"
        naming_path = naming_root / frame_dir.name / "naming.json" if naming_root else None
        sam_ready = sam_tracks.exists() and sam_contact.exists()
        naming_ready = naming_path is None or naming_path.exists()
        if annotation_path.exists() and len(frames) == 8 and frame_dir.name > args.start_after and (selected is None or frame_dir.name in selected) and (args.dry_run or (sam_ready and naming_ready)):
            jobs.append((frame_dir.name, frames, annotation_path, sam_tracks, sam_contact, naming_path))
    if args.max_clips:
        jobs = jobs[:args.max_clips]
    print(json.dumps({"clips": len(jobs), "output": str(output_dir), "dry_run": args.dry_run}), flush=True)
    if args.dry_run:
        for index, (clip_id, _, _, _, _, _) in enumerate(jobs, 1):
            print(f"[SteelBench {index}/{len(jobs)}] would process · {clip_id}")
        return 0

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from .env")
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-6.1-terra")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=600.0)
    for index, (clip_id, frames, annotation_path, sam_tracks_path, sam_contact, naming_path) in enumerate(jobs, 1):
        destination = output_dir / f"{clip_id}.json"
        if destination.exists():
            print(f"[SteelBench {index}/{len(jobs)}] resume · {clip_id}", flush=True)
            continue
        print(f"[SteelBench {index}/{len(jobs)}] VQA · {clip_id}", flush=True)
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        sam_payload = json.loads(sam_tracks_path.read_text(encoding="utf-8"))
        naming_payload = json.loads(naming_path.read_text(encoding="utf-8")) if naming_path else None
        sam_tracks = tracks_for_prompt(sam_payload, naming_payload)
        prompt = PROMPT.format(
            categories=json.dumps(CATEGORIES),
            sam_tracks=json.dumps(sam_tracks, ensure_ascii=False, separators=(",", ":")),
            annotation=json.dumps(annotation_for_prompt(annotation), ensure_ascii=False),
        )
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        content.extend({"type": "input_image", "image_url": data_url(frame), "detail": "high"} for frame in frames)
        content.append({"type": "input_image", "image_url": data_url(sam_contact), "detail": "high"})
        response = client.responses.parse(
            model=model,
            instructions="Return schema-valid, evidence-grounded SteelBench VQA only.",
            input=[{"role": "user", "content": content}],
            text_format=VQABatch,
            max_output_tokens=20000,
            store=False,
        )
        result = response.output_parsed
        if result is None:
            raise RuntimeError(f"No parsed output for {clip_id}")
        known_tracks = {track["track_id"] for track in sam_tracks}
        for item in result.items:
            unknown = set(item.evidence.sam_track_ids) - known_tracks
            if unknown:
                raise RuntimeError(f"Unknown SAM Track IDs for {clip_id}: {sorted(unknown)}")
        items = []
        for item_index, item in enumerate(result.items, 1):
            items.append({
                "item_id": f"VQA_{item_index:03d}",
                **item.model_dump(mode="json"),
                "draft": True,
                "requires_final_review": True,
            })
        payload = {
            "schema_version": "1.0.0-steelbench-frame-vqa",
            "video_id": clip_id,
            "source_type": "eight_ordered_frames",
            "source_frames": [str(frame) for frame in frames],
            "source_annotation": str(annotation_path),
            "source_sam_tracks": str(sam_tracks_path),
            "source_sam_contact": str(sam_contact),
            "source_vlm_naming": str(naming_path) if naming_path else None,
            "segmentation_model": sam_payload.get("model", "facebook/sam2.1-hiera-tiny"),
            "naming_model": naming_payload.get("model") if naming_payload else None,
            "vqa_model": model,
            "benchmark_ready": False,
            "requires_human_verification": True,
            "items": items,
            "skipped_categories": [item.model_dump(mode="json") for item in result.skipped_categories],
            "usage": response.usage.model_dump(mode="json") if response.usage else {},
        }
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
        print(f"[SteelBench {index}/{len(jobs)}] saved · questions={len(items)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
