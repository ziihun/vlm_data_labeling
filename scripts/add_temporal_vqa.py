from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPORAL_CATEGORIES = (
    "scene.temporal_understanding",
    "scene.task_state_completion_understanding",
    "alignment.temporal_alignment",
)
METADATA_PATTERN = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:seconds?|s\b))|(?:(?:frame|clip|track)[_ #\-]?\d+)",
    re.IGNORECASE,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TemporalEvent(StrictModel):
    event_id: str = Field(pattern=r"^EVT_\d{3,}$")
    description: str
    event_type: Literal["action", "spatial_change", "state_change", "completion", "restoration"]
    before_timestamp_seconds: float = Field(ge=0)
    after_timestamp_seconds: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def chronological(self) -> "TemporalEvent":
        if self.before_timestamp_seconds >= self.after_timestamp_seconds:
            raise ValueError("before must be earlier than after")
        return self


class TemporalQuestion(StrictModel):
    category: Literal[
        "scene.temporal_understanding",
        "scene.task_state_completion_understanding",
        "alignment.temporal_alignment",
    ]
    question: str
    answer: str
    answer_type: Literal["yes_no", "short_text"]
    event_ids: list[str] = Field(min_length=1)
    evidence_timestamps_seconds: list[float] = Field(min_length=2)
    rationale: str
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def no_metadata(self) -> "TemporalQuestion":
        if METADATA_PATTERN.search(self.question) or METADATA_PATTERN.search(self.answer):
            raise ValueError("question and answer cannot expose timestamps or frame metadata")
        return self


class TemporalResult(StrictModel):
    events: list[TemporalEvent]
    questions: list[TemporalQuestion]
    skipped_categories: list[str]


PROMPT = """Analyze the ordered 5 FPS contact sheets from one industrial CCTV video.
Each sheet is ordered left-to-right, top-to-bottom, and every cell displays its source timestamp.

Create only directly observable temporal events and English VQA for these categories:
- scene.temporal_understanding: observed order, movement, appearance/disappearance, or before/after state change.
- scene.task_state_completion_understanding: only an actually observed completion or restoration.
- alignment.temporal_alignment: whether a query's temporal ordering agrees with the observed event.

Rules:
1. Timestamps are evidence metadata only. Never mention timestamps, frame numbers, sheets, or sampling in a question or answer.
2. Do not infer an action from object presence. Require a visible before/after change.
3. Do not invent hidden intent, safety compliance, or events between sampled frames.
4. Generate every distinct grounded question supported by the visible events; do not create paraphrase duplicates.
5. Every question must cite one or more returned event_ids and exactly the event's before/after timestamps.
6. task_state_completion questions may cite only completion/restoration events.
7. Use positive or neutral questions. Do not manufacture negatives from non-detection.
8. If a category has no valid question, include its identifier in skipped_categories.
9. Use sequential event IDs EVT_001, EVT_002, ... .
"""


def extract_sheets(video: Path, work: Path, fps: float = 5.0) -> list[Path]:
    frames = work / "frames"
    frames.mkdir(parents=True)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
        "-vf", f"fps={fps:g},scale=640:-2", "-q:v", "3", str(frames / "%06d.jpg"),
    ], check=True)
    frame_paths = sorted(frames.glob("*.jpg"))
    sheets_dir = work / "sheets"
    sheets_dir.mkdir()
    sheets: list[Path] = []
    cell_w, cell_h, columns, rows = 320, 200, 4, 4
    for sheet_index, start in enumerate(range(0, len(frame_paths), columns * rows)):
        canvas = Image.new("RGB", (cell_w * columns, cell_h * rows), "black")
        draw = ImageDraw.Draw(canvas)
        for offset, frame_path in enumerate(frame_paths[start:start + columns * rows]):
            image = Image.open(frame_path).convert("RGB")
            image.thumbnail((cell_w, cell_h - 20))
            x = (offset % columns) * cell_w
            y = (offset // columns) * cell_h
            canvas.paste(image, (x + (cell_w - image.width) // 2, y + 20))
            timestamp = (start + offset) / fps
            draw.rectangle((x, y, x + cell_w, y + 20), fill="black")
            draw.text((x + 6, y + 4), f"t={timestamp:.1f}s", fill="white")
        path = sheets_dir / f"sheet_{sheet_index:02d}.jpg"
        canvas.save(path, quality=90)
        sheets.append(path)
    return sheets


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def next_item_number(items: list[dict]) -> int:
    numbers = []
    for item in items:
        match = re.fullmatch(r"VQA_(\d+)", item.get("item_id", ""))
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def validate_result(result: TemporalResult, duration: float) -> None:
    events = {event.event_id: event for event in result.events}
    if len(events) != len(result.events):
        raise ValueError("duplicate temporal event ID")
    questions = set()
    for event in result.events:
        if event.after_timestamp_seconds > duration + 0.25:
            raise ValueError("event timestamp exceeds video duration")
    for question in result.questions:
        normalized = question.question.strip().casefold()
        if normalized in questions:
            raise ValueError("duplicate temporal question")
        questions.add(normalized)
        cited = [events[event_id] for event_id in question.event_ids]
        if question.category == "scene.task_state_completion_understanding" and any(
            event.event_type not in {"completion", "restoration"} for event in cited
        ):
            raise ValueError("completion question cites a non-completion event")
        allowed = {
            round(timestamp, 1)
            for event in cited
            for timestamp in (event.before_timestamp_seconds, event.after_timestamp_seconds)
        }
        supplied = {round(timestamp, 1) for timestamp in question.evidence_timestamps_seconds}
        if supplied != allowed:
            raise ValueError("question timestamps must exactly match cited events")


def video_duration(path: Path) -> float:
    completed = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], check=True, capture_output=True, text=True)
    return float(completed.stdout.strip())


def update_payload(payload: dict, result: TemporalResult, model: str, usage: dict) -> dict:
    items = [item for item in payload.get("items", []) if item.get("category") not in TEMPORAL_CATEGORIES]
    item_number = next_item_number(items)
    event_by_id = {event.event_id: event for event in result.events}
    for question in result.questions:
        items.append({
            "item_id": f"VQA_{item_number:03d}",
            "category": question.category,
            "question": question.question,
            "answer": question.answer,
            "answer_type": question.answer_type,
            "polarity": "positive" if question.answer_type == "yes_no" else "not_applicable",
            "evidence": {
                "track_ids": [],
                "event_ids": question.event_ids,
                "negative_fact_ids": [],
                "spatial_relation_ids": [],
                "timestamps_seconds": question.evidence_timestamps_seconds,
                "rationale": question.rationale,
            },
            "confidence": question.confidence,
            "draft": True,
            "requires_final_review": True,
        })
        item_number += 1
    generated = {item["category"] for item in items}
    skipped = [item for item in payload.get("skipped_categories", []) if item.get("category") not in TEMPORAL_CATEGORIES]
    reason = "No directly observed before/after event supported a grounded question."
    for category in TEMPORAL_CATEGORIES:
        if category not in generated:
            skipped.append({"category": category, "reason": reason})
    payload["items"] = items
    payload["skipped_categories"] = skipped
    payload["verified_temporal_events"] = [event.model_dump(mode="json") for event in result.events]
    payload["temporal_generation"] = {
        "status": "completed", "model": model, "sampling_fps": 5.0,
        "question_count": len(result.questions), "usage": usage,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-6.1-terra")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=600.0)
    dataset_dir = args.dataset_dir.resolve()
    backup_dir = dataset_dir / "pre_temporal_backup"
    audit_dir = dataset_dir / "temporal_api"
    backup_dir.mkdir(exist_ok=True)
    audit_dir.mkdir(exist_ok=True)

    files = sorted(dataset_dir.glob("*.json"))
    for index, path in enumerate(files, start=1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("temporal_generation") or {}).get("status") == "completed":
            print(f"[Temporal {index}/{len(files)}] resume · {payload['video_id']}", flush=True)
            continue
        video = Path(payload["source_video"])
        duration = video_duration(video)
        print(f"[Temporal {index}/{len(files)}] analyze · {payload['video_id']}", flush=True)
        work = Path(tempfile.mkdtemp(prefix=f"temporal_{payload['video_id']}_"))
        try:
            sheets = extract_sheets(video, work)
            content: list[dict] = [{"type": "input_text", "text": PROMPT}]
            content.extend({"type": "input_image", "image_url": image_data_url(sheet), "detail": "high"} for sheet in sheets)
            response = client.responses.parse(
                model=model,
                instructions="Return schema-valid temporal events and grounded temporal VQA only.",
                input=[{"role": "user", "content": content}],
                text_format=TemporalResult,
                max_output_tokens=16000,
                store=False,
            )
            result = response.output_parsed
            if result is None:
                raise RuntimeError("OpenAI returned no parsed temporal result")
            validate_result(result, duration)
            usage = response.usage.model_dump(mode="json") if response.usage else {}
            if not (backup_dir / path.name).exists():
                shutil.copy2(path, backup_dir / path.name)
            (audit_dir / path.name).write_text(response.model_dump_json(indent=2) + "\n", encoding="utf-8")
            update_payload(payload, result, model, usage)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
            print(f"[Temporal {index}/{len(files)}] saved · events={len(result.events)} questions={len(result.questions)}", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
