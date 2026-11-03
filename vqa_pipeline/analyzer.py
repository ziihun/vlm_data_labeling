from __future__ import annotations

import base64
import json
from typing import Any

from openai import OpenAI

from .models import CATEGORY_ORDER, VideoVQA
from .video import FrameSample


SYSTEM_PROMPT = """You create evidence-grounded VQA data for industrial CCTV videos.

The user supplies uniformly sampled video frames in chronological order. Build a pilot
structured annotation and exactly seven Korean VQA items. Follow these constraints:

1. Use only visible evidence in the supplied frames. Sparse frames are not continuous
   video: do not claim unseen motion, causality, intent, authorization, PPE status, task
   completion, or object absence unless the frames support it.
2. Do not perform Risk Reasoning. Do not ask whether a scene is dangerous, explain risk,
   predict accidents, or recommend prevention.
3. Produce exactly one VQA item for every required category named in the user prompt.
4. Questions and answers must be concise Korean. Evidence rationales must be Korean.
5. Every evidence frame index must refer to a supplied frame. Use the corresponding
   timestamp. If completion/restoration is not visible, return `unclear` instead of guessing.
6. Object annotations are pseudo annotations from sampled frames. Use entity IDs like
   obj_001, but never claim segmentation masks or true tracking IDs.
7. Query-scene alignment items should include a useful mix of positive, negative, and
   hard-negative queries. A hard negative should contradict a directly observed attribute,
   relation, action, or time rather than inventing an arbitrary absent object.
8. Mark low-confidence, negative, hard-negative, or sparse-temporal conclusions for human
   review. The final data will always be human verified.
"""


class OpenAIVQAAnalyzer:
    """Calls the Responses API and validates structured, frame-grounded output."""

    def __init__(
        self,
        api_key: str,
        model: str,
        image_detail: str = "auto",
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.image_detail = image_detail
        self.client = client or OpenAI(api_key=api_key, timeout=180.0, max_retries=2)

    @staticmethod
    def _data_url(frame: FrameSample) -> str:
        encoded = base64.b64encode(frame.path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def build_user_content(
        self,
        video_id: str,
        duration: float,
        frames: list[FrameSample],
        pilot_index: int,
    ) -> list[dict[str, Any]]:
        polarity_rotation = (
            "object=positive, action_relation=hard_negative, temporal=negative"
            if pilot_index % 2 == 0
            else "object=hard_negative, action_relation=positive, temporal=positive"
        )
        prompt = f"""Analyze video `{video_id}` from {len(frames)} sampled frames.
Video duration: {duration:.3f} seconds.

Required categories, exactly once each:
{json.dumps(CATEGORY_ORDER, ensure_ascii=False, indent=2)}

Preferred alignment polarity mix for this video: {polarity_rotation}.
The dataset's ground-truth class is deliberately hidden to avoid label leakage.
Only use the chronological frames and their timestamps below.
"""
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for frame in frames:
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": (
                            f"Frame {frame.index} at "
                            f"{frame.timestamp_seconds:.3f} seconds"
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": self._data_url(frame),
                        "detail": self.image_detail,
                    },
                ]
            )
        return content

    @staticmethod
    def validate_grounding(result: VideoVQA, duration: float, frame_count: int) -> None:
        valid_indices = set(range(frame_count))

        def check_indices(indices: list[int], owner: str) -> None:
            if not indices:
                raise ValueError(f"{owner} has no evidence frames")
            if not set(indices).issubset(valid_indices):
                raise ValueError(f"{owner} references invalid frame indices: {indices}")

        for annotation in result.object_annotations:
            check_indices(annotation.evidence_frame_indices, annotation.entity_id)
            if annotation.first_seen_seconds > annotation.last_seen_seconds:
                raise ValueError(f"Invalid object time range for {annotation.entity_id}")
            if annotation.last_seen_seconds > duration + 0.1:
                raise ValueError(f"Object timestamp exceeds video duration: {annotation.entity_id}")

        for event in result.temporal_events:
            check_indices(event.evidence_frame_indices, event.event_id)
            if event.start_seconds > event.end_seconds or event.end_seconds > duration + 0.1:
                raise ValueError(f"Invalid event time range for {event.event_id}")

        check_indices(result.task_state.evidence_frame_indices, "task_state")
        for item in result.vqa:
            check_indices(item.evidence.frame_indices, item.category)
            if len(item.evidence.frame_indices) != len(item.evidence.timestamps_seconds):
                raise ValueError(f"Evidence index/timestamp mismatch for {item.category}")
            for timestamp in item.evidence.timestamps_seconds:
                if timestamp < 0 or timestamp > duration + 0.1:
                    raise ValueError(f"Evidence timestamp out of range for {item.category}")
            if item.timestamp_start is not None and item.timestamp_end is not None:
                if item.timestamp_start > item.timestamp_end:
                    raise ValueError(f"Invalid VQA time range for {item.category}")
                if item.timestamp_end > duration + 0.1:
                    raise ValueError(f"VQA timestamp exceeds video duration for {item.category}")

    def analyze(
        self,
        video_id: str,
        duration: float,
        frames: list[FrameSample],
        pilot_index: int,
    ) -> tuple[VideoVQA, dict[str, Any]]:
        response = self.client.responses.parse(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": self.build_user_content(
                        video_id=video_id,
                        duration=duration,
                        frames=frames,
                        pilot_index=pilot_index,
                    ),
                }
            ],
            text_format=VideoVQA,
            max_output_tokens=6000,
            store=False,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError(f"Model returned no parsed output (response id: {response.id})")
        self.validate_grounding(parsed, duration=duration, frame_count=len(frames))
        return parsed, {
            "response_id": response.id,
            "model": response.model,
            "usage": response.usage.model_dump(mode="json") if response.usage else None,
        }
