from __future__ import annotations

import base64
import io
import json
import urllib.request
from pathlib import Path
from typing import Any, TypeVar

from openai import OpenAI
from PIL import Image, ImageChops
from pydantic import BaseModel

from .models import CATEGORY_ORDER, Category
from .tracking_models import (
    CHANGE_REQUIRED_CATEGORIES,
    GroundedVQABatch,
    Sam3Verification,
    TrackDescription,
    TrackedAnnotation,
)


T = TypeVar("T", bound=BaseModel)


class DAMTrackDescriber:
    """Calls DAM's OpenAI-compatible server, with an explicit Terra MVP fallback."""

    def __init__(
        self,
        openai_client: OpenAI,
        openai_model: str,
        dam_base_url: str | None = None,
    ) -> None:
        self.openai_client = openai_client
        self.openai_model = openai_model
        self.dam_base_url = dam_base_url.rstrip("/") if dam_base_url else None

    @staticmethod
    def _data_url(path: Path) -> str:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _dam(self, frames: list[dict[str, Any]]) -> str:
        if self.dam_base_url is None:
            raise RuntimeError("DAM_BASE_URL is not configured")
        selected = frames
        if len(selected) not in (1, 8):
            selected = selected[:1]
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": self._data_url(Path(frame["rgba_path"]))},
            }
            for frame in selected
        ]
        content.append(
            {
                "type": "text",
                "text": "Describe the masked object, its visible attributes, state, and changes across the sequence in detail.",
            }
        )
        payload = json.dumps(
            {
                "model": "describe_anything_model",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 512,
                "temperature": 0.2,
                "top_p": 0.5,
                "num_beams": 1,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.dam_base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
        return str(result["choices"][0]["message"]["content"])

    def _terra(self, track: dict[str, Any]) -> tuple[str, str]:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    "DAM-3B-Video가 연결되지 않은 MVP fallback입니다. "
                    "각 이미지는 동일 SAM2 Track의 masked representative frame입니다. "
                    "객체 이름, 외형, 상태를 설명하고 실제로 보이는 변화만 서술하세요. "
                    "가려졌거나 모호한 내용은 추측하지 마세요.\n"
                    f"track={track['global_track_id']}"
                ),
            }
        ]
        for frame in track["representative_frames"]:
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": (
                            f"evidence source time metadata: "
                            f"{float(frame['source_timestamp_seconds']):.3f}s"
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": self._data_url(Path(frame["rgba_path"])),
                        "detail": "high",
                    },
                ]
            )
        response = self.openai_client.responses.create(
            model=self.openai_model,
            instructions=(
                "You describe one tracked object from chronological masked frames. "
                "Do not infer unobserved actions, intent, safety, completion, or restoration."
            ),
            input=[{"role": "user", "content": content}],
            max_output_tokens=1400,
            store=False,
        )
        return response.output_text, response.model

    def describe(self, track: dict[str, Any]) -> TrackDescription:
        frames = list(track["representative_frames"])
        if not frames:
            raise ValueError(f"Track {track['global_track_id']} has no DAM frames")
        if self.dam_base_url:
            description = self._dam(frames)
            status = "executed"
            executed_model = "nvidia/DAM-3B-Video"
        else:
            description, executed_model = self._terra(track)
            status = "fallback"
        return TrackDescription(
            global_track_id=track["global_track_id"],
            local_track_ids=track["local_track_ids"],
            description=description,
            provider_status=status,
            requested_model="nvidia/DAM-3B-Video",
            executed_model=executed_model,
            representative_frame_count=len(frames),
        )


class OpenAITrackedAnalyzer:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def _parse(
        self,
        schema: type[T],
        instructions: str,
        prompt: str,
        max_output_tokens: int,
    ) -> tuple[T, dict[str, Any]]:
        response = self.client.responses.parse(
            model=self.model,
            instructions=instructions,
            input=[{"role": "user", "content": prompt}],
            text_format=schema,
            max_output_tokens=max_output_tokens,
            store=False,
        )
        if response.output_parsed is None:
            raise RuntimeError(f"No parsed output from {response.id}")
        return response.output_parsed, {
            "response_id": response.id,
            "executed_model": response.model,
            "usage": response.usage.model_dump(mode="json") if response.usage else None,
        }

    def normalize(
        self,
        merged_tracks: dict[str, Any],
        descriptions: list[TrackDescription],
    ) -> tuple[TrackedAnnotation, dict[str, Any]]:
        track_evidence = [
            {
                "global_track_id": track["global_track_id"],
                "local_track_ids": track["local_track_ids"],
                "first_seen_seconds": track["first_seen_seconds"],
                "last_seen_seconds": track["last_seen_seconds"],
                "representative_frames": [
                    {
                        "clip_id": frame["clip_id"],
                        "frame_index": frame["frame_index"],
                        "source_timestamp_seconds": frame["source_timestamp_seconds"],
                    }
                    for frame in track["representative_frames"]
                ],
            }
            for track in merged_tracks["global_tracks"]
        ]
        prompt = f"""아래는 SAM2가 5 FPS로 추적한 전역 Track과 Track별 객체 설명입니다.
객체 이름과 속성을 구조화하세요. 모든 Track은 objects에 정확히 한 번 포함해야 합니다.

observed_changes는 대표 프레임 설명에서 before/after 상태가 실제로 다르게 관찰된 경우만 만드세요.
단순히 표본에서 위치가 다르다는 이유만으로 작업 완료, 복구, 행동 의도를 추측하지 마세요.
before/after clip_id, frame_index, timestamp는 제공된 representative_frames 값만 그대로 사용하세요.
변화를 직접 확인할 수 없으면 observed_changes를 빈 배열로 두세요.
SAM3 검증 전이므로 모든 객체 verification_status=pending_review, requires_human_review=true입니다.

tracks={json.dumps(track_evidence, ensure_ascii=False)}
descriptions={json.dumps([item.model_dump(mode='json') for item in descriptions], ensure_ascii=False)}
"""
        annotation, meta = self._parse(
            TrackedAnnotation,
            "산업 CCTV Track annotation을 보이는 증거만으로 구조화합니다. Risk Reasoning은 하지 않습니다.",
            prompt,
            max_output_tokens=9000,
        )
        expected = {track["global_track_id"] for track in merged_tracks["global_tracks"]}
        actual = {item.global_track_id for item in annotation.objects}
        if expected != actual or len(annotation.objects) != len(expected):
            raise ValueError(
                f"Track normalization coverage mismatch: missing={expected-actual}, extra={actual-expected}"
            )
        available_frames = {
            (
                frame["clip_id"],
                int(frame["frame_index"]),
                round(float(frame["source_timestamp_seconds"]), 3),
            )
            for track in merged_tracks["global_tracks"]
            for frame in track["representative_frames"]
        }
        for change in annotation.observed_changes:
            before = (
                change.before_clip_id,
                change.before_tracking_frame_index,
                round(change.before_timestamp_seconds, 3),
            )
            after = (
                change.after_clip_id,
                change.after_tracking_frame_index,
                round(change.after_timestamp_seconds, 3),
            )
            if before not in available_frames or after not in available_frames:
                raise ValueError(f"Observed change {change.change_id} invented evidence metadata")
        return annotation, meta

    @staticmethod
    def apply_sam3_results(
        annotation: TrackedAnnotation,
        verifications: list[Sam3Verification],
    ) -> TrackedAnnotation:
        by_track = {item.global_track_id: item for item in verifications}
        objects = []
        for item in annotation.objects:
            verification = by_track[item.global_track_id]
            status = {
                "matched": "sam3_matched",
                "mismatch": "sam3_mismatch",
                "blocked": "sam3_blocked",
            }[verification.status]
            objects.append(
                item.model_copy(
                    update={
                        "verification_status": status,
                        "mean_iou": verification.mean_iou,
                        "requires_human_review": verification.requires_human_review,
                    }
                )
            )
        return annotation.model_copy(update={"objects": objects})

    def generate_vqa(
        self,
        annotation: TrackedAnnotation,
        strict_verification: bool,
    ) -> tuple[GroundedVQABatch, dict[str, Any]]:
        prompt = f"""아래 검증 상태가 포함된 영상 annotation으로 한국어 VQA를 생성하세요.
카테고리 목록: {json.dumps(CATEGORY_ORDER, ensure_ascii=False)}

각 카테고리는 items 또는 skipped_categories 중 정확히 한 곳에만 넣으세요.
근거가 부족하면 질문을 억지로 만들지 말고 skipped_categories에 기록하세요.

절대 규칙:
1. question과 answer에 timestamp, 초 단위 숫자, frame index, clip/track 내부 ID를 쓰지 마세요.
2. timestamp와 frame index는 evidence metadata에만 기록하세요.
3. 시간·행동·작업완료·복구 질문은 directly_observed=true인 observed_change가 있을 때만 생성하세요.
4. 해당 문항 evidence에는 그 observed_change_id와 정확한 before/after metadata를 넣으세요.
5. 겹치는 clip은 이미 Track 단위로 병합됐습니다. 같은 객체나 사건 질문을 중복 생성하지 마세요.
6. Risk Reasoning은 생성하지 마세요.
7. strict_verification={str(strict_verification).lower()}입니다. true이면 sam3_matched 객체만 사용하세요.

annotation={json.dumps(annotation.model_dump(mode='json'), ensure_ascii=False)}
"""
        result, meta = self._parse(
            GroundedVQABatch,
            "근거가 관찰된 경우에만 질문을 만드는 보수적인 산업 VideoQA 생성기입니다.",
            prompt,
            max_output_tokens=7000,
        )
        changes = {item.change_id: item for item in annotation.observed_changes}
        objects = {item.global_track_id: item for item in annotation.objects}
        for item in result.items:
            if not set(item.evidence.global_track_ids).issubset(objects):
                raise ValueError(f"{item.category} references unknown Track ID")
            if strict_verification and any(
                objects[track_id].verification_status != "sam3_matched"
                for track_id in item.evidence.global_track_ids
            ):
                raise ValueError(f"{item.category} uses an unverified object")
            if item.category in CHANGE_REQUIRED_CATEGORIES:
                selected = [changes.get(change_id) for change_id in item.evidence.observed_change_ids]
                if not selected or any(change is None or not change.directly_observed for change in selected):
                    raise ValueError(f"{item.category} uses an unobserved change")
                if item.category == "scene.task_state_completion_understanding" and any(
                    change.change_type not in {"completion", "restoration"}
                    for change in selected
                    if change is not None
                ):
                    raise ValueError("Task completion question requires completion/restoration evidence")
        return result, meta


class Sam3NameVerifier:
    """Text-prompt SAM3 adapter. Without a configured server, records a blocked review."""

    def __init__(self, base_url: str | None, iou_threshold: float = 0.50) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.iou_threshold = iou_threshold

    @staticmethod
    def _png_base64(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _decode_mask(value: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("L").point(
            lambda pixel: 255 if pixel else 0
        )

    @staticmethod
    def _iou(first: Image.Image, second: Image.Image) -> float:
        intersection = ImageChops.multiply(first, second)
        union = ImageChops.lighter(first, second)
        intersection_pixels = sum(
            value * count for value, count in enumerate(intersection.histogram())
        ) / 255
        union_pixels = sum(
            value * count for value, count in enumerate(union.histogram())
        ) / 255
        return float(intersection_pixels / union_pixels) if union_pixels else 0.0

    def verify(self, annotation: TrackedAnnotation, merged_tracks: dict[str, Any]) -> list[Sam3Verification]:
        tracks = {item["global_track_id"]: item for item in merged_tracks["global_tracks"]}
        results = []
        for obj in annotation.objects:
            track = tracks[obj.global_track_id]
            if not self.base_url:
                results.append(
                    Sam3Verification(
                        global_track_id=obj.global_track_id,
                        canonical_name=obj.canonical_name,
                        status="blocked",
                        frame_ious=[],
                        mean_iou=None,
                        iou_threshold=self.iou_threshold,
                        requires_human_review=True,
                        reason="SAM3_BASE_URL/checkpoint is not configured on this Apple Silicon host.",
                    )
                )
                continue
            frames = []
            expected_masks: list[Image.Image] = []
            for frame in track["representative_frames"]:
                rgba = Image.open(frame["rgba_path"]).convert("RGBA")
                frames.append(self._png_base64(rgba.convert("RGB")))
                expected_masks.append(
                    Image.open(frame["mask_path"]).convert("L").point(
                        lambda pixel: 255 if pixel else 0
                    )
                )
            request_payload = json.dumps(
                {"text_prompt": obj.canonical_name, "frames_png_base64": frames}
            ).encode("utf-8")
            request = urllib.request.Request(
                f"{self.base_url}/redetect",
                data=request_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                detected = [self._decode_mask(value) for value in payload["masks_png_base64"]]
                ious = [self._iou(first, second) for first, second in zip(expected_masks, detected, strict=True)]
                mean = sum(ious) / len(ious) if ious else 0.0
                status = "matched" if mean >= self.iou_threshold else "mismatch"
                results.append(
                    Sam3Verification(
                        global_track_id=obj.global_track_id,
                        canonical_name=obj.canonical_name,
                        status=status,
                        frame_ious=[round(value, 6) for value in ious],
                        mean_iou=round(mean, 6),
                        iou_threshold=self.iou_threshold,
                        requires_human_review=status != "matched",
                        reason="SAM3 text redetection compared with SAM2 masks by mean IoU.",
                    )
                )
            except Exception as exc:
                results.append(
                    Sam3Verification(
                        global_track_id=obj.global_track_id,
                        canonical_name=obj.canonical_name,
                        status="blocked",
                        frame_ious=[],
                        mean_iou=None,
                        iou_threshold=self.iou_threshold,
                        requires_human_review=True,
                        reason=f"SAM3 server error: {type(exc).__name__}: {exc}",
                    )
                )
        return results
