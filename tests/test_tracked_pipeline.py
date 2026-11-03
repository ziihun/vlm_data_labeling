from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from vqa_pipeline.models import CATEGORY_ORDER
from vqa_pipeline.track_merge import OverlapTrackMerger
from vqa_pipeline.tracking_models import (
    EvidenceMetadata,
    GroundedVQABatch,
    GroundedVQAItem,
    SkippedCategory,
)
from vqa_pipeline.video_clips import OverlappingClipPreprocessor


def evidence(change_ids: list[str] | None = None) -> EvidenceMetadata:
    return EvidenceMetadata(
        global_track_ids=["G0001"],
        clip_ids=["clip_000", "clip_000"],
        tracking_frame_indices=[3, 7],
        source_timestamps_seconds=[0.6, 1.4],
        observed_change_ids=change_ids or [],
        rationale="두 근거 화면에서 직접 확인된다.",
    )


def test_clip_windows_are_10_seconds_with_5_second_overlap() -> None:
    windows = OverlappingClipPreprocessor().windows(14.981877)
    assert [(item.start_seconds, item.end_seconds) for item in windows] == [
        (0.0, 10.0),
        (5.0, 14.981877),
    ]


def test_question_cannot_ask_for_timestamp() -> None:
    with pytest.raises(ValidationError, match="must not expose timestamps"):
        GroundedVQAItem(
            category="scene.object_attribute_recognition",
            question="6.399초에 보이는 장비 색은 무엇인가요?",
            answer="파란색입니다.",
            answer_type="short_text",
            polarity="not_applicable",
            evidence=evidence(),
            confidence=0.9,
            requires_human_review=False,
        )


def test_temporal_question_requires_observed_change() -> None:
    with pytest.raises(ValidationError, match="requires an observed change"):
        GroundedVQAItem(
            category="scene.temporal_understanding",
            question="작업자가 떠난 뒤 펜스가 복구되었나요?",
            answer="아니요.",
            answer_type="yes_no",
            polarity="negative",
            evidence=evidence(),
            confidence=0.8,
            requires_human_review=True,
        )


def test_batch_allows_skipping_unsupported_categories() -> None:
    item = GroundedVQAItem(
        category="scene.object_attribute_recognition",
        question="설비 덮개는 무슨 색인가요?",
        answer="파란색입니다.",
        answer_type="short_text",
        polarity="not_applicable",
        evidence=evidence(),
        confidence=0.9,
        requires_human_review=False,
    )
    skipped = [
        SkippedCategory(
            category=category,
            reason_code="no_direct_before_after_change",
            detail="직접 관찰된 변화가 없다.",
        )
        for category in CATEGORY_ORDER
        if category != item.category
    ]
    batch = GroundedVQABatch(items=[item], skipped_categories=skipped)
    assert len(batch.items) == 1
    assert len(batch.skipped_categories) == 6


def test_mask_iou_uses_actual_mask_pixels() -> None:
    first = Image.new("L", (4, 4), 0)
    second = Image.new("L", (4, 4), 0)
    first.putpixel((1, 1), 255)
    first.putpixel((2, 1), 255)
    second.putpixel((2, 1), 255)
    second.putpixel((3, 1), 255)
    assert OverlapTrackMerger.mask_iou(first, second) == pytest.approx(1 / 3)
