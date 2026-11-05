from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from vqa_pipeline.annotation import apply_human_review
from vqa_pipeline.crops import MarginCropGenerator
from vqa_pipeline.naming_experiment import MODEL_SPECS, create_contact_sheet
from vqa_pipeline.naming_experiment import PROMPT_TEMPLATE as NAMING_PROMPT
from vqa_pipeline.openai_vqa import PROMPT_TEMPLATE as VQA_PROMPT
from vqa_pipeline.spatial_relations import StableBBoxSpatialRelationBuilder
from vqa_pipeline.vqa_mapping import DEFAULT_THRESHOLD, normalize_experiment_naming
from vqa_pipeline.spec_models import (
    CATEGORY_ORDER,
    DraftVQABatch,
    DraftVQAItem,
    HumanReview,
    SkippedCategory,
    VQAEvidence,
    VerifiedAnnotation,
    VerifiedObject,
)
from vqa_pipeline.track_merge import OverlapTrackMerger
from vqa_pipeline.video_clips import OverlappingClipPreprocessor


def test_preprocessing_windows_are_10_seconds_with_5_second_stride() -> None:
    windows = OverlappingClipPreprocessor().windows(14.981877)
    assert [(item.start_seconds, item.end_seconds) for item in windows] == [(0.0, 10.0), (5.0, 14.981877)]


def test_margin_crop_is_twenty_percent_and_capped_at_eight(tmp_path: Path) -> None:
    image = Image.new("RGB", (100, 100), "white")
    mask = Image.new("L", (100, 100), 0)
    for x in range(20, 40):
        for y in range(30, 50):
            mask.putpixel((x, y), 255)
    image_path, mask_path = tmp_path / "frame.jpg", tmp_path / "mask.png"
    image.save(image_path)
    mask.save(mask_path)
    observations = [{"frame_index": i, "source_timestamp_seconds": i * 0.2, "source_frame_path": str(image_path), "mask_path": str(mask_path), "bbox_xyxy": [20, 30, 40, 50]} for i in range(10)]
    payload = MarginCropGenerator().run({"tracks": [{"track_id": "G0001", "first_seen_seconds": 0, "last_seen_seconds": 2, "observations": observations}]}, tmp_path / "crops")
    assert len(payload["tracks"][0]["evidence"]) == 8
    context = Image.open(payload["tracks"][0]["evidence"][0]["context_crop_path"])
    assert context.size == (28, 28)


def test_mask_iou_uses_mask_pixels() -> None:
    first, second = Image.new("L", (4, 4), 0), Image.new("L", (4, 4), 0)
    first.putpixel((1, 1), 255)
    first.putpixel((2, 1), 255)
    second.putpixel((2, 1), 255)
    second.putpixel((3, 1), 255)
    assert OverlapTrackMerger.mask_iou(first, second) == pytest.approx(1 / 3)


def test_negative_vqa_requires_human_verified_negative_fact() -> None:
    with pytest.raises(ValidationError, match="human-verified negative fact"):
        DraftVQAItem(
            item_id="VQA_001", category="scene.object_attribute_recognition", question="Is the worker wearing gloves?", answer="No.",
            answer_type="yes_no", polarity="negative",
            evidence=VQAEvidence(track_ids=["G0001"], event_ids=[], negative_fact_ids=[], spatial_relation_ids=[], timestamps_seconds=[1.0], rationale="none"),
            confidence=0.8, draft=True, requires_final_review=True,
        )


def test_question_cannot_expose_timestamp() -> None:
    with pytest.raises(ValidationError, match="must not expose evidence metadata"):
        DraftVQAItem(
            item_id="VQA_001", category="scene.object_attribute_recognition", question="What color is visible at 1.0 seconds?", answer="Yellow.",
            answer_type="short_text", polarity="not_applicable",
            evidence=VQAEvidence(track_ids=["G0001"], event_ids=[], negative_fact_ids=[], spatial_relation_ids=[], timestamps_seconds=[1.0], rationale="visible"),
            confidence=0.8, draft=True, requires_final_review=True,
        )


def test_batch_can_skip_unsupported_categories() -> None:
    batch = DraftVQABatch(
        video_id="test", items=[],
        skipped_categories=[SkippedCategory(category=category, reason="insufficient verified evidence") for category in CATEGORY_ORDER],
    )
    assert len(batch.skipped_categories) == 7


def test_human_review_must_cover_every_track(tmp_path: Path) -> None:
    pending = {"objects": [{"track_id": "G0001"}, {"track_id": "G0002"}]}
    review = HumanReview(video_id="test", review_status="completed", object_reviews=[], verified_events=[], verified_negative_facts=[])
    with pytest.raises(ValueError, match="cover every Track"):
        apply_human_review(pending, review, tmp_path)


def test_naming_contact_sheet_normalizes_eight_images(tmp_path: Path) -> None:
    paths = []
    for index in range(8):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (20 + index, 40), (index * 20, 30, 50)).save(path)
        paths.append(path)
    output = tmp_path / "grid.jpg"
    create_contact_sheet(paths, [f"sample {index + 1}" for index in range(8)], output)
    with Image.open(output) as image:
        assert image.size == (1536, 836)


def test_naming_comparison_includes_non_generative_baselines() -> None:
    assert {"ram_plus", "siglip2", "yoloe_prompt_free"}.issubset(MODEL_SPECS)


def test_naming_comparison_includes_requested_hf_vlms() -> None:
    assert {"internvl3_5_4b", "molmo2_4b", "minicpm_v4"}.issubset(MODEL_SPECS)


def test_qwen9_gate_threshold_is_point_seven() -> None:
    assert DEFAULT_THRESHOLD == 0.7


def test_experiment_naming_is_normalized_for_strict_annotation_schema() -> None:
    normalized = normalize_experiment_naming({
        "model": "qwen9",
        "items": [{
            "track_id": "G0001", "canonical_object_name": "sign",
            "visible_attributes": {"color": "white"}, "observable_state": "stationary",
            "confidence": 0.95, "uncertainty_reason": "", "latency_seconds": 1.2,
            "parse_error": None,
        }],
    })
    assert normalized["items"][0]["raw_response"]
    assert "latency_seconds" not in normalized["items"][0]


def test_detailed_naming_prompt_avoids_vague_labels_and_calibrates_confidence() -> None:
    assert "most specific English noun phrase" in NAMING_PROMPT
    assert '"object", "item", "block", "panel"' in NAMING_PROMPT
    assert "0.90-1.00 only" in NAMING_PROMPT


def test_vqa_prompt_forbids_invented_timestamps_and_forced_categories() -> None:
    assert "never invent or interpolate" in VQA_PROMPT
    assert "Never force category coverage" in VQA_PROMPT
    assert "natural English" in VQA_PROMPT
    assert "Maximize the number and diversity" in VQA_PROMPT


def test_final_vqa_rejects_non_english_question() -> None:
    with pytest.raises(ValidationError, match="must be English"):
        DraftVQAItem(
            item_id="VQA_001", category="scene.object_attribute_recognition",
            question="표지판은 무슨 색인가요?", answer="Red.",
            answer_type="short_text", polarity="not_applicable",
            evidence=VQAEvidence(
                track_ids=["G0001"], event_ids=[], negative_fact_ids=[], spatial_relation_ids=[],
                timestamps_seconds=[1.0], rationale="visible",
            ),
            confidence=0.8, draft=True, requires_final_review=True,
        )


def test_stable_bbox_builder_creates_left_of_relation(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 100), "white").save(frame)
    def verified(track_id: str, bbox: list[int]) -> VerifiedObject:
        return VerifiedObject(
            video_id="test", track_id=track_id, object_name=track_id,
            visible_attributes={}, observable_state="stationary", bbox_xyxy=bbox,
            first_seen_seconds=0, last_seen_seconds=0.8, segmentation_model="sam2",
            naming_model="qwen", verification_status="approved",
            evidence_frames=[0], evidence_timestamps_seconds=[0.0],
        )
    annotation = VerifiedAnnotation(
        video_id="test",
        objects=[verified("G0001", [10, 40, 20, 50]), verified("G0002", [70, 40, 80, 50])],
        verified_events=[], verified_negative_facts=[],
    )
    tracks = {"tracks": [
        {"track_id": "G0001", "observations": [
            {"source_timestamp_seconds": i * 0.2, "source_frame_path": str(frame), "bbox_xyxy": [10, 40, 20, 50]}
            for i in range(5)
        ]},
        {"track_id": "G0002", "observations": [
            {"source_timestamp_seconds": i * 0.2, "source_frame_path": str(frame), "bbox_xyxy": [70, 40, 80, 50]}
            for i in range(5)
        ]},
    ]}
    result = StableBBoxSpatialRelationBuilder().build(annotation, tracks)
    assert len(result.verified_spatial_relations) == 1
    assert result.verified_spatial_relations[0].relation == "left_of"
