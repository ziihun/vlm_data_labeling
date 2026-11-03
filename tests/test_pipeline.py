import json
from pathlib import Path

import pytest

from vqa_pipeline.analyzer import OpenAIVQAAnalyzer
from vqa_pipeline.dataset import DatasetCatalog
from vqa_pipeline.models import CATEGORY_ORDER, Evidence, TaskState, VQAItem, VideoVQA
from vqa_pipeline.pipeline import VQAPilotPipeline
from vqa_pipeline.video import VideoProcessor


def make_result() -> VideoVQA:
    items = [
        VQAItem(
            category=category,
            question=f"질문 {index}?",
            answer="예",
            answer_type="yes_no",
            polarity="positive",
            timestamp_start=0.0,
            timestamp_end=1.0,
            evidence=Evidence(
                frame_indices=[0, 1],
                timestamps_seconds=[0.0, 1.0],
                rationale="두 프레임에 근거가 보인다.",
            ),
            confidence=0.8,
            requires_human_review=False,
        )
        for index, category in enumerate(CATEGORY_ORDER)
    ]
    return VideoVQA(
        scene_summary="작업 장면",
        object_annotations=[],
        temporal_events=[],
        task_state=TaskState(
            activity="작업",
            completion_status="unclear",
            restoration_status="unclear",
            evidence="샘플 프레임만으로 불명확",
            evidence_frame_indices=[0, 1],
        ),
        vqa=items,
    )


def test_uniform_timestamps_include_safe_start_and_end() -> None:
    assert VideoProcessor.uniform_timestamps(10.0, 3) == [0.0, 4.975, 9.95]


def test_diverse_selection_prefers_distinct_labels() -> None:
    samples = [
        {"filepath": "data/b.mp4", "ground_truth": {"label": "Safe Walkway"}},
        {"filepath": "data/a.mp4", "ground_truth": {"label": "Safe Walkway"}},
        {
            "filepath": "data/c.mp4",
            "ground_truth": {"label": "Unauthorized Intervention"},
        },
    ]
    selected = DatasetCatalog(Path("unused")).select_diverse_samples(samples, 2)
    assert [sample["filepath"] for sample in selected] == ["data/a.mp4", "data/c.mp4"]


def test_model_requires_all_categories_once() -> None:
    result = make_result()
    assert [item.category for item in result.vqa] == list(CATEGORY_ORDER)
    result.vqa.pop()
    with pytest.raises(ValueError, match="exactly one item"):
        VideoVQA.model_validate(result.model_dump())


def test_grounding_rejects_invalid_frame_index() -> None:
    result = make_result()
    result.vqa[0].evidence.frame_indices = [99]
    with pytest.raises(ValueError, match="invalid frame indices"):
        OpenAIVQAAnalyzer.validate_grounding(result, duration=5.0, frame_count=2)


def test_zero_completed_videos_cannot_be_complete() -> None:
    summary = VQAPilotPipeline.build_summary(
        requested=5,
        completed=0,
        failed=1,
        records=[],
        category_counts={},
    )
    assert summary["all_categories_complete"] is False
    assert summary["videos_unattempted"] == 4


def test_review_video_is_a_complete_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    target = tmp_path / "review" / "source.mp4"
    source.write_bytes(b"full-motion-video")
    result = VideoProcessor.create_review_video(source, target)
    assert result.read_bytes() == source.read_bytes()


def test_downloaded_dataset_has_expected_shape() -> None:
    samples_path = (
        Path(__file__).resolve().parents[1]
        / "raw_datasets"
        / "Safe_and_Unsafe_Behaviours"
        / "samples.json"
    )
    if not samples_path.exists():
        pytest.skip("raw dataset is not present")
    payload = json.loads(samples_path.read_text(encoding="utf-8"))
    assert len(payload["samples"]) == 691
