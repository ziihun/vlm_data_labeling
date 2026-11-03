from pathlib import Path

import pytest
from PIL import Image

from vqa_pipeline.discovery import DiscoveryArtifactWriter, DiscoveryFirstPipeline
from vqa_pipeline.discovery_models import (
    DiscoveryAnnotation,
    DiscoveryObjectAnnotation,
    FrameSemantics,
    RegionSemantics,
)
from vqa_pipeline.models import TaskState


def semantics(*region_ids: str) -> list[FrameSemantics]:
    return [
        FrameSemantics(
            frame_index=0,
            timestamp_seconds=0.0,
            regions=[
                RegionSemantics(
                    region_id=region_id,
                    object_name="객체",
                    visible_attributes=[],
                    visible_state="보임",
                    description="테스트 객체",
                    confidence=0.9,
                    uncertainty="없음",
                )
                for region_id in region_ids
            ],
        )
    ]


def annotation(*region_ids: str) -> DiscoveryAnnotation:
    return DiscoveryAnnotation(
        scene_summary="테스트 장면",
        object_annotations=[
            DiscoveryObjectAnnotation(
                entity_id="E01",
                canonical_name="객체",
                attributes=[],
                estimated_count=1,
                first_seen_seconds=0.0,
                last_seen_seconds=0.0,
                evidence_frame_indices=[0],
                sam_region_ids=list(region_ids),
                association_confidence=0.9,
                association_note="샘플 프레임 연관",
            )
        ],
        temporal_events=[],
        task_state=TaskState(
            activity="불명확",
            completion_status="unclear",
            restoration_status="unclear",
            evidence="표본만으로 판단 불가",
            evidence_frame_indices=[0],
        ),
        limitations=["테스트"],
    )


def test_region_coverage_accepts_every_id_exactly_once() -> None:
    DiscoveryFirstPipeline.validate_region_coverage(
        semantics("F00_M001", "F00_M002"),
        annotation("F00_M001", "F00_M002"),
    )


def test_region_coverage_rejects_missing_id() -> None:
    with pytest.raises(ValueError, match="F00_M002"):
        DiscoveryFirstPipeline.validate_region_coverage(
            semantics("F00_M001", "F00_M002"),
            annotation("F00_M001"),
        )


def test_crop_sheet_contains_every_proposal(tmp_path: Path) -> None:
    crops = tmp_path / "crops"
    crops.mkdir()
    proposals = []
    for index in range(5):
        region_id = f"F00_M{index + 1:03d}"
        Image.new("RGB", (40, 30), (index * 20, 100, 100)).save(crops / f"{region_id}.jpg")
        proposals.append({"region_id": region_id, "crop_path": f"crops/{region_id}.jpg"})

    sheet_path = DiscoveryArtifactWriter.crop_sheet(tmp_path, proposals)
    assert sheet_path.exists()
    assert Image.open(sheet_path).size == (760, 290)
