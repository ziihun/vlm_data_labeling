from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .models import CATEGORY_ORDER, Evidence, StrictModel, TaskState, TemporalEvent, VQAItem


class RegionSemantics(StrictModel):
    region_id: str
    object_name: str
    visible_attributes: list[str]
    visible_state: str
    description: str
    confidence: float = Field(ge=0, le=1)
    uncertainty: str


class FrameSemantics(StrictModel):
    frame_index: int
    timestamp_seconds: float
    regions: list[RegionSemantics]


class DiscoveryObjectAnnotation(StrictModel):
    entity_id: str
    canonical_name: str
    attributes: list[str]
    estimated_count: int = Field(ge=1)
    first_seen_seconds: float = Field(ge=0)
    last_seen_seconds: float = Field(ge=0)
    evidence_frame_indices: list[int]
    sam_region_ids: list[str]
    association_confidence: float = Field(ge=0, le=1)
    association_note: str


class DiscoveryAnnotation(StrictModel):
    scene_summary: str
    object_annotations: list[DiscoveryObjectAnnotation]
    temporal_events: list[TemporalEvent]
    task_state: TaskState
    limitations: list[str]


class VQACollection(StrictModel):
    vqa: list[VQAItem]

    @model_validator(mode="after")
    def require_exact_category_coverage(self) -> "VQACollection":
        actual = [item.category for item in self.vqa]
        if len(actual) != len(CATEGORY_ORDER) or set(actual) != set(CATEGORY_ORDER):
            raise ValueError(
                "vqa must contain exactly one item for each required category; "
                f"got {actual}"
            )
        if len({item.question.strip() for item in self.vqa}) != len(self.vqa):
            raise ValueError("VQA questions must be unique")
        return self


class DiscoveryResult(StrictModel):
    annotation: DiscoveryAnnotation
    vqa: list[VQAItem]
    annotation_source: Literal["sam2_regions_plus_visual_descriptions"]
