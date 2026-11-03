from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Category = Literal[
    "scene.object_attribute_recognition",
    "scene.action_spatial_understanding",
    "scene.temporal_understanding",
    "scene.task_state_completion_understanding",
    "alignment.object_attribute_alignment",
    "alignment.action_relation_alignment",
    "alignment.temporal_alignment",
]

CATEGORY_ORDER: tuple[Category, ...] = (
    "scene.object_attribute_recognition",
    "scene.action_spatial_understanding",
    "scene.temporal_understanding",
    "scene.task_state_completion_understanding",
    "alignment.object_attribute_alignment",
    "alignment.action_relation_alignment",
    "alignment.temporal_alignment",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObjectAnnotation(StrictModel):
    entity_id: str = Field(description="Stable ID inside this sampled-video analysis, such as obj_001")
    name: str
    attributes: list[str]
    estimated_count: int = Field(ge=1)
    first_seen_seconds: float = Field(ge=0)
    last_seen_seconds: float = Field(ge=0)
    evidence_frame_indices: list[int]


class TemporalEvent(StrictModel):
    event_id: str
    description: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    evidence_frame_indices: list[int]


class TaskState(StrictModel):
    activity: str
    completion_status: Literal["completed", "in_progress", "unclear"]
    restoration_status: Literal["restored", "not_restored", "not_applicable", "unclear"]
    evidence: str
    evidence_frame_indices: list[int]


class Evidence(StrictModel):
    frame_indices: list[int]
    timestamps_seconds: list[float]
    rationale: str


class VQAItem(StrictModel):
    category: Category
    question: str
    answer: str
    answer_type: Literal["yes_no", "short_text", "timestamp"]
    polarity: Literal["positive", "negative", "hard_negative", "not_applicable"]
    timestamp_start: float | None
    timestamp_end: float | None
    evidence: Evidence
    confidence: float = Field(ge=0, le=1)
    requires_human_review: bool


class VideoVQA(StrictModel):
    scene_summary: str
    object_annotations: list[ObjectAnnotation]
    temporal_events: list[TemporalEvent]
    task_state: TaskState
    vqa: list[VQAItem]

    @model_validator(mode="after")
    def require_exact_category_coverage(self) -> "VideoVQA":
        actual = [item.category for item in self.vqa]
        if len(actual) != len(CATEGORY_ORDER) or set(actual) != set(CATEGORY_ORDER):
            raise ValueError(
                "vqa must contain exactly one item for each required category; "
                f"got {actual}"
            )
        if len({item.question.strip() for item in self.vqa}) != len(self.vqa):
            raise ValueError("VQA questions must be unique within a video")
        return self
