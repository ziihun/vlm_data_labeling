from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, model_validator

from .models import CATEGORY_ORDER, Category, StrictModel


CHANGE_REQUIRED_CATEGORIES: frozenset[Category] = frozenset(
    {
        "scene.action_spatial_understanding",
        "scene.temporal_understanding",
        "scene.task_state_completion_understanding",
        "alignment.action_relation_alignment",
        "alignment.temporal_alignment",
    }
)

FORBIDDEN_QA_PATTERN = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:초|seconds?|s\b))|"
    r"(?:(?:frame|프레임)\s*#?\s*\d+)|"
    r"(?:[CFG]\d{2,3}[_-]?(?:T|M)?\d{2,3})",
    re.IGNORECASE,
)


class TrackDescription(StrictModel):
    global_track_id: str
    local_track_ids: list[str]
    description: str
    provider_status: Literal["executed", "fallback"]
    requested_model: str
    executed_model: str
    representative_frame_count: int = Field(ge=1, le=8)


class TrackedObject(StrictModel):
    global_track_id: str
    local_track_ids: list[str]
    canonical_name: str
    attributes: list[str]
    description: str
    first_seen_seconds: float = Field(ge=0)
    last_seen_seconds: float = Field(ge=0)
    verification_status: Literal[
        "sam3_matched", "sam3_mismatch", "sam3_blocked", "pending_review"
    ]
    mean_iou: float | None = Field(default=None, ge=0, le=1)
    requires_human_review: bool


class ObservedChange(StrictModel):
    change_id: str
    change_type: Literal[
        "appearance",
        "disappearance",
        "movement",
        "state_change",
        "action",
        "completion",
        "restoration",
    ]
    global_track_ids: list[str]
    before_state: str
    after_state: str
    before_timestamp_seconds: float = Field(ge=0)
    after_timestamp_seconds: float = Field(ge=0)
    before_clip_id: str
    after_clip_id: str
    before_tracking_frame_index: int = Field(ge=0)
    after_tracking_frame_index: int = Field(ge=0)
    directly_observed: bool
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_before_after(self) -> "ObservedChange":
        if self.before_timestamp_seconds >= self.after_timestamp_seconds:
            raise ValueError("Observed change requires before time < after time")
        if self.before_state.strip() == self.after_state.strip():
            raise ValueError("Observed change requires distinct before and after states")
        return self


class TrackedAnnotation(StrictModel):
    scene_summary: str
    objects: list[TrackedObject]
    observed_changes: list[ObservedChange]
    limitations: list[str]


class EvidenceMetadata(StrictModel):
    global_track_ids: list[str]
    clip_ids: list[str]
    tracking_frame_indices: list[int]
    source_timestamps_seconds: list[float]
    observed_change_ids: list[str]
    rationale: str

    @model_validator(mode="after")
    def validate_parallel_frame_metadata(self) -> "EvidenceMetadata":
        count = len(self.clip_ids)
        if count == 0:
            raise ValueError("Evidence must contain at least one frame reference")
        if not (
            count
            == len(self.tracking_frame_indices)
            == len(self.source_timestamps_seconds)
        ):
            raise ValueError("Evidence clip/frame/timestamp arrays must have equal lengths")
        return self


class GroundedVQAItem(StrictModel):
    category: Category
    question: str
    answer: str
    answer_type: Literal["yes_no", "short_text"]
    polarity: Literal["positive", "negative", "hard_negative", "not_applicable"]
    evidence: EvidenceMetadata
    confidence: float = Field(ge=0, le=1)
    requires_human_review: bool

    @model_validator(mode="after")
    def prevent_metadata_questions(self) -> "GroundedVQAItem":
        if FORBIDDEN_QA_PATTERN.search(self.question) or FORBIDDEN_QA_PATTERN.search(self.answer):
            raise ValueError(
                "Question/answer must not expose timestamps, frame indices, or internal IDs"
            )
        if self.category in CHANGE_REQUIRED_CATEGORIES:
            if not self.evidence.observed_change_ids:
                raise ValueError(f"{self.category} requires an observed change")
            if len(set(self.evidence.source_timestamps_seconds)) < 2:
                raise ValueError(f"{self.category} requires distinct before/after evidence")
        return self


class SkippedCategory(StrictModel):
    category: Category
    reason_code: Literal[
        "no_direct_before_after_change",
        "insufficient_verified_annotation",
        "duplicate_overlap_event",
        "no_grounded_question",
    ]
    detail: str


class GroundedVQABatch(StrictModel):
    items: list[GroundedVQAItem]
    skipped_categories: list[SkippedCategory]

    @model_validator(mode="after")
    def cover_each_category_once_without_forcing_questions(self) -> "GroundedVQABatch":
        item_categories = [item.category for item in self.items]
        skipped_categories = [item.category for item in self.skipped_categories]
        combined = item_categories + skipped_categories
        if len(combined) != len(CATEGORY_ORDER) or set(combined) != set(CATEGORY_ORDER):
            raise ValueError(
                "Every category must appear exactly once as generated or skipped; "
                f"got {combined}"
            )
        if len(set(item_categories)) != len(item_categories):
            raise ValueError("Generated categories must be unique")
        if len(set(skipped_categories)) != len(skipped_categories):
            raise ValueError("Skipped categories must be unique")
        return self


class Sam3Verification(StrictModel):
    global_track_id: str
    canonical_name: str
    status: Literal["matched", "mismatch", "blocked"]
    frame_ious: list[float]
    mean_iou: float | None = Field(default=None, ge=0, le=1)
    iou_threshold: float = Field(ge=0, le=1)
    requires_human_review: bool
    reason: str
