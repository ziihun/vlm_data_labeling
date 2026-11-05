from __future__ import annotations

import re
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

METADATA_PATTERN = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:초|seconds?|s\b))|"
    r"(?:(?:frame|프레임)\s*#?\s*\d+)|"
    r"(?:\b(?:clip|track)[_-]?\d+\b)|(?:\bG\d{4}\b)",
    re.IGNORECASE,
)
HANGUL_PATTERN = re.compile(r"[\uac00-\ud7a3]")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObjectNaming(StrictModel):
    track_id: str
    canonical_object_name: str
    visible_attributes: dict[str, str]
    observable_state: str
    confidence: float = Field(ge=0, le=1)
    raw_response: str
    parse_error: str | None = None


class ObjectReview(StrictModel):
    track_id: str
    decision: Literal["approved", "modified", "rejected"]
    object_name: str
    visible_attributes: dict[str, str]
    observable_state: str
    reviewer_notes: str = ""

    @model_validator(mode="after")
    def validate_review(self) -> "ObjectReview":
        if self.decision != "rejected" and not self.object_name.strip():
            raise ValueError("Approved/modified objects require object_name")
        return self


class VerifiedEvent(StrictModel):
    event_id: str
    description: str
    event_type: Literal[
        "action", "spatial_change", "state_change", "completion", "restoration"
    ]
    participant_track_ids: list[str]
    before_timestamp_seconds: float = Field(ge=0)
    after_timestamp_seconds: float = Field(ge=0)
    reviewer_notes: str = ""

    @model_validator(mode="after")
    def chronological(self) -> "VerifiedEvent":
        if self.before_timestamp_seconds >= self.after_timestamp_seconds:
            raise ValueError("Verified event requires before < after")
        return self


class VerifiedNegativeFact(StrictModel):
    fact_id: str
    description: str
    supporting_track_ids: list[str]
    evidence_timestamps_seconds: list[float] = Field(min_length=1)
    reviewer_notes: str = ""


class HumanReview(StrictModel):
    video_id: str
    review_status: Literal["completed"]
    object_reviews: list[ObjectReview]
    verified_events: list[VerifiedEvent]
    verified_negative_facts: list[VerifiedNegativeFact]


class VerifiedObject(StrictModel):
    video_id: str
    track_id: str
    object_name: str
    visible_attributes: dict[str, str]
    observable_state: str
    bbox_xyxy: list[int]
    first_seen_seconds: float = Field(ge=0)
    last_seen_seconds: float = Field(ge=0)
    segmentation_model: str
    naming_model: str
    verification_status: Literal["approved", "modified"]
    evidence_frames: list[int]
    evidence_timestamps_seconds: list[float]


class VerifiedSpatialRelation(StrictModel):
    relation_id: str
    subject_track_id: str
    relation: Literal["left_of", "right_of", "above", "below"]
    reference_track_id: str
    evidence_timestamps_seconds: list[float] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    co_visible_frame_count: int = Field(ge=1)
    median_axis_separation_ratio: float = Field(ge=0)
    derivation: Literal["stable_bbox_geometry"] = "stable_bbox_geometry"
    verification_status: Literal["auto_derived", "approved", "modified"] = "auto_derived"


class VerifiedAnnotation(StrictModel):
    video_id: str
    objects: list[VerifiedObject]
    verified_events: list[VerifiedEvent]
    verified_negative_facts: list[VerifiedNegativeFact]
    verified_spatial_relations: list[VerifiedSpatialRelation] = Field(default_factory=list)


class VQAEvidence(StrictModel):
    track_ids: list[str]
    event_ids: list[str]
    negative_fact_ids: list[str]
    spatial_relation_ids: list[str]
    timestamps_seconds: list[float] = Field(min_length=1)
    rationale: str


class DraftVQAItem(StrictModel):
    item_id: str = Field(pattern=r"^VQA_\d{3,}$")
    category: Category
    question: str
    answer: str
    answer_type: Literal["yes_no", "short_text"]
    polarity: Literal["positive", "negative", "hard_negative", "not_applicable"]
    evidence: VQAEvidence
    confidence: float = Field(ge=0, le=1)
    draft: Literal[True] = True
    requires_final_review: Literal[True] = True

    @model_validator(mode="after")
    def no_metadata_in_qa(self) -> "DraftVQAItem":
        if METADATA_PATTERN.search(self.question) or METADATA_PATTERN.search(self.answer):
            raise ValueError("Question/answer must not expose evidence metadata")
        if HANGUL_PATTERN.search(self.question) or HANGUL_PATTERN.search(self.answer):
            raise ValueError("Final question and answer must be English")
        if self.polarity in {"negative", "hard_negative"} and not self.evidence.negative_fact_ids:
            raise ValueError("Negative VQA requires a human-verified negative fact")
        return self


class SkippedCategory(StrictModel):
    category: Category
    reason: str


class DraftVQABatch(StrictModel):
    video_id: str
    items: list[DraftVQAItem]
    skipped_categories: list[SkippedCategory]

    @model_validator(mode="after")
    def exact_category_coverage(self) -> "DraftVQABatch":
        generated = [item.category for item in self.items]
        skipped = [item.category for item in self.skipped_categories]
        generated_set, skipped_set = set(generated), set(skipped)
        if generated_set & skipped_set:
            raise ValueError("A category cannot be both generated and skipped")
        if generated_set | skipped_set != set(CATEGORY_ORDER):
            raise ValueError("Every category must have one or more items or be skipped")
        if len(set(skipped)) != len(skipped):
            raise ValueError("Duplicate skipped category")
        item_ids = [item.item_id for item in self.items]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("Duplicate VQA item_id")
        normalized_questions = [item.question.strip().casefold() for item in self.items]
        if len(set(normalized_questions)) != len(normalized_questions):
            raise ValueError("Duplicate VQA question")
        return self


class FinalVQAReviewItem(StrictModel):
    item_id: str
    category: Category
    decision: Literal["approved", "modified", "rejected"]
    question: str
    answer: str
    reviewer_notes: str = ""


class FinalVQAReview(StrictModel):
    video_id: str
    review_status: Literal["completed"]
    items: list[FinalVQAReviewItem]
