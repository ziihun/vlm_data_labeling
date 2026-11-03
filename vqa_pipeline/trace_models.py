from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TraceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NormalizedBox(TraceModel):
    x_min: int = Field(ge=0, le=1000)
    y_min: int = Field(ge=0, le=1000)
    x_max: int = Field(ge=0, le=1000)
    y_max: int = Field(ge=0, le=1000)

    @model_validator(mode="after")
    def validate_order(self) -> "NormalizedBox":
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("box minimum coordinates must be less than maximum coordinates")
        return self


class RegionProposal(TraceModel):
    target_phrase_korean: str
    target_phrase_english: str
    bbox_normalized_0_1000: NormalizedBox
    visible_cues: list[str]
    why_this_region_answers_the_question: str
    localization_confidence: float = Field(ge=0, le=1)


class RoiDescription(TraceModel):
    detailed_description_korean: str
    likely_object_name_korean: str
    likely_object_name_english: str
    visible_attributes: list[str]
    visible_state_or_action: list[str]
    uncertainties: list[str]


class NormalizedObject(TraceModel):
    canonical_name_korean: str
    canonical_name_english: str
    attributes: list[str]
    state: list[str]
    evidence_summary_korean: str
    confidence: float = Field(ge=0, le=1)


class MvpVQA(TraceModel):
    category: str
    question: str
    answer: str
    evidence_description: str
    confidence: float = Field(ge=0, le=1)
    requires_human_review: bool
