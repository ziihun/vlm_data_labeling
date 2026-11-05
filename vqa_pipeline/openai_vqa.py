from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from .spec_models import (
    CATEGORY_ORDER,
    DraftVQABatch,
    FinalVQAReview,
    VerifiedAnnotation,
)


PROMPT_TEMPLATE = """You are generating draft benchmark questions for industrial video understanding.

Use ONLY the supplied gated structured annotations. Every included object passed the naming confidence gate or human review; excluded Tracks are unavailable evidence.
For each category below, generate every distinct high-quality item that the supplied evidence supports. There is no fixed per-category quota or item-count cap. Stop only when another question would be a paraphrase, artificial variation, or unsupported by the evidence. Skip a category only when zero valid items can be grounded:
{categories}

Requirements:
1. Use verified visual facts only.
2. Do not invent objects, actions, or events.
3. Do not ask about timestamps, frame indices, Clip IDs, or Track IDs.
4. Store evidence timestamps only in evidence metadata.
5. Generate temporal/action/completion questions only when a matching verified event exists.
6. Generate negative questions only when a matching human-verified negative fact exists.
7. Skip categories with insufficient evidence.
8. Do not generate Risk Reasoning questions.
9. Write every question and answer in natural English. Use English JSON keys and category identifiers.
10. Set draft=true and requires_final_review=true for every generated item.
11. Prefer a clear, natural question whose answer is uniquely supported by the annotation. Do not merely restate the complete annotation.
12. For object/attribute questions, cite the supporting Track and use only its supplied name, attributes, and observable state.
13. Evidence timestamps must be copied from the supplied annotation; never invent or interpolate them.
14. If no category-valid evidence exists, return that category in skipped_categories with a concrete reason. Never force category coverage with hallucinated content.
15. Do not treat object presence, a static state, or a sequence of sampled timestamps as proof of an action or temporal change.
16. Assign every item a unique sequential item_id such as VQA_001, VQA_002, and so on.
17. Maximize the number and diversity of grounded items in every supported category. Systematically cover distinct objects, attributes, states, relations, and verified events. Never create paraphrase duplicates or split one fact into artificial wording variants.
18. A category listed in items must not appear in skipped_categories. A category with no items must appear exactly once in skipped_categories.
19. Spatial position questions are allowed only from verified_spatial_relations. Cite its relation_id in spatial_relation_ids and both participating Track IDs in track_ids.
20. Interpret left/right/above/below from the image viewer's perspective. Do not convert it to a person's own left/right.
21. For scene.action_spatial_understanding and alignment.action_relation_alignment, a verified spatial relation is valid evidence even when no action event exists.

Gated structured annotation sent to the API:
{annotation}
"""


class OpenAIVQAGenerator:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def run(self, annotation: VerifiedAnnotation, output_dir: Path) -> DraftVQABatch:
        prompt = PROMPT_TEMPLATE.format(
            categories=json.dumps(CATEGORY_ORDER, ensure_ascii=False),
            annotation=annotation.model_dump_json(),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "api_input_annotation.json").write_text(
            annotation.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        response = self.client.responses.parse(
            model=self.model,
            instructions="Follow the benchmark specification exactly and return schema-valid JSON.",
            input=[{"role": "user", "content": prompt}],
            text_format=DraftVQABatch,
            max_output_tokens=20000,
            store=False,
        )
        (output_dir / "raw_response.json").write_text(response.model_dump_json(indent=2) + "\n", encoding="utf-8")
        result = response.output_parsed
        if result is None:
            raise RuntimeError("OpenAI returned no parsed VQA output")
        self.validate(result, annotation)
        (output_dir / "generated_vqa.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return result

    @staticmethod
    def validate(result: DraftVQABatch, annotation: VerifiedAnnotation) -> None:
        if result.video_id != annotation.video_id:
            raise ValueError("OpenAI VQA video_id does not match the source annotation")
        tracks = {item.track_id for item in annotation.objects}
        events = {item.event_id: item for item in annotation.verified_events}
        negative_facts = {item.fact_id for item in annotation.verified_negative_facts}
        spatial_relations = {
            item.relation_id: item for item in annotation.verified_spatial_relations
        }
        temporal_categories = {
            "scene.temporal_understanding", "scene.task_state_completion_understanding",
            "alignment.temporal_alignment",
        }
        for item in result.items:
            if not set(item.evidence.track_ids).issubset(tracks):
                raise ValueError(f"{item.category} references an unverified Track")
            if not set(item.evidence.event_ids).issubset(events):
                raise ValueError(f"{item.category} references an unverified event")
            if not set(item.evidence.negative_fact_ids).issubset(negative_facts):
                raise ValueError(f"{item.category} references an unverified negative fact")
            if not set(item.evidence.spatial_relation_ids).issubset(spatial_relations):
                raise ValueError(f"{item.category} references an unverified spatial relation")
            if item.category in temporal_categories and not item.evidence.event_ids:
                raise ValueError(f"{item.category} requires a human-verified event")
            if item.category in {
                "scene.action_spatial_understanding", "alignment.action_relation_alignment"
            } and not (item.evidence.event_ids or item.evidence.spatial_relation_ids):
                raise ValueError(f"{item.category} requires a verified event or spatial relation")
            if item.category == "scene.task_state_completion_understanding" and any(
                events[event_id].event_type not in {"completion", "restoration"}
                for event_id in item.evidence.event_ids
            ):
                raise ValueError("Task completion VQA requires completion/restoration event")
            allowed_timestamps: set[float] = set()
            for obj in annotation.objects:
                if obj.track_id in item.evidence.track_ids:
                    allowed_timestamps.update(round(value, 6) for value in obj.evidence_timestamps_seconds)
            for event_id in item.evidence.event_ids:
                event = events[event_id]
                allowed_timestamps.update({round(event.before_timestamp_seconds, 6), round(event.after_timestamp_seconds, 6)})
            for fact in annotation.verified_negative_facts:
                if fact.fact_id in item.evidence.negative_fact_ids:
                    allowed_timestamps.update(round(value, 6) for value in fact.evidence_timestamps_seconds)
            for relation_id in item.evidence.spatial_relation_ids:
                relation = spatial_relations[relation_id]
                if not {
                    relation.subject_track_id, relation.reference_track_id
                }.issubset(item.evidence.track_ids):
                    raise ValueError(f"{item.category} must cite both Tracks in {relation_id}")
                allowed_timestamps.update(
                    round(value, 6) for value in relation.evidence_timestamps_seconds
                )
            supplied = {round(value, 6) for value in item.evidence.timestamps_seconds}
            if not supplied.issubset(allowed_timestamps):
                raise ValueError(f"{item.category} contains invented evidence timestamps")


def apply_final_review(draft: DraftVQABatch, review: FinalVQAReview, output_dir: Path) -> dict:
    draft_by_id = {item.item_id: item for item in draft.items}
    review_ids = {item.item_id for item in review.items}
    if review_ids != set(draft_by_id):
        raise ValueError("Final review must cover every generated VQA exactly once")
    verified = []
    for item in review.items:
        if item.decision == "rejected":
            continue
        source = draft_by_id[item.item_id]
        if source.category != item.category:
            raise ValueError(f"Final review category mismatch for {item.item_id}")
        verified.append({
            **source.model_dump(mode="json"),
            "question": item.question,
            "answer": item.answer,
            "draft": False,
            "requires_final_review": False,
            "verification_status": item.decision,
            "reviewer_notes": item.reviewer_notes,
        })
    payload = {"video_id": review.video_id, "benchmark_ready": True, "items": verified}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "verified_vqa.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
