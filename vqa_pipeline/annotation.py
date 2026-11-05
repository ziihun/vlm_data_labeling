from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .spec_models import HumanReview, ObjectNaming, VerifiedAnnotation, VerifiedObject


def build_pending_annotation(video_id: str, tracks: dict[str, Any], crops: dict[str, Any], naming_payload: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    crop_by_id = {item["track_id"]: item for item in crops["tracks"]}
    naming_by_id = {item["track_id"]: ObjectNaming.model_validate(item) for item in naming_payload["items"]}
    objects = []
    for track in tracks["tracks"]:
        naming = naming_by_id[track["track_id"]]
        crop = crop_by_id[track["track_id"]]
        latest_bbox = next(item["bbox_xyxy"] for item in reversed(track["observations"]) if item["bbox_xyxy"])
        objects.append({
            "video_id": video_id,
            "track_id": track["track_id"],
            "object_name": naming.canonical_object_name,
            "visible_attributes": naming.visible_attributes,
            "observable_state": naming.observable_state,
            "bbox_xyxy": latest_bbox,
            "first_seen_seconds": track["first_seen_seconds"],
            "last_seen_seconds": track["last_seen_seconds"],
            "segmentation_model": tracks["model"],
            "naming_model": naming_payload["model"],
            "verification_status": "pending",
            "evidence_frames": [item["frame_index"] for item in crop["evidence"]],
            "evidence_timestamps_seconds": [item["source_timestamp_seconds"] for item in crop["evidence"]],
            "evidence": crop["evidence"],
            "naming_confidence": naming.confidence,
            "naming_parse_error": naming.parse_error,
            "raw_response": naming.raw_response,
        })
    payload = {"video_id": video_id, "language_policy": "English JSON and annotation values", "objects": objects}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "objects.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def apply_human_review(pending: dict[str, Any], review: HumanReview, output_dir: Path) -> VerifiedAnnotation:
    pending_by_id = {item["track_id"]: item for item in pending["objects"]}
    reviewed_ids = {item.track_id for item in review.object_reviews}
    if reviewed_ids != set(pending_by_id):
        raise ValueError(f"Human review must cover every Track exactly once: missing={set(pending_by_id)-reviewed_ids}, extra={reviewed_ids-set(pending_by_id)}")
    objects = []
    for item in review.object_reviews:
        if item.decision == "rejected":
            continue
        source = pending_by_id[item.track_id]
        objects.append(VerifiedObject(
            video_id=review.video_id,
            track_id=item.track_id,
            object_name=item.object_name,
            visible_attributes=item.visible_attributes,
            observable_state=item.observable_state,
            bbox_xyxy=source["bbox_xyxy"],
            first_seen_seconds=source["first_seen_seconds"],
            last_seen_seconds=source["last_seen_seconds"],
            segmentation_model=source["segmentation_model"],
            naming_model=source["naming_model"],
            verification_status=item.decision,
            evidence_frames=source["evidence_frames"],
            evidence_timestamps_seconds=source["evidence_timestamps_seconds"],
        ))
    valid_tracks = {item.track_id for item in objects}
    for event in review.verified_events:
        if not set(event.participant_track_ids).issubset(valid_tracks):
            raise ValueError(f"Verified event {event.event_id} references rejected/unknown Track")
    for fact in review.verified_negative_facts:
        if not set(fact.supporting_track_ids).issubset(valid_tracks):
            raise ValueError(f"Negative fact {fact.fact_id} references rejected/unknown Track")
    verified = VerifiedAnnotation(
        video_id=review.video_id,
        objects=objects,
        verified_events=review.verified_events,
        verified_negative_facts=review.verified_negative_facts,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "verified_annotations.json").write_text(verified.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return verified
