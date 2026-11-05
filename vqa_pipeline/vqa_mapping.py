from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any

from .annotation import build_pending_annotation
from .spec_models import CATEGORY_ORDER, DraftVQABatch, DraftVQAItem, SkippedCategory, VQAEvidence, VerifiedAnnotation, VerifiedObject


DEFAULT_THRESHOLD = 0.7


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_experiment_naming(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert benchmark-runner output to the strict pipeline naming schema."""
    items = []
    for source in payload["items"]:
        items.append({
            "track_id": source["track_id"],
            "canonical_object_name": source["canonical_object_name"],
            "visible_attributes": source.get("visible_attributes", {}),
            "observable_state": source.get("observable_state", "unknown"),
            "confidence": source["confidence"],
            "raw_response": source.get("raw_response") or json.dumps(source, ensure_ascii=False),
            "parse_error": source.get("parse_error"),
        })
    return {
        "model": payload["model"],
        "runner": payload.get("runner", "naming_experiment"),
        "input_policy": payload.get("input_policy", "masked crops and context images"),
        "items": items,
    }


def promote_qwen9(output_root: Path, source_naming: Path, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    tracks = json.loads((output_root / "01_sam2" / "tracks.json").read_text(encoding="utf-8"))
    crops = json.loads((output_root / "02_crops" / "crops.json").read_text(encoding="utf-8"))
    manifest_path = output_root / "manifest.json"
    video_id = (
        json.loads(manifest_path.read_text(encoding="utf-8"))["video_id"]
        if manifest_path.exists() else output_root.name.replace("_sam2", "")
    )
    experiment_naming = json.loads(source_naming.read_text(encoding="utf-8"))
    naming = normalize_experiment_naming(experiment_naming)
    expected_ids = {item["track_id"] for item in tracks["tracks"]}
    actual_ids = {item["track_id"] for item in naming["items"]}
    if actual_ids != expected_ids:
        raise ValueError(f"Qwen9 naming Track coverage mismatch: missing={expected_ids-actual_ids}, extra={actual_ids-expected_ids}")

    stage03 = output_root / "03_qwen_naming"
    current_path = stage03 / "naming.json"
    if current_path.exists():
        current = json.loads(current_path.read_text(encoding="utf-8"))
        if current.get("model") != naming.get("model"):
            archive = stage03 / "qwen3_5_4b_baseline.json"
            if not archive.exists():
                shutil.copy2(current_path, archive)
    stage03.mkdir(parents=True, exist_ok=True)
    write_json(stage03 / "qwen3_5_9b_experiment.json", experiment_naming)
    write_json(current_path, naming)
    write_json(stage03 / "selected_naming.json", naming)
    source_prompt = source_naming.parent / "prompt.txt"
    source_raw = source_naming.parent / "raw_response.json"
    if source_prompt.exists():
        shutil.copy2(source_prompt, stage03 / "prompt.txt")
    if source_raw.exists():
        shutil.copy2(source_raw, stage03 / "raw_response.json")

    stage04 = output_root / "04_annotation"
    old_objects = stage04 / "objects.json"
    if old_objects.exists():
        old_payload = json.loads(old_objects.read_text(encoding="utf-8"))
        if old_payload.get("objects") and old_payload["objects"][0].get("naming_model") != naming["model"]:
            archive = stage04 / "objects_qwen3_5_4b_baseline.json"
            if not archive.exists():
                shutil.copy2(old_objects, archive)
    pending = build_pending_annotation(video_id, tracks, crops, naming, stage04)

    naming_by_id = {item["track_id"]: item for item in naming["items"]}
    decisions = []
    accepted_objects = []
    for obj in pending["objects"]:
        item = naming_by_id[obj["track_id"]]
        name = str(item["canonical_object_name"]).strip()
        confidence = float(item["confidence"])
        needs_review = name.lower() == "unknown" or confidence < threshold or bool(item.get("parse_error"))
        status = "needs_human_review" if needs_review else "auto_accepted"
        reasons = []
        if name.lower() == "unknown":
            reasons.append("object_name_is_unknown")
        if confidence < threshold:
            reasons.append(f"confidence_below_{threshold:g}")
        if item.get("parse_error"):
            reasons.append("parse_error")
        decisions.append({
            "track_id": obj["track_id"],
            "object_name": name,
            "confidence": confidence,
            "status": status,
            "reasons": reasons,
        })
        obj["verification_status"] = status
        if not needs_review:
            accepted_objects.append(VerifiedObject(
                video_id=pending["video_id"],
                track_id=obj["track_id"],
                object_name=obj["object_name"],
                visible_attributes=obj["visible_attributes"],
                observable_state=obj["observable_state"],
                bbox_xyxy=obj["bbox_xyxy"],
                first_seen_seconds=obj["first_seen_seconds"],
                last_seen_seconds=obj["last_seen_seconds"],
                segmentation_model=obj["segmentation_model"],
                naming_model=obj["naming_model"],
                verification_status="approved",
                evidence_frames=obj["evidence_frames"],
                evidence_timestamps_seconds=obj["evidence_timestamps_seconds"],
            ))
    write_json(stage04 / "objects.json", pending)
    gate = {
        "video_id": pending["video_id"],
        "selected_naming_model": naming["model"],
        "policy": {"auto_accept": f"name != unknown and confidence >= {threshold:g} and no parse_error", "mandatory_review": f"name == unknown or confidence < {threshold:g} or parse_error"},
        "auto_accepted_count": sum(item["status"] == "auto_accepted" for item in decisions),
        "needs_human_review_count": sum(item["status"] == "needs_human_review" for item in decisions),
        "items": decisions,
    }
    write_json(output_root / "05_human_review" / "naming_gate.json", gate)
    auto_verified = VerifiedAnnotation(
        video_id=pending["video_id"], objects=accepted_objects,
        verified_events=[], verified_negative_facts=[],
    )
    (output_root / "05_human_review" / "auto_verified_annotations.json").write_text(
        auto_verified.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return {"pending": pending, "gate": gate, "verified": auto_verified}


def build_vqa_mapping(output_root: Path, promoted: dict[str, Any]) -> DraftVQABatch:
    verified: VerifiedAnnotation = promoted["verified"]
    by_track = {item.track_id: item for item in verified.objects}
    items: list[DraftVQAItem] = []
    sign = by_track.get("G0002")
    if sign and sign.visible_attributes.get("border", "").lower() == "red":
        timestamps = sign.evidence_timestamps_seconds
        items.extend([
            DraftVQAItem(
                item_id="VQA_001",
                category="scene.object_attribute_recognition",
                question="What color is the border of the white rectangular sign?",
                answer="Red.",
                answer_type="short_text",
                polarity="not_applicable",
                evidence=VQAEvidence(
                    track_ids=[sign.track_id], event_ids=[], negative_fact_ids=[], spatial_relation_ids=[],
                    timestamps_seconds=timestamps,
                    rationale="The auto-accepted Qwen3.5-9B annotation identifies a white rectangular sign with a red border.",
                ),
                confidence=sign.visible_attributes.get("border", "").lower() == "red" and 0.95 or 0.7,
                draft=True, requires_final_review=True,
            ),
            DraftVQAItem(
                item_id="VQA_002",
                category="alignment.object_attribute_alignment",
                question="Is a white rectangular sign visible in the video?",
                answer="Yes.",
                answer_type="yes_no",
                polarity="positive",
                evidence=VQAEvidence(
                    track_ids=[sign.track_id], event_ids=[], negative_fact_ids=[], spatial_relation_ids=[],
                    timestamps_seconds=timestamps,
                    rationale="The auto-accepted object Track is consistently named sign and has white, rectangular attributes.",
                ),
                confidence=0.95,
                draft=True, requires_final_review=True,
            ),
        ])
    generated_categories = {item.category for item in items}
    skipped = []
    for category in CATEGORY_ORDER:
        if category in generated_categories:
            continue
        if category in {
            "scene.action_spatial_understanding", "scene.temporal_understanding",
            "scene.task_state_completion_understanding", "alignment.action_relation_alignment",
            "alignment.temporal_alignment",
        }:
            reason = "No human-verified event or before/after state change is available."
        else:
            reason = "No auto-accepted annotation provides sufficient grounded evidence."
        skipped.append(SkippedCategory(category=category, reason=reason))
    batch = DraftVQABatch(video_id=verified.video_id, items=items, skipped_categories=skipped)
    stage06 = output_root / "06_vqa"
    stage06.mkdir(parents=True, exist_ok=True)
    payload = {
        **batch.model_dump(mode="json"),
        "benchmark_ready": False,
        "mapping_policy": "Only auto-accepted Qwen3.5-9B objects are eligible. Temporal/action categories require human-verified events.",
        "blocked_track_ids": [item["track_id"] for item in promoted["gate"]["items"] if item["status"] == "needs_human_review"],
    }
    write_json(stage06 / "vqa_mapping_preview.json", payload)
    render_mapping_report(output_root, promoted, payload)
    return batch


def render_mapping_report(output_root: Path, promoted: dict[str, Any], mapping: dict[str, Any]) -> None:
    overlay_videos = sorted((output_root / "01_sam2").glob("clip_*/tracking_overlay.mp4"))
    overlay_cards = "".join(
        f"<article><h3>{html.escape(path.parent.name)}</h3><video controls preload='metadata' src='{html.escape(path.relative_to(output_root).as_posix())}'></video></article>"
        for path in overlay_videos
    )
    gate_rows = "".join(
        f"<tr><td>{html.escape(item['track_id'])}</td><td>{html.escape(item['object_name'])}</td><td>{item['confidence']}</td><td class='{item['status']}'>{item['status']}</td><td>{html.escape(', '.join(item['reasons']) or '—')}</td></tr>"
        for item in promoted["gate"]["items"]
    )
    item_rows = "".join(
        f"<tr><td>{html.escape(item['category'])}</td><td>{html.escape(item['question'])}</td><td>{html.escape(item['answer'])}</td><td>{html.escape(', '.join(item['evidence']['track_ids']))}</td><td>{html.escape(str(item['evidence']['timestamps_seconds']))}</td></tr>"
        for item in mapping["items"]
    )
    skipped_rows = "".join(
        f"<tr><td>{html.escape(item['category'])}</td><td>{html.escape(item['reason'])}</td></tr>"
        for item in mapping["skipped_categories"]
    )
    document = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>SAM2 → Qwen9 → VQA Final Mapping Review</title><style>body{{font-family:system-ui;background:#10151b;color:#eef3f8;margin:0}}main{{max-width:1250px;margin:auto;padding:28px}}section{{background:#1a222c;border:1px solid #344252;border-radius:12px;padding:18px;margin:18px 0}}table{{width:100%;border-collapse:collapse}}th,td{{border-bottom:1px solid #344252;text-align:left;padding:9px;vertical-align:top}}a{{color:#7dd3fc}}video{{width:100%;max-height:520px;background:#000;border-radius:8px}}.videos{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}}.auto_accepted{{color:#86efac}}.needs_human_review{{color:#fca5a5}}code{{background:#0d1117;padding:2px 5px;border-radius:4px}}</style></head><body><main>
<h1>Qwen3.5-9B → Structured Annotation → VQA Mapping</h1><p>이 파일은 mapping 검증용 draft다. 질문/정답 최종 검수 전에는 Benchmark가 아니다.</p>
<section><h2>프로세스 경계</h2><p><code>SAM2 종료</code> → <code>Qwen3.5-9B Naming 종료</code> → <code>Annotation Gate</code> → <code>VQA Mapping 별도 실행</code></p></section>
<section><h2>1. 원본 영상</h2><video controls preload='metadata' src='source_video.mp4'></video></section>
<section><h2>2. SAM2 인식 결과 — BBox / Mask Overlay</h2><div class='videos'>{overlay_cards}</div><p><a href='report.html'>전체 Track과 Crop 상세 보고서 열기</a></p></section>
<section><h2>3. Qwen3.5-9B Naming Gate</h2><p>자동 통과 {promoted['gate']['auto_accepted_count']} · 사람 검수 필요 {promoted['gate']['needs_human_review_count']}</p><table><thead><tr><th>Track</th><th>Name</th><th>Confidence</th><th>Status</th><th>Reason</th></tr></thead><tbody>{gate_rows}</tbody></table></section>
<section><h2>4. 생성 가능한 Draft VQA Mapping</h2><table><thead><tr><th>Category</th><th>Question</th><th>Answer</th><th>Evidence Track</th><th>Evidence timestamps</th></tr></thead><tbody>{item_rows}</tbody></table></section>
<section><h2>5. Skip된 카테고리</h2><table><thead><tr><th>Category</th><th>Reason</th></tr></thead><tbody>{skipped_rows}</tbody></table></section>
<p><a href='06_vqa/vqa_mapping_preview.json'>vqa_mapping_preview.json</a> · <a href='05_human_review/naming_gate.json'>naming_gate.json</a> · <a href='04_annotation/objects.json'>objects.json</a></p>
</main></body></html>"""
    (output_root / "vqa_mapping_report.html").write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote Qwen3.5-9B naming and build a grounded VQA mapping preview")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-naming", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    promoted = promote_qwen9(args.output_root, args.source_naming, args.confidence_threshold)
    batch = build_vqa_mapping(args.output_root, promoted)
    print(json.dumps({
        "selected_model": promoted["gate"]["selected_naming_model"],
        "auto_accepted": promoted["gate"]["auto_accepted_count"],
        "needs_human_review": promoted["gate"]["needs_human_review_count"],
        "draft_vqa_items": len(batch.items),
        "skipped_categories": len(batch.skipped_categories),
        "report": str((args.output_root / "vqa_mapping_report.html").resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
