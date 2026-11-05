from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .openai_vqa import OpenAIVQAGenerator
from .report import StageReport
from .spatial_relations import StableBBoxSpatialRelationBuilder
from .vqa_mapping import DEFAULT_THRESHOLD, promote_qwen9, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finalize(
    output_root: Path,
    source_naming: Path,
    model: str,
    confidence_threshold: float = DEFAULT_THRESHOLD,
    json_only: bool = False,
    export_json: Path | None = None,
    cleanup_intermediate: bool = False,
) -> dict[str, Any]:
    preprocessing = _load_json(output_root / "00_preprocessing" / "manifest.json")
    frame_info = preprocessing["video_info"]
    promoted = promote_qwen9(output_root, source_naming, confidence_threshold)
    annotation = promoted["verified"]
    if not annotation.objects:
        raise RuntimeError("No naming Tracks passed the VQA evidence gate")
    annotation = StableBBoxSpatialRelationBuilder().build(
        annotation,
        _load_json(output_root / "01_sam2" / "tracks.json"),
        output_root / "04_annotation" / "spatial_relations.json",
        frame_size=(int(frame_info["width"]), int(frame_info["height"])),
    )
    (output_root / "05_human_review" / "auto_verified_annotations.json").write_text(
        annotation.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    vqa_dir = output_root / "06_vqa"
    draft = OpenAIVQAGenerator(
        OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=300.0), model
    ).run(annotation, vqa_dir)

    manifest_path = output_root / "manifest.json"
    manifest = _load_json(manifest_path)
    manifest.update({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "SAM2 -> process exit -> Qwen3.5-9B detailed naming -> process exit -> annotation gate -> OpenAI VQA -> human verification",
        "naming_model": promoted["gate"]["selected_naming_model"],
        "openai_model": model,
        "vqa_language": "English",
        "vqa_generation_policy": "all distinct grounded items; no per-category count cap",
    })
    manifest.setdefault("language_policy", {})["final_question_answer"] = "English"
    stages = manifest["stages"]
    stages["03_qwen_naming"] = {
        "status": "completed",
        "detail": f"Detailed naming completed for {len(promoted['pending']['objects'])} Tracks",
        "model": promoted["gate"]["selected_naming_model"],
        "artifacts": ["03_qwen_naming/prompt.txt", "03_qwen_naming/raw_response.json", "03_qwen_naming/naming.json"],
    }
    stages["04_annotation"] = {
        "status": "completed",
        "detail": f"{promoted['gate']['auto_accepted_count']} Tracks passed the {confidence_threshold:g} confidence gate; {len(annotation.verified_spatial_relations)} stable spatial relations derived; {promoted['gate']['needs_human_review_count']} Tracks excluded pending review",
        "artifacts": ["04_annotation/objects.json", "04_annotation/spatial_relations.json"],
    }
    stages["05_human_review"] = {
        "status": "pending_review",
        "detail": f"Review {promoted['gate']['needs_human_review_count']} blocked naming Tracks and all auto-accepted names",
        "artifacts": ["05_human_review/naming_gate.json", "05_human_review/auto_verified_annotations.json"],
    }
    stages["06_vqa"] = {
        "status": "completed",
        "detail": f"OpenAI generated {len(draft.items)} draft VQA items and skipped {len(draft.skipped_categories)} unsupported categories",
        "model": model,
        "artifacts": ["06_vqa/api_input_annotation.json", "06_vqa/prompt.txt", "06_vqa/raw_response.json", "06_vqa/generated_vqa.json"],
    }
    stages["07_final"] = {
        "status": "pending_review",
        "detail": "Review every naming result and draft Q&A; benchmark_ready remains false",
        "artifact": None if json_only else "final_review.html",
    }
    write_json(manifest_path, manifest)

    final_report: Path | None = None
    if not json_only:
        clip_tracks = [
            _load_json(output_root / "01_sam2" / f"clip_{int(clip['clip_index']):03d}" / "tracks.json")
            for clip in preprocessing["clips"]
        ]
        report_path = StageReport(output_root).write(
            manifest, preprocessing, clip_tracks, promoted["pending"], draft.model_dump(mode="json"),
            [item.model_dump(mode="json") for item in annotation.verified_spatial_relations],
        )
        final_report = output_root / "final_review.html"
        shutil.copy2(report_path, final_report)

    summary = {
        "video_id": annotation.video_id,
        "naming_model": promoted["gate"]["selected_naming_model"],
        "openai_model": model,
        "total_tracks": len(promoted["pending"]["objects"]),
        "vqa_eligible_tracks": promoted["gate"]["auto_accepted_count"],
        "naming_review_required_tracks": [
            item["track_id"] for item in promoted["gate"]["items"]
            if item["status"] == "needs_human_review"
        ],
        "spatial_relation_count": len(annotation.verified_spatial_relations),
        "generated_vqa_count": len(draft.items),
        "skipped_category_count": len(draft.skipped_categories),
        "benchmark_ready": False,
        "final_report": str(final_report.resolve()) if final_report else None,
    }
    if export_json is not None:
        export_json = export_json.resolve()
        if export_json.is_relative_to(output_root.resolve()):
            raise ValueError("--export-json must be outside --output-root")
        raw_response = _load_json(vqa_dir / "raw_response.json")
        final_payload = {
            "schema_version": "1.0.0-vqa-json-only",
            "video_id": annotation.video_id,
            "source_video": preprocessing["source_video"],
            "segmentation_model": manifest.get("segmentation_model"),
            "naming_model": promoted["gate"]["selected_naming_model"],
            "vqa_model": model,
            "benchmark_ready": False,
            "requires_human_verification": True,
            "usage": raw_response.get("usage"),
            **draft.model_dump(mode="json"),
        }
        write_json(export_json, final_payload)
        summary["final_json"] = str(export_json)
    write_json(output_root / "final_run_summary.json", summary)
    if cleanup_intermediate:
        if export_json is None:
            raise ValueError("--cleanup-intermediate requires --export-json")
        if output_root.resolve() in {PROJECT_ROOT.resolve(), PROJECT_ROOT.parent.resolve()}:
            raise ValueError("Refusing to clean a project root")
        shutil.rmtree(output_root)
    return summary


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    parser = argparse.ArgumentParser(description="Run OpenAI VQA from gated detailed Qwen naming and render final review HTML")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-naming", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-6.1-terra"))
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--export-json", type=Path)
    parser.add_argument("--cleanup-intermediate", action="store_true")
    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from .env")
    summary = finalize(
        output_root=args.output_root.resolve(),
        source_naming=args.source_naming.resolve(),
        model=args.model,
        confidence_threshold=args.confidence_threshold,
        json_only=args.json_only,
        export_json=args.export_json,
        cleanup_intermediate=args.cleanup_intermediate,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
