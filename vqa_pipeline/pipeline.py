from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AuthenticationError, PermissionDeniedError

from .analyzer import OpenAIVQAAnalyzer
from .artifacts import ArtifactWriter
from .config import PilotConfig
from .dataset import DatasetCatalog
from .models import CATEGORY_ORDER
from .video import VideoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "raw_datasets" / "Safe_and_Unsafe_Behaviours"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "vqa_datasets" / "pilot_5"


class VQAPilotPipeline:
    """Coordinates dataset selection, video processing, analysis, and artifacts."""

    def __init__(
        self,
        config: PilotConfig,
        catalog: DatasetCatalog,
        video_processor: VideoProcessor,
        artifact_writer: ArtifactWriter,
        analyzer: OpenAIVQAAnalyzer | None,
    ) -> None:
        self.config = config
        self.catalog = catalog
        self.video_processor = video_processor
        self.artifact_writer = artifact_writer
        self.analyzer = analyzer

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _build_manifest(self, selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "schema_version": "0.2.0-pilot",
            "created_at": self._utc_now(),
            "status": "running",
            "scope": {
                "included": ["Scene Understanding", "Query-Scene Alignment"],
                "excluded": ["Risk Reasoning"],
                "categories": list(CATEGORY_ORDER),
            },
            "architecture": [
                "DatasetCatalog",
                "VideoProcessor",
                "OpenAIVQAAnalyzer",
                "ArtifactWriter",
                "VQAPilotPipeline",
            ],
            "pipeline": {
                "mode": "sampled-frame OpenAI VLM pilot + full-motion review",
                "model": self.config.model,
                "frame_count": self.config.frames_per_video,
                "image_detail": self.config.image_detail,
                "max_frame_width": self.config.max_frame_width,
                "limitations": [
                    "No SAM 2 masks or tracking in this pilot",
                    "No DAM-3B-Video descriptions in this pilot",
                    "No SAM 3 re-detection verification in this pilot",
                    "Sparse model-input frames can miss short events; review full MP4 videos",
                ],
            },
            "selected_videos": [
                {
                    "filepath": sample["filepath"],
                    "dataset_label": sample.get("ground_truth", {}).get("label"),
                    "tags": sample.get("tags", []),
                }
                for sample in selected
            ],
            "prepared_videos": 0,
            "completed_videos": 0,
            "failed_videos": 0,
            "errors": [],
        }

    @staticmethod
    def build_summary(
        requested: int,
        completed: int,
        failed: int,
        records: list[dict[str, Any]],
        category_counts: dict[str, int],
    ) -> dict[str, Any]:
        attempted = completed + failed
        return {
            "videos_requested": requested,
            "videos_attempted": attempted,
            "videos_completed": completed,
            "videos_failed": failed,
            "videos_unattempted": requested - attempted,
            "vqa_items": len(records),
            "category_counts": category_counts,
            "all_categories_complete": (
                completed > 0
                and all(
                    category_counts.get(category, 0) == completed
                    for category in CATEGORY_ORDER
                )
            ),
            "human_review_required": sum(
                1 for record in records if record["requires_human_review"]
            ),
            "review_page": "review.html",
        }

    def run(self) -> int:
        selected = self.catalog.select(self.config.num_videos)
        manifest = self._build_manifest(selected)
        self.artifact_writer.write_json("manifest.json", manifest)
        wrappers: list[dict[str, Any]] = []
        blocking_status: str | None = None

        for index, sample in enumerate(selected):
            relative_path = Path(sample["filepath"])
            video_path = self.config.dataset_root / relative_path
            video_id = video_path.stem
            print(
                f"[{index + 1}/{len(selected)}] {video_id}: frames + review video",
                flush=True,
            )
            try:
                processed = self.video_processor.process(
                    video_path=video_path,
                    output_root=self.config.output_root,
                    video_id=video_id,
                )
                manifest["prepared_videos"] += 1
                if self.config.prepare_only:
                    continue

                if self.analyzer is None:
                    raise RuntimeError("OpenAIVQAAnalyzer is required for API generation")
                print(
                    f"[{index + 1}/{len(selected)}] {video_id}: calling {self.config.model}",
                    flush=True,
                )
                result, response_meta = self.analyzer.analyze(
                    video_id=video_id,
                    duration=processed.duration_seconds,
                    frames=processed.frames,
                    pilot_index=index,
                )
                annotation_relative = Path("annotations") / f"{video_id}.json"
                wrapper = {
                    "schema_version": "0.2.0-pilot",
                    "source": {
                        "video_id": video_id,
                        "video_path": str(video_path.relative_to(self.config.project_root)),
                        "review_video_path": str(
                            processed.review_video_path.relative_to(self.config.output_root)
                        ),
                        "dataset_split": sample.get("tags", [None])[0],
                        "dataset_label": sample.get("ground_truth", {}).get("label"),
                        "duration_seconds": processed.duration_seconds,
                        "sampled_frames": [
                            {
                                "index": frame.index,
                                "timestamp_seconds": frame.timestamp_seconds,
                                "path": str(frame.path.relative_to(self.config.output_root)),
                            }
                            for frame in processed.frames
                        ],
                    },
                    "generation": {
                        **response_meta,
                        "generated_at": self._utc_now(),
                        "annotation_method": "sampled_frame_openai_vlm_pilot",
                    },
                    "result": result.model_dump(mode="json"),
                    "annotation_path": str(annotation_relative),
                }
                self.artifact_writer.write_annotation(video_id, wrapper)
                wrappers.append(wrapper)
                manifest["completed_videos"] += 1
                print(f"[{index + 1}/{len(selected)}] {video_id}: generated 7 VQA items")
            except Exception as exc:
                manifest["failed_videos"] += 1
                is_auth_error = isinstance(exc, AuthenticationError)
                is_permission_error = isinstance(exc, PermissionDeniedError)
                if is_auth_error:
                    safe_message = (
                        "OpenAI authentication failed (HTTP 401). "
                        "Replace OPENAI_API_KEY in .env."
                    )
                    blocking_status = "blocked_authentication"
                elif is_permission_error:
                    safe_message = (
                        f"OpenAI project has no access to model {self.config.model} "
                        "(HTTP 403). Select an accessible model."
                    )
                    blocking_status = "blocked_model_access"
                else:
                    safe_message = str(exc)
                manifest["errors"].append(
                    {
                        "video_id": video_id,
                        "error_type": type(exc).__name__,
                        "message": safe_message,
                    }
                )
                print(
                    f"[{index + 1}/{len(selected)}] {video_id}: ERROR: {safe_message}",
                    file=sys.stderr,
                )
                if blocking_status:
                    manifest["status"] = blocking_status
            finally:
                self.artifact_writer.write_json("manifest.json", manifest)
            if blocking_status:
                break

        if self.config.prepare_only:
            manifest["status"] = "prepared"
            manifest["finished_at"] = self._utc_now()
            self.artifact_writer.write_json("manifest.json", manifest)
            print(f"Prepared frames and review videos in {self.config.output_root}")
            return 0

        records, category_counts = self.artifact_writer.finalize(wrappers)
        summary = self.build_summary(
            requested=self.config.num_videos,
            completed=manifest["completed_videos"],
            failed=manifest["failed_videos"],
            records=records,
            category_counts=category_counts,
        )
        self.artifact_writer.write_json("summary.json", summary)
        if blocking_status:
            manifest["status"] = blocking_status
        else:
            manifest["status"] = (
                "completed" if manifest["failed_videos"] == 0 else "partial"
            )
        manifest["finished_at"] = self._utc_now()
        self.artifact_writer.write_json("manifest.json", manifest)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if manifest["failed_videos"] == 0 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a 5-video industrial VQA pilot")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--max-frame-width", type=int, default=768)
    parser.add_argument("--image-detail", choices=("low", "auto", "high"), default="auto")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    # This project-local key must win over a stale key exported by the parent shell.
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not args.prepare_only and not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env")
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-6.1-terra")
    config = PilotConfig(
        project_root=PROJECT_ROOT,
        dataset_root=args.dataset_root.resolve(),
        output_root=args.output_dir.resolve(),
        model=model,
        num_videos=args.num_videos,
        frames_per_video=args.frames_per_video,
        max_frame_width=args.max_frame_width,
        image_detail=args.image_detail,
        prepare_only=args.prepare_only,
    )
    pipeline = VQAPilotPipeline(
        config=config,
        catalog=DatasetCatalog(config.dataset_root),
        video_processor=VideoProcessor(
            frame_count=config.frames_per_video,
            max_width=config.max_frame_width,
        ),
        artifact_writer=ArtifactWriter(config.output_root),
        analyzer=(
            None
            if config.prepare_only
            else OpenAIVQAAnalyzer(
                api_key=api_key,
                model=config.model,
                image_detail=config.image_detail,
            )
        ),
    )
    return pipeline.run()


if __name__ == "__main__":
    raise SystemExit(main())
