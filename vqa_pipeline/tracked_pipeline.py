from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .track_merge import OverlapTrackMerger
from .tracked_analyzer import (
    DAMTrackDescriber,
    OpenAITrackedAnalyzer,
    Sam3NameVerifier,
)
from .tracking_models import TrackDescription, TrackedAnnotation
from .video_clips import OverlappingClipPreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = (
    PROJECT_ROOT
    / "raw_datasets"
    / "Safe_and_Unsafe_Behaviours"
    / "data"
    / "0_te1.mp4"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "vqa_datasets" / "tracked_mvp" / "0_te1"


@dataclass(frozen=True, slots=True)
class TrackedPipelineConfig:
    video_path: Path
    output_root: Path
    model: str
    sam2_python: Path
    clip_seconds: float = 10.0
    stride_seconds: float = 5.0
    tracking_fps: float = 5.0
    strict_models: bool = False
    strict_verification: bool = False
    dam_base_url: str | None = None
    sam3_base_url: str | None = None


class Sam2TrackingRunner:
    def __init__(self, python_path: Path, worker_path: Path) -> None:
        self.python_path = python_path
        self.worker_path = worker_path

    def run(self, manifest_path: Path, clip_index: int, output_dir: Path) -> dict[str, Any]:
        result_path = output_dir / "tracking.json"
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            }
        )
        command = [
            str(self.python_path),
            str(self.worker_path),
            "--preprocessing-manifest",
            str(manifest_path),
            "--clip-index",
            str(clip_index),
            "--output-dir",
            str(output_dir),
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "no output")[-8000:]
            raise RuntimeError(f"SAM2 tracking failed: {detail}") from exc
        if not result_path.exists():
            raise RuntimeError(f"SAM2 worker returned no tracking.json: {completed.stdout}")
        return json.loads(result_path.read_text(encoding="utf-8"))


class TrackedArtifactWriter:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def build_overlay_video(tracking_root: Path) -> Path:
        output = tracking_root / "tracking_overlay_5fps.mp4"
        if output.exists():
            return output
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                "5",
                "-i",
                str(tracking_root / "overlays" / "%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            check=True,
        )
        return output

    def write_report(self, manifest: dict[str, Any]) -> Path:
        preprocessing = manifest["stages"]["01_preprocessing"]["output"]
        tracking = manifest["stages"]["02_sam2_tracking"]["output"]
        merged = manifest["stages"]["03_overlap_merge"]["output"]
        descriptions = manifest["stages"]["04_dam_description"]["output"]
        annotation = manifest["stages"]["05_openai_normalization"]["output"]
        verification = manifest["stages"]["06_sam3_verification"]["output"]
        vqa = manifest["stages"]["07_vqa"]["output"]

        clip_sections = []
        for clip, result in zip(preprocessing["clips"], tracking, strict=True):
            clip_index = int(clip["clip_index"])
            clip_sections.append(
                f"""<section class="card"><h2>{html.escape(clip['clip_id'])}</h2>
                <p>Source FPS {clip['source_fps']:.2f} · Track FPS {clip['tracking_fps']:.1f} · {result['tracking_frame_count']} tracking frames · {result['initial_object_count']} tracks</p>
                <div class="videos"><figure><video controls src="{html.escape(str(Path(clip['clip_path']).relative_to(self.output_root)))}"></video><figcaption>원본 FPS Clip</figcaption></figure>
                <figure><video controls src="sam2/clip_{clip_index:03d}/tracking_overlay_5fps.mp4"></video><figcaption>SAM2 Track ID + Mask</figcaption></figure></div></section>"""
            )
        description_by_id = {item["global_track_id"]: item for item in descriptions}
        verification_by_id = {item["global_track_id"]: item for item in verification}
        track_rows = "".join(
            f"<tr><td>{track['global_track_id']}</td><td>{html.escape(', '.join(track['local_track_ids']))}</td><td>{len(track['representative_frames'])}</td><td>{html.escape(description_by_id[track['global_track_id']]['description'])}</td><td>{verification_by_id[track['global_track_id']]['status']}</td></tr>"
            for track in merged["global_tracks"]
        )
        change_rows = "".join(
            f"<tr><td>{change['change_id']}</td><td>{change['change_type']}</td><td>{html.escape(change['before_state'])}</td><td>{html.escape(change['after_state'])}</td><td>{change['before_timestamp_seconds']:.3f}, {change['after_timestamp_seconds']:.3f}</td></tr>"
            for change in annotation["observed_changes"]
        ) or '<tr><td colspan="5">직접 관찰된 before/after 변화 없음</td></tr>'
        vqa_rows = "".join(
            f"<tr><td>{item['category']}</td><td>{html.escape(item['question'])}</td><td>{html.escape(item['answer'])}</td><td>{html.escape(str(item['evidence']['source_timestamps_seconds']))}</td></tr>"
            for item in vqa["items"]
        ) or '<tr><td colspan="4">생성 가능한 근거 문항 없음</td></tr>'
        skipped_rows = "".join(
            f"<tr><td>{item['category']}</td><td>{item['reason_code']}</td><td>{html.escape(item['detail'])}</td></tr>"
            for item in vqa["skipped_categories"]
        )
        document = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tracked VQA MVP</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f3f5f7;max-width:1500px;margin:auto;padding:24px;color:#17202a}}.card{{background:white;padding:20px;border-radius:14px;margin-bottom:22px;box-shadow:0 2px 14px #0001}}.hero{{border-left:8px solid #2471a3}}.videos{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}}video{{width:100%;background:#111}}figure{{margin:0}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{border:1px solid #d5d8dc;padding:8px;text-align:left;vertical-align:top}}th{{background:#eef2f3}}pre{{white-space:pre-wrap;word-break:break-word;background:#f8f9f9;padding:14px}}</style></head><body>
<section class="card hero"><h1>Track-grounded VQA MVP · {html.escape(manifest['video_id'])}</h1><p>원본 FPS Clip → 5 FPS SAM2 tracking → Track별 최대 8 DAM frames → 구조화 → SAM3 IoU → 보수적 VQA</p><video controls src="source_video.mp4"></video></section>
{''.join(clip_sections)}
<section class="card"><h2>Overlap Track 병합</h2><p>local tracks {sum(item['initial_object_count'] for item in tracking)} → global tracks {merged['global_track_count']}</p><table><thead><tr><th>Global</th><th>Local</th><th>DAM frames</th><th>Description</th><th>SAM3</th></tr></thead><tbody>{track_rows}</tbody></table></section>
<section class="card"><h2>직접 관찰된 변화</h2><table><thead><tr><th>ID</th><th>Type</th><th>Before</th><th>After</th><th>Evidence metadata only</th></tr></thead><tbody>{change_rows}</tbody></table></section>
<section class="card"><h2>생성된 VQA</h2><p>Timestamp와 frame index는 Evidence 열에만 표시된다.</p><table><thead><tr><th>Category</th><th>Question</th><th>Answer</th><th>Evidence timestamps</th></tr></thead><tbody>{vqa_rows}</tbody></table></section>
<section class="card"><h2>생성하지 않은 카테고리</h2><table><thead><tr><th>Category</th><th>Reason</th><th>Detail</th></tr></thead><tbody>{skipped_rows}</tbody></table><details><summary>Full manifest</summary><pre>{html.escape(json.dumps(manifest, ensure_ascii=False, indent=2))}</pre></details></section>
</body></html>"""
        path = self.output_root / "report.html"
        path.write_text(document, encoding="utf-8")
        return path


class TrackedVQAPipeline:
    def __init__(self, config: TrackedPipelineConfig, api_key: str) -> None:
        self.config = config
        self.client = OpenAI(api_key=api_key, timeout=300.0, max_retries=2)
        self.writer = TrackedArtifactWriter(config.output_root)
        self.preprocessor = OverlappingClipPreprocessor(
            clip_seconds=config.clip_seconds,
            stride_seconds=config.stride_seconds,
            tracking_fps=config.tracking_fps,
        )
        self.sam2 = Sam2TrackingRunner(
            config.sam2_python,
            PROJECT_ROOT / "workers" / "sam2_track.py",
        )
        self.dam = DAMTrackDescriber(
            self.client, config.model, config.dam_base_url
        )
        self.analyzer = OpenAITrackedAnalyzer(self.client, config.model)
        self.sam3 = Sam3NameVerifier(config.sam3_base_url)

    def run(self) -> Path:
        if self.config.strict_models and not self.config.dam_base_url:
            raise RuntimeError("strict_models requires DAM_BASE_URL")
        if self.config.strict_models and not self.config.sam3_base_url:
            raise RuntimeError("strict_models requires SAM3_BASE_URL")
        output_root = self.config.output_root
        shutil.copy2(self.config.video_path, output_root / "source_video.mp4")
        video_id = self.config.video_path.stem
        preprocessing = self.preprocessor.prepare(
            self.config.video_path, output_root, video_id
        )

        track_results = []
        track_roots = []
        for clip in preprocessing["clips"]:
            clip_index = int(clip["clip_index"])
            root = output_root / "sam2" / f"clip_{clip_index:03d}"
            print(f"SAM2 tracking clip {clip_index + 1}/{len(preprocessing['clips'])}", flush=True)
            result = self.sam2.run(
                output_root / "preprocessing.json", clip_index, root
            )
            self.writer.build_overlay_video(root)
            track_results.append(result)
            track_roots.append(root)

        merged = OverlapTrackMerger().merge(list(zip(track_results, track_roots, strict=True)))
        self.writer.write_json(output_root / "merged_tracks.json", merged)

        descriptions = []
        for index, track in enumerate(merged["global_tracks"]):
            description_path = output_root / "dam" / f"{track['global_track_id']}.json"
            if description_path.exists():
                description = TrackDescription.model_validate_json(
                    description_path.read_text(encoding="utf-8")
                )
            else:
                print(f"Track description {index + 1}/{len(merged['global_tracks'])}", flush=True)
                description = self.dam.describe(track)
                self.writer.write_json(description_path, description.model_dump(mode="json"))
            descriptions.append(description)

        annotation_path = output_root / "annotation_before_sam3.json"
        if annotation_path.exists():
            annotation = TrackedAnnotation.model_validate_json(
                annotation_path.read_text(encoding="utf-8")
            )
            normalization_meta = {"resumed_from_existing_artifact": True}
        else:
            annotation, normalization_meta = self.analyzer.normalize(merged, descriptions)
            self.writer.write_json(annotation_path, annotation.model_dump(mode="json"))

        verifications = self.sam3.verify(annotation, merged)
        verified_annotation = self.analyzer.apply_sam3_results(annotation, verifications)
        self.writer.write_json(
            output_root / "annotation.json", verified_annotation.model_dump(mode="json")
        )
        vqa, vqa_meta = self.analyzer.generate_vqa(
            verified_annotation, self.config.strict_verification
        )
        self.writer.write_json(output_root / "vqa.json", vqa.model_dump(mode="json"))

        manifest = {
            "schema_version": "0.3.0-track-grounded",
            "video_id": video_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "requirements": {
                "source_fps_preserved": True,
                "clip_seconds": self.config.clip_seconds,
                "stride_seconds": self.config.stride_seconds,
                "sam2_tracking_fps": self.config.tracking_fps,
                "dam_max_frames_per_track": 8,
                "timestamps_are_evidence_metadata_only": True,
                "unsupported_change_questions_are_skipped": True,
                "overlap_deduplicated_before_vqa": True,
            },
            "stages": {
                "01_preprocessing": {"status": "executed", "output": preprocessing},
                "02_sam2_tracking": {"status": "executed", "output": track_results},
                "03_overlap_merge": {"status": "executed", "output": merged},
                "04_dam_description": {
                    "status": (
                        "executed"
                        if all(item.provider_status == "executed" for item in descriptions)
                        else "fallback"
                    ),
                    "fallback_is_not_dam_output": any(
                        item.provider_status == "fallback" for item in descriptions
                    ),
                    "output": [item.model_dump(mode="json") for item in descriptions],
                },
                "05_openai_normalization": {
                    "status": "executed",
                    "model": self.config.model,
                    "api": normalization_meta,
                    "output": annotation.model_dump(mode="json"),
                },
                "06_sam3_verification": {
                    "status": (
                        "executed"
                        if all(item.status != "blocked" for item in verifications)
                        else "blocked"
                    ),
                    "output": [item.model_dump(mode="json") for item in verifications],
                },
                "07_vqa": {
                    "status": "executed",
                    "model": self.config.model,
                    "strict_verification": self.config.strict_verification,
                    "api": vqa_meta,
                    "output": vqa.model_dump(mode="json"),
                },
            },
        }
        self.writer.write_json(output_root / "manifest.json", manifest)
        return self.writer.write_report(manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the track-grounded industrial VideoQA MVP")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--sam2-python", type=Path, default=PROJECT_ROOT / ".venv-sam2/bin/python")
    parser.add_argument("--strict-models", action="store_true")
    parser.add_argument("--strict-verification", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    config = TrackedPipelineConfig(
        video_path=args.video.resolve(),
        output_root=args.output_dir.resolve(),
        model=args.model or os.getenv("OPENAI_MODEL", "gpt-6.1-terra"),
        sam2_python=args.sam2_python.absolute(),
        strict_models=args.strict_models,
        strict_verification=args.strict_verification,
        dam_base_url=os.getenv("DAM_BASE_URL") or None,
        sam3_base_url=os.getenv("SAM3_BASE_URL") or None,
    )
    print(TrackedVQAPipeline(config, api_key).run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
