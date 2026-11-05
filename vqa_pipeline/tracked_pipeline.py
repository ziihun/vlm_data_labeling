from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .annotation import apply_human_review, build_pending_annotation
from .crops import MarginCropGenerator
from .openai_vqa import OpenAIVQAGenerator, apply_final_review
from .report import StageReport
from .naming_experiment import prepare_assets
from .spec_models import DraftVQABatch, FinalVQAReview, HumanReview
from .track_merge import OverlapTrackMerger
from .video_clips import OverlappingClipPreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "raw_datasets" / "Safe_and_Unsafe_Behaviours" / "data" / "0_te1.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "0_te1"


@dataclass(frozen=True, slots=True)
class Config:
    video_path: Path
    output_root: Path
    vision_python: Path
    qwen_python: Path
    openai_model: str
    naming_model: str = "mlx-community/Qwen3.5-9B-MLX-4bit"
    stop_after: str = "vqa"
    segmenter: str = "sam2"
    clip_seconds: float = 10.0
    stride_seconds: float = 5.0
    tracking_fps: float = 5.0
    redetect_interval_frames: int = 10
    lean: bool = False


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Pipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        if config.segmenter not in {"sam2", "sam3"}:
            raise ValueError("segmenter must be sam2 or sam3")
        self.stage01 = f"01_{config.segmenter}"
        self.stage_names = (
            "00_preprocessing", self.stage01, "02_crops", "03_qwen_naming",
            "04_annotation", "05_human_review", "06_vqa", "07_final",
        )
        self.reporter = StageReport(config.output_root)
        self.manifest_path = config.output_root / "manifest.json"
        self.manifest = {
            "schema_version": "2.0.0-revised-spec",
            "video_id": config.video_path.stem,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": f"{config.segmenter.upper()} -> process exit -> Qwen3.5-9B -> process exit -> Structured Annotation -> separate VQA process",
            "segmentation_model": config.segmenter,
            "language_policy": {"prompts": "English", "json_keys_and_values": "English", "final_question_answer": "English", "html_ui": "Korean"},
            "stages": {name: {"status": "skipped", "detail": "Not executed yet"} for name in self.stage_names},
        }
        config.output_root.mkdir(parents=True, exist_ok=True)

    def status(self, stage: str, status: str, detail: str, **extra: Any) -> None:
        self.manifest["stages"][stage] = {"status": status, "detail": detail, **extra}
        write_json(self.manifest_path, self.manifest)

    def fail_remaining(self, failed_stage: str, detail: str, preprocessing: dict | None = None, clips: list | None = None, pending: dict | None = None) -> Path:
        self.status(failed_stage, "failed", detail, traceback=traceback.format_exc())
        failed_index = self.stage_names.index(failed_stage)
        for stage in self.stage_names[failed_index + 1 :]:
            self.manifest["stages"][stage] = {"status": "skipped", "detail": f"Blocked by {failed_stage}"}
        write_json(self.manifest_path, self.manifest)
        if self.config.lean:
            return self.manifest_path
        return self.reporter.write(self.manifest, preprocessing, clips, pending, None)

    @staticmethod
    def run_process(command: list[str], env: dict[str, str] | None = None) -> None:
        completed = subprocess.run(command, text=True, capture_output=True, env=env)
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        if completed.returncode:
            raise RuntimeError((completed.stderr or completed.stdout or "Process failed")[-12000:])

    @staticmethod
    def overlay_video(root: Path, fps: float) -> None:
        output = root / "tracking_overlay.mp4"
        if output.exists():
            return
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", f"{fps:g}", "-i", str(root / "overlays" / "%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
        ], check=True)

    def run(self) -> Path:
        output = self.config.output_root
        if not self.config.lean:
            shutil.copy2(self.config.video_path, output / "source_video.mp4")
        preprocessing = None
        clip_results: list[dict[str, Any]] = []
        pending = None

        try:
            stage00 = output / "00_preprocessing"
            preprocessing = OverlappingClipPreprocessor(
                self.config.clip_seconds, self.config.stride_seconds, self.config.tracking_fps
            ).prepare(self.config.video_path, stage00, self.config.video_path.stem)
            self.status("00_preprocessing", "completed", f"{len(preprocessing['clips'])} clips; source FPS preserved", artifact="00_preprocessing/manifest.json")
        except Exception as exc:
            return self.fail_remaining("00_preprocessing", f"{type(exc).__name__}: {exc}")

        try:
            if not self.config.vision_python.exists():
                raise RuntimeError(f"Vision Python environment not found: {self.config.vision_python}")
            roots = []
            for clip in preprocessing["clips"]:
                index = int(clip["clip_index"])
                root = output / self.stage01 / f"clip_{index:03d}"
                result_path = root / "tracks.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    print(f"[01] {self.config.segmenter.upper()} clip {index + 1}/{len(preprocessing['clips'])}", flush=True)
                    command = [
                        str(self.config.vision_python), str(PROJECT_ROOT / "workers" / f"{self.config.segmenter}_track.py"),
                        "--preprocessing-manifest", str(stage00 / "manifest.json"),
                        "--clip-index", str(index), "--output-dir", str(root),
                        "--redetect-interval-frames", str(self.config.redetect_interval_frames),
                    ]
                    if self.config.lean:
                        command.append("--no-render")
                    self.run_process(command, os.environ.copy())
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                if not self.config.lean:
                    self.overlay_video(root, self.config.tracking_fps)
                clip_results.append(result)
                roots.append(root)
            merged = OverlapTrackMerger().merge(list(zip(clip_results, roots, strict=True)))
            write_json(output / self.stage01 / "tracks.json", merged)
            devices = sorted({item["device"] for item in clip_results})
            model_id = clip_results[0]["model"]
            cuda_note = ""
            if self.config.segmenter == "sam3":
                official_cuda = all(item["official_cuda_prerequisite_met"] for item in clip_results)
                cuda_note = f"; official CUDA prerequisite met={official_cuda}"
            detail = f"{merged['track_count']} global Tracks; devices={devices}{cuda_note}"
            self.status(self.stage01, "completed", detail, model=model_id, artifact=f"{self.stage01}/tracks.json", local_track_count=sum(item["track_count"] for item in clip_results), global_track_count=merged["track_count"])
        except Exception as exc:
            return self.fail_remaining(self.stage01, f"{type(exc).__name__}: {exc}", preprocessing, clip_results)

        try:
            crops = MarginCropGenerator(0.20, 8).run(merged, output / "02_crops")
            self.status("02_crops", "completed", f"20% margin crops for {len(crops['tracks'])} Tracks", artifact="02_crops/crops.json")
        except Exception as exc:
            return self.fail_remaining("02_crops", f"{type(exc).__name__}: {exc}", preprocessing, clip_results)

        if self.config.stop_after == "sam":
            self.status("03_qwen_naming", "skipped", "Separate process boundary: rerun with --stop-after naming")
            self.status("04_annotation", "skipped", "Naming process has not run")
            self.status("05_human_review", "skipped", "Naming process has not run")
            self.status("06_vqa", "skipped", "VQA process has not run")
            self.status("07_final", "skipped", "VQA process has not run")
            if self.config.lean:
                shutil.rmtree(output / "00_preprocessing" / "clips", ignore_errors=True)
                for clip_root in (output / self.stage01).glob("clip_*"):
                    if clip_root.is_dir():
                        shutil.rmtree(clip_root)
                return self.manifest_path
            return self.reporter.write(self.manifest, preprocessing, clip_results, None, None)

        try:
            naming_dir = output / "03_qwen_naming"
            naming_path = naming_dir / "naming.json"
            existing_model = None
            if naming_path.exists():
                existing_model = json.loads(naming_path.read_text(encoding="utf-8")).get("model")
            if not naming_path.exists() or existing_model != self.config.naming_model:
                if not self.config.qwen_python.exists():
                    raise RuntimeError(f"Qwen MLX Python environment not found: {self.config.qwen_python}")
                assets_dir = naming_dir / "input"
                prepare_assets(output / "02_crops" / "crops.json", assets_dir, output / self.stage01 / "tracks.json")
                print(f"[03] {self.config.naming_model} object naming (separate process)", flush=True)
                self.run_process([
                    str(self.config.qwen_python), str(PROJECT_ROOT / "workers" / "local_vlm_name.py"),
                    "--assets-json", str(assets_dir / "assets.json"),
                    "--output-dir", str(naming_dir),
                    "--model", self.config.naming_model,
                ], os.environ.copy())
            naming = json.loads(naming_path.read_text(encoding="utf-8"))
            if {item["track_id"] for item in naming["items"]} != {item["track_id"] for item in merged["tracks"]}:
                raise ValueError("Qwen naming does not cover every Track")
            self.status("03_qwen_naming", "completed", f"Named {len(naming['items'])} Tracks; human verification still required", model=naming["model"], artifacts=["03_qwen_naming/prompt.txt", "03_qwen_naming/raw_response.json", "03_qwen_naming/naming.json"])
        except Exception as exc:
            return self.fail_remaining("03_qwen_naming", f"{type(exc).__name__}: {exc}", preprocessing, clip_results)

        try:
            pending = build_pending_annotation(self.config.video_path.stem, merged, crops, naming, output / "04_annotation")
            self.status("04_annotation", "completed", f"{len(pending['objects'])} pending objects; not benchmark ground truth", artifact="04_annotation/objects.json")
        except Exception as exc:
            return self.fail_remaining("04_annotation", f"{type(exc).__name__}: {exc}", preprocessing, clip_results)

        if self.config.stop_after == "naming":
            self.status("05_human_review", "pending_review", "Review unknown or confidence < 0.7 Tracks before final VQA")
            self.status("06_vqa", "skipped", "Separate VQA process has not run")
            self.status("07_final", "skipped", "VQA process has not run")
            return self.reporter.write(self.manifest, preprocessing, clip_results, pending, None)

        review_dir = output / "05_human_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        review_path = review_dir / "review.json"
        if not review_path.exists():
            (review_dir / "README.txt").write_text("Use report.html to review every object, download review.json, and place it in this directory.\n", encoding="utf-8")
            self.status("05_human_review", "pending_review", "Open report.html, verify all object names/attributes/states, then save review.json here")
            self.status("06_vqa", "skipped", "Human review is not completed; OpenAI API was not called")
            self.status("07_final", "skipped", "No draft VQA exists")
            return self.reporter.write(self.manifest, preprocessing, clip_results, pending, None)

        try:
            review = HumanReview.model_validate_json(review_path.read_text(encoding="utf-8"))
            if review.video_id != self.config.video_path.stem:
                raise ValueError("Human review video_id mismatch")
            verified = apply_human_review(pending, review, review_dir)
            self.status("05_human_review", "completed", f"{len(verified.objects)} approved/modified objects", artifact="05_human_review/verified_annotations.json")
        except Exception as exc:
            return self.fail_remaining("05_human_review", f"{type(exc).__name__}: {exc}", preprocessing, clip_results, pending)

        try:
            vqa_path = output / "06_vqa" / "generated_vqa.json"
            if vqa_path.exists():
                draft = DraftVQABatch.model_validate_json(vqa_path.read_text(encoding="utf-8"))
            else:
                api_key = os.getenv("OPENAI_API_KEY", "")
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY is missing")
                draft = OpenAIVQAGenerator(OpenAI(api_key=api_key, timeout=300.0), self.config.openai_model).run(verified, output / "06_vqa")
            self.status("06_vqa", "completed", f"{len(draft.items)} draft questions; final review required", model=self.config.openai_model, artifacts=["06_vqa/prompt.txt", "06_vqa/raw_response.json", "06_vqa/generated_vqa.json"])
        except Exception as exc:
            return self.fail_remaining("06_vqa", f"{type(exc).__name__}: {exc}", preprocessing, clip_results, pending)

        final_dir = output / "07_final"
        final_review_path = final_dir / "final_review.json"
        if not final_review_path.exists():
            final_dir.mkdir(parents=True, exist_ok=True)
            self.status("07_final", "pending_review", "Review every draft Q&A in report.html and save final_review.json here")
            return self.reporter.write(self.manifest, preprocessing, clip_results, pending, draft.model_dump(mode="json"))
        try:
            final_review = FinalVQAReview.model_validate_json(final_review_path.read_text(encoding="utf-8"))
            final = apply_final_review(draft, final_review, final_dir)
            self.status("07_final", "completed", f"{len(final['items'])} human-verified benchmark questions", artifact="07_final/verified_vqa.json")
        except Exception as exc:
            return self.fail_remaining("07_final", f"{type(exc).__name__}: {exc}", preprocessing, clip_results, pending)
        return self.reporter.write(self.manifest, preprocessing, clip_results, pending, draft.model_dump(mode="json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Revised track-grounded VQA dataset pipeline")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--segmenter", choices=("sam2", "sam3"), default="sam2")
    parser.add_argument("--vision-python", type=Path, default=PROJECT_ROOT / ".venv-sam2/bin/python")
    parser.add_argument("--qwen-python", type=Path, default=PROJECT_ROOT / ".venv-qwen/bin/python")
    parser.add_argument("--openai-model", default=None)
    parser.add_argument("--naming-model", default="mlx-community/Qwen3.5-9B-MLX-4bit")
    parser.add_argument("--stop-after", choices=("sam", "naming", "vqa"), default="vqa")
    parser.add_argument("--redetect-interval-frames", type=int, default=10)
    parser.add_argument("--lean", action="store_true", help="Skip visualization/report files and retain only data needed by later stages")
    return parser.parse_args()


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    args = parse_args()
    default_output = DEFAULT_OUTPUT if args.segmenter == "sam3" else PROJECT_ROOT / "outputs" / "0_te1_sam2"
    config = Config(
        video_path=args.video.resolve(), output_root=(args.output_dir or default_output).resolve(),
        vision_python=args.vision_python.absolute(), qwen_python=args.qwen_python.absolute(),
        openai_model=args.openai_model or os.getenv("OPENAI_MODEL", "gpt-6.1-terra"),
        naming_model=args.naming_model,
        stop_after=args.stop_after,
        segmenter=args.segmenter,
        redetect_interval_frames=args.redetect_interval_frames,
        lean=args.lean,
    )
    pipeline = Pipeline(config)
    print(pipeline.run())
    if any(stage.get("status") == "failed" for stage in pipeline.manifest["stages"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
