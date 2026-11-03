from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw
from pydantic import BaseModel

from .trace_models import MvpVQA, NormalizedObject, RegionProposal, RoiDescription


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_ROOT = PROJECT_ROOT / "vqa_datasets" / "pilot_5"
DEFAULT_TRACE_ROOT = PROJECT_ROOT / "vqa_datasets" / "mvp_trace"
T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class TraceConfig:
    pilot_root: Path
    output_root: Path
    video_id: str
    category: str
    model: str
    sam2_python: Path
    sam2_worker: Path


class VQARecordStore:
    def __init__(self, pilot_root: Path) -> None:
        self.pilot_root = pilot_root

    def find(self, video_id: str, category: str) -> dict[str, Any]:
        with (self.pilot_root / "vqa.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["video_id"] == video_id and record["category"] == category:
                    return record
        raise ValueError(f"No VQA record for video={video_id}, category={category}")

    def annotation(self, video_id: str) -> dict[str, Any]:
        path = self.pilot_root / "annotations" / f"{video_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def evidence_frame(self, video_id: str, frame_index: int) -> tuple[Path, float]:
        annotation = self.annotation(video_id)
        for frame in annotation["source"]["sampled_frames"]:
            if frame["index"] == frame_index:
                return self.pilot_root / frame["path"], float(frame["timestamp_seconds"])
        raise ValueError(f"Frame {frame_index} not found for {video_id}")


class OpenAITraceAnalyzer:
    """Produces explicit structured outputs for each OpenAI-powered trace stage."""

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, timeout=180.0, max_retries=2)

    @staticmethod
    def _data_url(path: Path) -> str:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return f"data:{media_type};base64,{encoded}"

    def _parse(
        self,
        schema: type[T],
        instructions: str,
        content: list[dict[str, Any]] | str,
    ) -> tuple[T, dict[str, Any]]:
        response = self.client.responses.parse(
            model=self.model,
            instructions=instructions,
            input=[{"role": "user", "content": content}],
            text_format=schema,
            max_output_tokens=2500,
            store=False,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError(f"No parsed output from response {response.id}")
        return parsed, {
            "response_id": response.id,
            "executed_model": response.model,
            "usage": response.usage.model_dump(mode="json") if response.usage else None,
        }

    def propose_region(
        self,
        question: str,
        answer: str,
        frame_path: Path,
        timestamp: float,
    ) -> tuple[RegionProposal, dict[str, Any]]:
        prompt = f"""질문에 답하는 데 가장 직접적으로 필요한 한 개의 시각 영역을 찾으세요.
전체 장면이 아니라 가능한 한 작은 관련 객체/영역을 선택하세요.
좌표는 이미지 전체를 0~1000으로 정규화한 x_min, y_min, x_max, y_max입니다.

질문: {question}
기존 답변 후보: {answer}
프레임 시각: {timestamp:.3f}초
"""
        return self._parse(
            RegionProposal,
            "산업 CCTV 질문을 근거 영역에 연결하는 visual localizer입니다. 보이는 것만 사용하세요.",
            [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": self._data_url(frame_path), "detail": "high"},
            ],
        )

    def describe_roi(
        self,
        crop_path: Path,
        question: str,
    ) -> tuple[RoiDescription, dict[str, Any]]:
        prompt = f"""이 이미지는 질문 관련 ROI crop입니다.
DAM-3B-Video가 로컬 하드웨어에서 실행 불가하여 수행하는 명시적 fallback 설명입니다.
보이는 객체, 속성, 상태만 자세히 기술하고 불확실한 것은 uncertainties에 넣으세요.
질문: {question}
"""
        return self._parse(
            RoiDescription,
            "산업 영상 ROI를 상세하게 설명하되 보이지 않는 행동이나 의도를 추정하지 마세요.",
            [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": self._data_url(crop_path), "detail": "high"},
            ],
        )

    def normalize_object(
        self,
        description: RoiDescription,
    ) -> tuple[NormalizedObject, dict[str, Any]]:
        return self._parse(
            NormalizedObject,
            "ROI 설명을 제조 CCTV용 표준 객체명, 속성, 상태로 정규화하세요.",
            json.dumps(description.model_dump(mode="json"), ensure_ascii=False),
        )

    def compose_vqa(
        self,
        original: dict[str, Any],
        normalized: NormalizedObject,
        crop_path: Path,
    ) -> tuple[MvpVQA, dict[str, Any]]:
        prompt = f"""단계별 산출물로 하나의 검증 가능한 MVP VQA를 다시 구성하세요.
원래 category를 유지하고, ROI에서 직접 확인되는 내용만 답으로 사용하세요.

원래 category: {original['category']}
원래 질문: {original['question']}
원래 답변: {original['answer']}
정규화 객체: {json.dumps(normalized.model_dump(mode='json'), ensure_ascii=False)}
"""
        return self._parse(
            MvpVQA,
            "산업 CCTV의 한 ROI에 근거한 짧고 명확한 한국어 VQA를 만드세요.",
            [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": self._data_url(crop_path), "detail": "high"},
            ],
        )


class Sam2Runner:
    def __init__(self, python_path: Path, worker_path: Path) -> None:
        self.python_path = python_path
        self.worker_path = worker_path

    def run(self, image_path: Path, box: list[int], output_dir: Path) -> dict[str, Any]:
        environment = os.environ.copy()
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        command = [
            str(self.python_path),
            str(self.worker_path),
            "--image",
            str(image_path),
            "--box",
            *[str(value) for value in box],
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
            detail = (exc.stderr or exc.stdout or "no worker output")[-5000:]
            raise RuntimeError(f"SAM 2 worker failed: {detail}") from exc
        result_path = output_dir / "result.json"
        if not result_path.exists():
            raise RuntimeError(f"SAM 2 worker returned no result: {completed.stdout}")
        return json.loads(result_path.read_text(encoding="utf-8"))


class TraceArtifactWriter:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, name: str) -> Path:
        path = self.output_root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def write_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def pixel_box(proposal: RegionProposal, image_path: Path) -> list[int]:
        width, height = Image.open(image_path).size
        box = proposal.bbox_normalized_0_1000
        return [
            int(round(box.x_min / 1000 * width)),
            int(round(box.y_min / 1000 * height)),
            int(round(box.x_max / 1000 * width)),
            int(round(box.y_max / 1000 * height)),
        ]

    @staticmethod
    def draw_box(image_path: Path, box: list[int], output_path: Path, label: str) -> None:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline=(255, 220, 0), width=5)
        draw.rectangle(
            [box[0], max(0, box[1] - 28), min(image.width, box[0] + 360), box[1]],
            fill=(255, 220, 0),
        )
        draw.text((box[0] + 5, max(0, box[1] - 24)), label[:45], fill=(0, 0, 0))
        image.save(output_path, quality=95)

    def write_report(self, manifest: dict[str, Any]) -> None:
        def json_block(value: Any) -> str:
            return html.escape(json.dumps(value, ensure_ascii=False, indent=2))

        final_vqa = manifest["stages"]["07_final_vqa"]["output"]
        stage_cards = []
        image_map = {
            "01_evidence": ["evidence_frame.jpg"],
            "02_region_proposal": ["bbox_overlay.jpg"],
            "03_sam2": ["overlay.png", "mask.png", "roi_crop_context.jpg", "roi_crop.png"],
        }
        for stage_name, stage in manifest["stages"].items():
            status = stage["status"]
            media = "".join(
                f'<img src="{stage_name}/{asset}" alt="{asset}">'
                for asset in image_map.get(stage_name, [])
            )
            stage_cards.append(
                f"""
                <section class="stage {html.escape(status)}">
                  <h2>{html.escape(stage_name)}</h2>
                  <div class="meta"><b>Status:</b> {html.escape(status)} · <b>Requested:</b> {html.escape(str(stage.get('requested_model')))} · <b>Executed:</b> {html.escape(str(stage.get('executed_model')))}</div>
                  <div class="media">{media}</div>
                  <details open><summary>Stage JSON</summary><pre>{json_block(stage)}</pre></details>
                </section>
                """
            )

        document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VQA Stage Trace MVP</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f3f5f7;color:#17202a;max-width:1500px;margin:auto;padding:24px}}
.hero,.stage{{background:white;border-radius:14px;padding:20px;margin-bottom:24px;box-shadow:0 2px 14px #0001}}
.hero{{border-left:8px solid #2471a3}} .executed{{border-left:7px solid #1e8449}} .fallback{{border-left:7px solid #d68910}} .blocked{{border-left:7px solid #c0392b}}
video{{width:min(100%,960px);background:#111}} .media{{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}} .media img{{max-width:min(100%,680px);max-height:520px;object-fit:contain;border:1px solid #ccd1d1;background:#111}}
pre{{white-space:pre-wrap;word-break:break-word;background:#f8f9f9;padding:14px;border-radius:8px}} .meta{{color:#566573}} code{{background:#eef2f3;padding:2px 5px}}
</style></head><body>
<div class="hero"><h1>Question-conditioned VQA Stage Trace</h1>
<p><b>Final MVP question:</b> {html.escape(final_vqa['question'])}</p><p><b>Answer:</b> {html.escape(final_vqa['answer'])}</p>
<p>초록=실제 실행, 주황=fallback, 빨강=환경 제약으로 미실행. 각 단계의 모델 정체와 원본 JSON을 그대로 표시한다.</p>
<video controls preload="metadata" src="source_video.mp4"></video></div>
{''.join(stage_cards)}
</body></html>"""
        (self.output_root / "report.html").write_text(document, encoding="utf-8")


class QuestionTracePipeline:
    def __init__(
        self,
        config: TraceConfig,
        records: VQARecordStore,
        analyzer: OpenAITraceAnalyzer,
        sam2: Sam2Runner,
        writer: TraceArtifactWriter,
    ) -> None:
        self.config = config
        self.records = records
        self.analyzer = analyzer
        self.sam2 = sam2
        self.writer = writer

    def run(self) -> Path:
        original = self.records.find(self.config.video_id, self.config.category)
        evidence_index = int(original["evidence"]["frame_indices"][0])
        frame_path, timestamp = self.records.evidence_frame(
            self.config.video_id, evidence_index
        )
        source_video = self.config.pilot_root / original["review_video_path"]
        shutil.copy2(source_video, self.config.output_root / "source_video.mp4")

        stages: dict[str, Any] = {}
        stages["00_question"] = {
            "status": "executed",
            "requested_model": None,
            "executed_model": "existing pilot VQA selector",
            "output": {
                "video_id": original["video_id"],
                "category": original["category"],
                "question": original["question"],
                "answer": original["answer"],
                "evidence_frame_index": evidence_index,
                "evidence_timestamp_seconds": timestamp,
            },
        }

        evidence_dir = self.writer.stage_dir("01_evidence")
        evidence_copy = evidence_dir / "evidence_frame.jpg"
        shutil.copy2(frame_path, evidence_copy)
        stages["01_evidence"] = {
            "status": "executed",
            "requested_model": None,
            "executed_model": "ffmpeg sampled evidence",
            "output": {
                "frame_index": evidence_index,
                "timestamp_seconds": timestamp,
                "artifact": "evidence_frame.jpg",
            },
        }

        proposal, proposal_meta = self.analyzer.propose_region(
            original["question"], original["answer"], evidence_copy, timestamp
        )
        proposal_dir = self.writer.stage_dir("02_region_proposal")
        pixel_box = self.writer.pixel_box(proposal, evidence_copy)
        proposal_payload = {
            "status": "executed",
            "requested_model": self.config.model,
            "executed_model": proposal_meta["executed_model"],
            "purpose": "question-conditioned box prompt for SAM 2",
            "output": {
                **proposal.model_dump(mode="json"),
                "bbox_pixels": pixel_box,
            },
            "api": proposal_meta,
        }
        self.writer.write_json(proposal_dir / "result.json", proposal_payload)
        self.writer.draw_box(
            evidence_copy,
            pixel_box,
            proposal_dir / "bbox_overlay.jpg",
            proposal.target_phrase_english,
        )
        stages["02_region_proposal"] = proposal_payload

        sam2_dir = self.writer.stage_dir("03_sam2")
        sam2_result = self.sam2.run(evidence_copy, pixel_box, sam2_dir)
        stages["03_sam2"] = sam2_result

        crop_path = sam2_dir / "roi_crop_context.jpg"
        roi_description, description_meta = self.analyzer.describe_roi(
            crop_path, original["question"]
        )
        dam_dir = self.writer.stage_dir("04_dam")
        dam_payload = {
            "status": "fallback",
            "requested_model": "nvidia/DAM-3B-Video",
            "executed_model": description_meta["executed_model"],
            "blocked_reason": (
                "DAM-3B-Video official support is NVIDIA Ampere/Hopper/Lovelace on Linux; "
                "this host is Apple Silicon with 16 GB memory. The public NVIDIA Space is paused."
            ),
            "fallback_is_not_dam_output": True,
            "output": roi_description.model_dump(mode="json"),
            "api": description_meta,
        }
        self.writer.write_json(dam_dir / "result.json", dam_payload)
        stages["04_dam"] = dam_payload

        normalized, normalization_meta = self.analyzer.normalize_object(roi_description)
        normalization_dir = self.writer.stage_dir("05_normalization")
        normalization_payload = {
            "status": "executed",
            "requested_model": self.config.model,
            "executed_model": normalization_meta["executed_model"],
            "input_stage": "04_dam fallback output",
            "output": normalized.model_dump(mode="json"),
            "api": normalization_meta,
        }
        self.writer.write_json(normalization_dir / "result.json", normalization_payload)
        stages["05_normalization"] = normalization_payload

        sam3_dir = self.writer.stage_dir("06_sam3")
        sam3_payload = {
            "status": "blocked",
            "requested_model": "facebook/sam3.1",
            "executed_model": None,
            "blocked_reason": (
                "Official SAM 3 requires a CUDA 12.6 compatible GPU and approved checkpoint "
                "access; neither is available on this Apple Silicon host."
            ),
            "verification_result": "not_verified",
        }
        self.writer.write_json(sam3_dir / "result.json", sam3_payload)
        stages["06_sam3"] = sam3_payload

        final_vqa, final_meta = self.analyzer.compose_vqa(original, normalized, crop_path)
        final_dir = self.writer.stage_dir("07_final_vqa")
        final_payload = {
            "status": "executed",
            "requested_model": self.config.model,
            "executed_model": final_meta["executed_model"],
            "lineage": [
                "00_question",
                "01_evidence",
                "02_region_proposal",
                "03_sam2",
                "04_dam:fallback",
                "05_normalization",
                "06_sam3:blocked",
            ],
            "output": final_vqa.model_dump(mode="json"),
            "api": final_meta,
        }
        self.writer.write_json(final_dir / "result.json", final_payload)
        stages["07_final_vqa"] = final_payload

        manifest = {
            "schema_version": "0.1.0-question-trace",
            "video_id": self.config.video_id,
            "source_pilot": str(self.config.pilot_root.relative_to(PROJECT_ROOT)),
            "stages": stages,
        }
        self.writer.write_json(self.config.output_root / "trace.json", manifest)
        self.writer.write_report(manifest)
        return self.config.output_root / "report.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one question-conditioned VQA trace")
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_TRACE_ROOT / "0_te1_object")
    parser.add_argument("--video-id", default="0_te1")
    parser.add_argument("--category", default="scene.object_attribute_recognition")
    parser.add_argument("--model", default=None)
    parser.add_argument("--sam2-python", type=Path, default=PROJECT_ROOT / ".venv-sam2/bin/python")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-6.1-terra")
    config = TraceConfig(
        pilot_root=args.pilot_root.resolve(),
        output_root=args.output_root.resolve(),
        video_id=args.video_id,
        category=args.category,
        model=model,
        # Do not resolve this symlink: resolving it escapes the virtual environment.
        sam2_python=args.sam2_python.absolute(),
        sam2_worker=PROJECT_ROOT / "workers" / "sam2_segment.py",
    )
    pipeline = QuestionTracePipeline(
        config=config,
        records=VQARecordStore(config.pilot_root),
        analyzer=OpenAITraceAnalyzer(api_key, model),
        sam2=Sam2Runner(config.sam2_python, config.sam2_worker),
        writer=TraceArtifactWriter(config.output_root),
    )
    report = pipeline.run()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
