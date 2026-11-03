from __future__ import annotations

import argparse
import base64
import html
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from .discovery_models import (
    DiscoveryAnnotation,
    DiscoveryResult,
    FrameSemantics,
    VQACollection,
)
from .models import CATEGORY_ORDER


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_ROOT = PROJECT_ROOT / "vqa_datasets" / "pilot_5"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "vqa_datasets" / "discovery_mvp" / "0_te1"
T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class DiscoveryConfig:
    pilot_root: Path
    output_root: Path
    video_id: str
    model: str
    sam2_python: Path
    sam2_worker: Path
    frame_limit: int
    points_per_crop: int
    pred_iou_thresh: float
    stability_score_thresh: float


class DiscoverySource:
    def __init__(self, pilot_root: Path, video_id: str, frame_limit: int) -> None:
        self.pilot_root = pilot_root
        self.video_id = video_id
        self.frame_limit = frame_limit
        annotation_path = pilot_root / "annotations" / f"{video_id}.json"
        self.annotation = json.loads(annotation_path.read_text(encoding="utf-8"))

    @property
    def duration(self) -> float:
        return float(self.annotation["source"]["duration_seconds"])

    @property
    def frames(self) -> list[dict[str, Any]]:
        result = []
        for frame in self.annotation["source"]["sampled_frames"][: self.frame_limit]:
            result.append(
                {
                    "index": int(frame["index"]),
                    "timestamp_seconds": float(frame["timestamp_seconds"]),
                    "path": str((self.pilot_root / frame["path"]).resolve()),
                }
            )
        return result

    @property
    def review_video(self) -> Path:
        return self.pilot_root / self.annotation["source"]["review_video_path"]


class Sam2DiscoveryRunner:
    def __init__(self, python_path: Path, worker_path: Path) -> None:
        self.python_path = python_path
        self.worker_path = worker_path

    def run(
        self,
        input_manifest: Path,
        output_dir: Path,
        points_per_crop: int,
        pred_iou_thresh: float,
        stability_score_thresh: float,
    ) -> dict[str, Any]:
        environment = os.environ.copy()
        environment.update(
            {
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        command = [
            str(self.python_path),
            str(self.worker_path),
            "--manifest",
            str(input_manifest),
            "--output-dir",
            str(output_dir),
            "--points-per-crop",
            str(points_per_crop),
            "--pred-iou-thresh",
            str(pred_iou_thresh),
            "--stability-score-thresh",
            str(stability_score_thresh),
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
            detail = (exc.stderr or exc.stdout or "no worker output")[-8000:]
            raise RuntimeError(f"SAM 2 discovery worker failed: {detail}") from exc
        result_path = output_dir / "result.json"
        if not result_path.exists():
            raise RuntimeError(f"SAM 2 discovery returned no result: {completed.stdout}")
        return json.loads(result_path.read_text(encoding="utf-8"))


class DiscoveryArtifactWriter:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def crop_sheet(frame_dir: Path, proposals: list[dict[str, Any]]) -> Path:
        columns = 4
        cell_width, cell_height = 190, 145
        rows = max(1, (len(proposals) + columns - 1) // columns)
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#20252b")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()
        for position, proposal in enumerate(proposals):
            crop = Image.open(frame_dir / proposal["crop_path"]).convert("RGB")
            crop.thumbnail((cell_width - 12, cell_height - 30))
            x = (position % columns) * cell_width + 6
            y = (position // columns) * cell_height + 24
            sheet.paste(crop, (x, y))
            draw.text((x, y - 18), proposal["region_id"], fill="white", font=font)
        path = frame_dir / "all_crops_contact_sheet.jpg"
        sheet.save(path, quality=92)
        return path

    def write_report(self, manifest: dict[str, Any]) -> Path:
        sam = manifest["stages"]["01_sam2_discovery"]["output"]
        descriptions = manifest["stages"]["02_region_description"]["output"]
        description_by_id = {
            region["region_id"]: region
            for frame in descriptions
            for region in frame["regions"]
        }
        frame_sections = []
        for frame in sam["frames"]:
            frame_index = int(frame["frame_index"])
            proposal_rows = []
            for proposal in frame["proposals"]:
                region = description_by_id.get(proposal["region_id"], {})
                proposal_rows.append(
                    f"""<article class="proposal">
                    <img src="01_sam2/frame_{frame_index:02d}/{html.escape(proposal['crop_path'])}">
                    <b>{html.escape(proposal['region_id'])}</b>
                    <span>{html.escape(region.get('object_name', '설명 없음'))}</span>
                    <small>score={proposal['score']} · bbox={html.escape(str(proposal['bbox_xyxy']))}</small>
                    <small>{html.escape(region.get('description', ''))}</small>
                    </article>"""
                )
            frame_sections.append(
                f"""<section class="card">
                <h2>Frame {frame_index} · {frame['timestamp_seconds']:.3f}s · {frame['retained_proposal_count']} masks</h2>
                <div class="compare">
                  <figure><img src="01_sam2/frame_{frame_index:02d}/all_bboxes.jpg"><figcaption>모든 SAM2 bbox + ID</figcaption></figure>
                  <figure><img src="01_sam2/frame_{frame_index:02d}/all_masks_overlay.png"><figcaption>모든 SAM2 mask overlay</figcaption></figure>
                </div>
                <details><summary>모든 region crop 및 설명 ({len(proposal_rows)})</summary><div class="grid">{''.join(proposal_rows)}</div></details>
                </section>"""
            )

        annotation = manifest["stages"]["03_normalization"]["output"]
        vqa = manifest["stages"]["05_vqa"]["output"]["vqa"]
        vqa_rows = "".join(
            f"<tr><td>{html.escape(item['category'])}</td><td>{html.escape(item['question'])}</td><td>{html.escape(item['answer'])}</td><td>{html.escape(str(item['evidence']['frame_indices']))}</td></tr>"
            for item in vqa
        )
        object_rows = "".join(
            f"<tr><td>{html.escape(obj['entity_id'])}</td><td>{html.escape(obj['canonical_name'])}</td><td>{html.escape(', '.join(obj['sam_region_ids']))}</td><td>{html.escape(', '.join(obj['attributes']))}</td></tr>"
            for obj in annotation["object_annotations"]
        )
        document = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SAM2 Discovery-first VQA MVP</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f3f5f7;color:#17202a;max-width:1500px;margin:auto;padding:24px}}
.card{{background:white;border-radius:14px;padding:20px;margin:0 0 22px;box-shadow:0 2px 14px #0001}} .hero{{border-left:8px solid #2471a3}}
video{{width:min(100%,960px);background:#111}} .compare{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}}
figure{{margin:0}} figure img{{width:100%;max-height:560px;object-fit:contain;background:#111}} figcaption{{color:#566573}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;margin-top:15px}} .proposal{{border:1px solid #ccd1d1;border-radius:8px;padding:8px;display:flex;flex-direction:column;gap:4px}}
.proposal img{{width:100%;height:130px;object-fit:contain;background:#111}} small{{color:#566573;word-break:break-word}} table{{width:100%;border-collapse:collapse}} th,td{{border:1px solid #d5d8dc;padding:8px;text-align:left;vertical-align:top}} th{{background:#eef2f3}} pre{{white-space:pre-wrap;word-break:break-word;background:#f8f9f9;padding:14px}}
.ok{{color:#1e8449}} .fallback{{color:#b9770e}} .blocked{{color:#c0392b}}
</style></head><body>
<section class="card hero"><h1>Discovery-first VQA MVP · {html.escape(manifest['video_id'])}</h1>
<p><b>SAM2:</b> 8개 프레임에서 원시/유지 proposal {sam['total_raw_proposals']}/{sam['total_retained_proposals']}개. SAM2 내장 score·stability·NMS를 통과한 결과를 추가 cap 없이 전부 표시한다.</p>
<p><span class="ok">SAM2 실제 실행</span> · <span class="fallback">DAM 단계는 Terra fallback</span> · <span class="blocked">SAM3는 미실행</span></p>
<video controls preload="metadata" src="source_video.mp4"></video></section>
<section class="card"><h2>실제 데이터 흐름</h2><pre>영상 → 균등 샘플 프레임 → SAM2 자동 전체 mask proposal → 모든 bbox/ID/crop → 객체별 설명(DAM fallback) → LLM 객체 통합/표준화 → 7개 VQA</pre></section>
{''.join(frame_sections)}
<section class="card"><h2>프레임 간 객체 표준화 결과</h2><p>{html.escape(annotation['scene_summary'])}</p><table><thead><tr><th>Entity</th><th>표준 이름</th><th>근거 SAM region IDs</th><th>속성</th></tr></thead><tbody>{object_rows}</tbody></table><details><summary>전체 annotation JSON</summary><pre>{html.escape(json.dumps(annotation, ensure_ascii=False, indent=2))}</pre></details></section>
<section class="card"><h2>최종 VQA</h2><table><thead><tr><th>Category</th><th>Question</th><th>Answer</th><th>Evidence frames</th></tr></thead><tbody>{vqa_rows}</tbody></table><details><summary>전체 실행 manifest</summary><pre>{html.escape(json.dumps(manifest, ensure_ascii=False, indent=2))}</pre></details></section>
</body></html>"""
        path = self.output_root / "report.html"
        path.write_text(document, encoding="utf-8")
        return path


class OpenAIDiscoveryAnalyzer:
    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, timeout=240.0, max_retries=2)

    @staticmethod
    def _data_url(path: Path) -> str:
        media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

    def _parse(
        self,
        schema: type[T],
        instructions: str,
        content: list[dict[str, Any]] | str,
        max_output_tokens: int = 6000,
    ) -> tuple[T, dict[str, Any]]:
        response = self.client.responses.parse(
            model=self.model,
            instructions=instructions,
            input=[{"role": "user", "content": content}],
            text_format=schema,
            max_output_tokens=max_output_tokens,
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

    def describe_frame(
        self,
        frame: dict[str, Any],
        frame_dir: Path,
        contact_sheet: Path,
    ) -> tuple[FrameSemantics, dict[str, Any]]:
        compact_regions = [
            {
                "region_id": proposal["region_id"],
                "bbox_xyxy": proposal["bbox_xyxy"],
                "sam_score": proposal["score"],
            }
            for proposal in frame["proposals"]
        ]
        prompt = f"""SAM2가 자동 생성한 모든 유지 region을 설명하세요.
각 region_id를 정확히 한 번씩 출력해야 하며 빠뜨리거나 새 ID를 만들면 안 됩니다.
SAM2는 의미 이름이 없는 분할 모델이므로, bbox 이미지와 crop sheet를 보고 이름·속성·상태를 붙이세요.
겹치는 영역이나 객체 일부도 독립 region으로 설명하고, 모호하면 object_name을 '불명확한 영역'으로 쓰세요.

frame_index={frame['frame_index']}, timestamp={frame['timestamp_seconds']}
regions={json.dumps(compact_regions, ensure_ascii=False)}
"""
        result, meta = self._parse(
            FrameSemantics,
            "산업 CCTV의 SAM2 region을 보이는 근거만으로 설명하는 annotator입니다. 행동 의도나 위험을 추론하지 마세요.",
            [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": self._data_url(frame_dir / "all_bboxes.jpg"), "detail": "high"},
                {"type": "input_image", "image_url": self._data_url(contact_sheet), "detail": "high"},
            ],
            max_output_tokens=9000,
        )
        expected = {item["region_id"] for item in compact_regions}
        actual = {item.region_id for item in result.regions}
        if expected != actual or len(result.regions) != len(expected):
            raise ValueError(
                f"Region coverage mismatch for frame {frame['frame_index']}: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        if result.frame_index != int(frame["frame_index"]):
            raise ValueError("Frame semantics index mismatch")
        return result, meta

    def normalize(
        self,
        video_id: str,
        duration: float,
        semantics: list[FrameSemantics],
    ) -> tuple[DiscoveryAnnotation, dict[str, Any]]:
        payload = [item.model_dump(mode="json") for item in semantics]
        prompt = f"""video_id={video_id}, duration={duration:.3f}s.
아래는 SAM2가 발견한 모든 유지 region에 대한 프레임별 설명입니다.
같은 실제 객체로 보이는 region들을 하나의 entity로 묶고 표준 이름과 속성을 만드세요.
모든 object annotation은 근거 sam_region_ids를 포함해야 합니다.
이것은 진짜 tracking이 아니라 샘플 프레임 사이의 LLM 시각 연관이므로 association_note에 그 한계를 명시하세요.
Risk Reasoning은 하지 마세요. 관찰되지 않은 행동·완료·복구를 추정하지 마세요.

{json.dumps(payload, ensure_ascii=False)}
"""
        return self._parse(
            DiscoveryAnnotation,
            "SAM2 region 설명을 영상 수준의 검증 가능한 산업 CCTV annotation으로 표준화하세요.",
            prompt,
            max_output_tokens=9000,
        )

    def compose_vqa(
        self,
        video_id: str,
        duration: float,
        annotation: DiscoveryAnnotation,
        timestamps: dict[int, float],
    ) -> tuple[VQACollection, dict[str, Any]]:
        prompt = f"""video_id={video_id}, duration={duration:.3f}s.
아래 구조화 annotation만 근거로 정확히 7개 한국어 VQA를 만드세요.
각 category를 정확히 한 번 사용하세요: {json.dumps(CATEGORY_ORDER, ensure_ascii=False)}
Risk Reasoning은 제외합니다. 질문·답변은 짧게 쓰고 근거 frame index와 정확한 timestamp를 함께 넣으세요.
최종 사용자가 내부 식별자를 몰라도 이해할 수 있도록 질문과 답변에 E01, F00_M001 같은 entity/region ID를 노출하지 마세요.
대신 '파란색 기계 덮개', '오른쪽 통로의 사람'처럼 자연스러운 객체 이름과 위치를 사용하세요.
진짜 연속 tracking이 아닌 샘플 프레임 기반이라는 한계를 지키고, 불확실한 시간/완료 상태를 단정하지 마세요.
frame timestamps={json.dumps(timestamps, ensure_ascii=False)}
annotation={json.dumps(annotation.model_dump(mode='json'), ensure_ascii=False)}
"""
        result, meta = self._parse(
            VQACollection,
            "구조화된 SAM2 기반 annotation에서 근거 추적 가능한 VQA를 생성하세요.",
            prompt,
            max_output_tokens=6500,
        )
        valid_indices = set(timestamps)
        for item in result.vqa:
            if not set(item.evidence.frame_indices).issubset(valid_indices):
                raise ValueError(f"Invalid evidence frame in {item.category}")
            expected_times = [timestamps[index] for index in item.evidence.frame_indices]
            if any(abs(a - b) > 0.01 for a, b in zip(expected_times, item.evidence.timestamps_seconds, strict=True)):
                raise ValueError(f"Timestamp mismatch in {item.category}")
        return result, meta


class DiscoveryFirstPipeline:
    def __init__(
        self,
        config: DiscoveryConfig,
        source: DiscoverySource,
        sam2: Sam2DiscoveryRunner,
        analyzer: OpenAIDiscoveryAnalyzer,
        writer: DiscoveryArtifactWriter,
    ) -> None:
        self.config = config
        self.source = source
        self.sam2 = sam2
        self.analyzer = analyzer
        self.writer = writer

    @staticmethod
    def validate_region_coverage(
        frame_semantics: list[FrameSemantics],
        annotation: DiscoveryAnnotation,
    ) -> None:
        discovered_ids = {
            region.region_id for frame in frame_semantics for region in frame.regions
        }
        normalized_ids = [
            region_id
            for item in annotation.object_annotations
            for region_id in item.sam_region_ids
        ]
        if set(normalized_ids) != discovered_ids or len(normalized_ids) != len(discovered_ids):
            raise ValueError(
                "Normalization must use every discovered SAM region exactly once: "
                f"missing={sorted(discovered_ids - set(normalized_ids))}, "
                f"duplicates={len(normalized_ids) - len(set(normalized_ids))}"
            )

    def run(self) -> Path:
        input_dir = self.config.output_root / "00_input"
        sam_dir = self.config.output_root / "01_sam2"
        description_dir = self.config.output_root / "02_region_descriptions"
        normalization_dir = self.config.output_root / "03_normalization"
        sam3_dir = self.config.output_root / "04_sam3"
        vqa_dir = self.config.output_root / "05_vqa"
        input_manifest = {"video_id": self.config.video_id, "frames": self.source.frames}
        self.writer.write_json(input_dir / "manifest.json", input_manifest)
        shutil.copy2(self.source.review_video, self.config.output_root / "source_video.mp4")

        sam_result_path = sam_dir / "result.json"
        if sam_result_path.exists():
            sam_result = json.loads(sam_result_path.read_text(encoding="utf-8"))
        else:
            sam_result = self.sam2.run(
                input_dir / "manifest.json",
                sam_dir,
                self.config.points_per_crop,
                self.config.pred_iou_thresh,
                self.config.stability_score_thresh,
            )

        frame_semantics: list[FrameSemantics] = []
        description_api = []
        for frame in sam_result["frames"]:
            frame_dir = sam_dir / f"frame_{int(frame['frame_index']):02d}"
            contact_sheet = self.writer.crop_sheet(frame_dir, frame["proposals"])
            semantics_path = description_dir / f"frame_{int(frame['frame_index']):02d}.json"
            if semantics_path.exists():
                semantics = FrameSemantics.model_validate_json(
                    semantics_path.read_text(encoding="utf-8")
                )
                api = {"resumed_from_existing_artifact": True}
            else:
                semantics, api = self.analyzer.describe_frame(frame, frame_dir, contact_sheet)
            frame_semantics.append(semantics)
            description_api.append(api)
            self.writer.write_json(
                semantics_path,
                semantics.model_dump(mode="json"),
            )

        normalization_path = normalization_dir / "result.json"
        if normalization_path.exists():
            annotation = DiscoveryAnnotation.model_validate_json(
                normalization_path.read_text(encoding="utf-8")
            )
            normalization_api = {
                "executed_model": self.config.model,
                "resumed_from_existing_artifact": True,
            }
        else:
            annotation, normalization_api = self.analyzer.normalize(
                self.config.video_id, self.source.duration, frame_semantics
            )
            self.writer.write_json(normalization_path, annotation.model_dump(mode="json"))

        self.validate_region_coverage(frame_semantics, annotation)

        sam3_payload = {
            "status": "blocked",
            "requested_model": "facebook/sam3.1",
            "executed_model": None,
            "blocked_reason": "CUDA 12.6 compatible GPU and approved checkpoint access are unavailable on this Apple Silicon host.",
            "verification_result": "not_verified",
        }
        self.writer.write_json(sam3_dir / "result.json", sam3_payload)

        timestamps = {
            int(frame["index"]): float(frame["timestamp_seconds"])
            for frame in self.source.frames
        }
        vqa, vqa_api = self.analyzer.compose_vqa(
            self.config.video_id, self.source.duration, annotation, timestamps
        )
        final = DiscoveryResult(
            annotation=annotation,
            vqa=vqa.vqa,
            annotation_source="sam2_regions_plus_visual_descriptions",
        )
        self.writer.write_json(vqa_dir / "result.json", final.model_dump(mode="json"))

        stages = {
            "00_input": {
                "status": "executed",
                "requested_model": None,
                "executed_model": "existing ffmpeg sampled frames",
                "output": input_manifest,
            },
            "01_sam2_discovery": {
                "status": "executed",
                "requested_model": sam_result["requested_model"],
                "executed_model": sam_result["executed_model"],
                "output": sam_result,
            },
            "02_region_description": {
                "status": "fallback",
                "requested_model": "nvidia/DAM-3B-Video",
                "executed_model": self.config.model,
                "fallback_is_not_dam_output": True,
                "blocked_reason": "DAM-3B-Video official local path requires supported NVIDIA/Linux hardware; Terra describes every SAM region for this MVP.",
                "output": [item.model_dump(mode="json") for item in frame_semantics],
                "api": description_api,
            },
            "03_normalization": {
                "status": "executed",
                "requested_model": self.config.model,
                "executed_model": normalization_api["executed_model"],
                "association_method": "llm_visual_association_not_true_tracking",
                "output": annotation.model_dump(mode="json"),
                "api": normalization_api,
            },
            "04_sam3": sam3_payload,
            "05_vqa": {
                "status": "executed",
                "requested_model": self.config.model,
                "executed_model": vqa_api["executed_model"],
                "input": "03_normalization structured annotation",
                "output": final.model_dump(mode="json"),
                "api": vqa_api,
            },
        }
        manifest = {
            "schema_version": "0.1.0-discovery-first-mvp",
            "video_id": self.config.video_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stages": stages,
        }
        self.writer.write_json(self.config.output_root / "manifest.json", manifest)
        return self.writer.write_report(manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a discovery-first SAM2 → description → VQA MVP")
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--video-id", default="0_te1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--sam2-python", type=Path, default=PROJECT_ROOT / ".venv-sam2/bin/python")
    parser.add_argument("--frame-limit", type=int, default=8)
    parser.add_argument("--points-per-crop", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.82)
    parser.add_argument("--stability-score-thresh", type=float, default=0.90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-6.1-terra")
    config = DiscoveryConfig(
        pilot_root=args.pilot_root.resolve(),
        output_root=args.output_root.resolve(),
        video_id=args.video_id,
        model=model,
        sam2_python=args.sam2_python.absolute(),
        sam2_worker=PROJECT_ROOT / "workers" / "sam2_discover.py",
        frame_limit=args.frame_limit,
        points_per_crop=args.points_per_crop,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
    )
    source = DiscoverySource(config.pilot_root, config.video_id, config.frame_limit)
    pipeline = DiscoveryFirstPipeline(
        config=config,
        source=source,
        sam2=Sam2DiscoveryRunner(config.sam2_python, config.sam2_worker),
        analyzer=OpenAIDiscoveryAnalyzer(api_key, model),
        writer=DiscoveryArtifactWriter(config.output_root),
    )
    print(pipeline.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
