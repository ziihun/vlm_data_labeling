from __future__ import annotations

import argparse
import base64
import html
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field


PROMPT_TEMPLATE = """You are performing fine-grained object naming for one SAM 2 Track from an industrial video.

Image 1 is a chronological contact sheet of the SAME segmentation target. Pixels outside the mask are darkened.
Image 2 is the matching chronological context sheet. A red box marks the target region.

Identify ONLY the segmented target, not a nearby, overlapping, or containing object. Compare all cells before answering.
Use the most specific English noun phrase that is consistently supported by visible shape, construction, markings, and scene context.
Prefer names such as "wall-mounted split air conditioner", "industrial conveyor belt", or "red-bordered storage sign" over vague names such as "object", "item", "block", "panel", or a color-only label.
Do not add a function, brand, safety meaning, or material unless it is directly visible. Transcribe legible text exactly; do not guess missing letters.
Record useful visible attributes when supported, such as color, shape, material_appearance, printed_text, pattern, mounting, orientation, and distinctive_parts. Omit unsupported attribute keys.
Describe only a directly observable state such as moving/stationary, open/closed, illuminated/unlit, occupied/empty, intact/damaged, or "unknown". Do not infer an action from a static appearance.
Do not infer gender, identity, intent, danger, causality, or off-screen facts.
If the target is too small, ambiguous, inconsistent across samples, or only a segmentation fragment, use "unknown" instead of guessing.

Calibrate confidence:
- 0.90-1.00 only when the same fine-grained identity is clear across multiple samples.
- 0.70-0.89 when the general identity is clear but fine details remain uncertain.
- below 0.70 when the identity is ambiguous; normally return "unknown".

Return one JSON object with exactly these keys:
- canonical_object_name: specific, visually grounded English noun phrase, or "unknown"
- visible_attributes: JSON object whose keys and string values are English
- observable_state: short English phrase describing a directly visible state, or "unknown"
- confidence: number from 0 to 1
- uncertainty_reason: short English explanation, empty only when identification is clear

Return valid JSON only. Do not use Markdown.
"""


MODEL_SPECS: dict[str, dict[str, str]] = {
    "qwen3_5_4b": {
        "model": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "kind": "local",
    },
    "qwen3_5_9b": {
        "model": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "kind": "local",
    },
    "gemma3_12b": {
        "model": "mlx-community/gemma-3-12b-it-4bit",
        "kind": "local",
    },
    "openai_terra": {
        "model": "gpt-6.1-terra",
        "kind": "api",
        "score": "self-reported confidence",
    },
    "ram_plus": {
        "model": "RAM++ Swin-Large 14M",
        "kind": "tagger",
        "score": "support ratio across Track crops",
    },
    "siglip2": {
        "model": "google/siglip2-base-patch16-256",
        "kind": "zero-shot classifier",
        "score": "mean sigmoid similarity over Track crops",
    },
    "yoloe_prompt_free": {
        "model": "yoloe-26s-seg-pf.pt",
        "kind": "open-vocabulary detector",
        "score": "detector confidence after SAM2 mask match",
    },
    "internvl3_5_4b": {
        "model": "OpenGVLab/InternVL3_5-4B-hf",
        "kind": "local VLM",
        "score": "self-reported confidence",
    },
    "molmo2_4b": {
        "model": "allenai/Molmo2-4B",
        "kind": "local VLM",
        "score": "self-reported confidence",
    },
    "minicpm_v4": {
        "model": "openbmb/MiniCPM-V-4",
        "kind": "local VLM",
        "score": "self-reported confidence",
    },
}

for _spec in MODEL_SPECS.values():
    _spec.setdefault("score", "self-reported confidence")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NamingAnswer(StrictModel):
    canonical_object_name: str
    visible_attributes: dict[str, str]
    observable_state: str
    confidence: float = Field(ge=0, le=1)
    uncertainty_reason: str


def _fit_image(source: Image.Image, size: tuple[int, int]) -> Image.Image:
    source = source.convert("RGB")
    ratio = min(size[0] / source.width, size[1] / source.height)
    resized = source.resize(
        (max(1, round(source.width * ratio)), max(1, round(source.height * ratio))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, (20, 20, 20))
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return canvas


def create_contact_sheet(
    image_paths: list[Path],
    labels: list[str],
    output_path: Path,
    *,
    cell_size: tuple[int, int] = (384, 384),
    columns: int = 4,
) -> None:
    if not image_paths:
        raise ValueError("A contact sheet requires at least one image")
    rows = (len(image_paths) + columns - 1) // columns
    label_height = 34
    sheet = Image.new(
        "RGB",
        (cell_size[0] * columns, (cell_size[1] + label_height) * rows),
        (12, 12, 12),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    for index, (path, label) in enumerate(zip(image_paths, labels, strict=True)):
        row, column = divmod(index, columns)
        x, y = column * cell_size[0], row * (cell_size[1] + label_height)
        with Image.open(path) as source:
            cell = _fit_image(source, cell_size)
        sheet.paste(cell, (x, y))
        draw.rectangle((x, y + cell_size[1], x + cell_size[0], y + cell_size[1] + label_height), fill=(35, 35, 35))
        draw.text((x + 10, y + cell_size[1] + 5), label, fill=(245, 245, 245), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def prepare_assets(
    crops_json: Path,
    experiment_dir: Path,
    tracks_json: Path | None = None,
) -> dict[str, Any]:
    crops = json.loads(crops_json.read_text(encoding="utf-8"))
    if tracks_json is None:
        candidate = crops_json.parent.parent / "01_sam2" / "tracks.json"
        tracks_json = candidate if candidate.exists() else None
    source_by_track_timestamp: dict[tuple[str, float], str] = {}
    if tracks_json is not None and tracks_json.exists():
        source_tracks = json.loads(tracks_json.read_text(encoding="utf-8"))
        for source_track in source_tracks["tracks"]:
            for observation in source_track["observations"]:
                source_by_track_timestamp[(
                    source_track["track_id"],
                    round(float(observation["source_timestamp_seconds"]), 6),
                )] = observation["source_frame_path"]
    assets_dir = experiment_dir / "assets"
    tracks = []
    for track in crops["tracks"]:
        evidence = []
        for item in track["evidence"]:
            enriched = dict(item)
            source_path = source_by_track_timestamp.get((
                track["track_id"], round(float(item["source_timestamp_seconds"]), 6)
            ))
            if source_path:
                enriched["source_frame_path"] = source_path
            evidence.append(enriched)
        labels = [f"sample {item['evidence_index'] + 1}" for item in evidence]
        masked_path = assets_dir / f"{track['track_id']}_masked_grid.jpg"
        context_path = assets_dir / f"{track['track_id']}_context_grid.jpg"
        create_contact_sheet([Path(item["masked_crop_path"]) for item in evidence], labels, masked_path)
        create_contact_sheet([Path(item["context_crop_path"]) for item in evidence], labels, context_path)
        tracks.append({
            "track_id": track["track_id"],
            "first_seen_seconds": track["first_seen_seconds"],
            "last_seen_seconds": track["last_seen_seconds"],
            "masked_grid_path": str(masked_path.resolve()),
            "context_grid_path": str(context_path.resolve()),
            "evidence": evidence,
        })
    payload = {
        "source_crops_json": str(crops_json.resolve()),
        "source_tracks_json": str(tracks_json.resolve()) if tracks_json else None,
        "prompt": PROMPT_TEMPLATE,
        "tracks": tracks,
    }
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "assets.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def run_openai(assets_json: Path, output_dir: Path, model: str) -> dict[str, Any]:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from .env")
    assets = json.loads(assets_json.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.txt").write_text(PROMPT_TEMPLATE, encoding="utf-8")
    client = OpenAI()
    items: list[dict[str, Any]] = []
    raw_items: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    started = time.perf_counter()
    for index, track in enumerate(assets["tracks"], start=1):
        track_started = time.perf_counter()
        response = client.responses.parse(
            model=model,
            instructions="Follow the image-grounded object naming specification exactly.",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT_TEMPLATE},
                    {"type": "input_image", "image_url": _data_url(Path(track["masked_grid_path"])), "detail": "high"},
                    {"type": "input_image", "image_url": _data_url(Path(track["context_grid_path"])), "detail": "high"},
                ],
            }],
            text_format=NamingAnswer,
            max_output_tokens=800,
            store=False,
        )
        elapsed = time.perf_counter() - track_started
        parsed = response.output_parsed
        parse_error = None
        if parsed is None:
            parsed = NamingAnswer(
                canonical_object_name="unknown",
                visible_attributes={},
                observable_state="unknown",
                confidence=0.0,
                uncertainty_reason="API response had no parsed output",
            )
            parse_error = "OpenAI response.output_parsed is None"
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        item = {
            "track_id": track["track_id"],
            **parsed.model_dump(mode="json"),
            "latency_seconds": round(elapsed, 3),
            "parse_error": parse_error,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
        items.append(item)
        raw_items.append({
            "track_id": track["track_id"],
            "response": response.model_dump(mode="json"),
        })
        print(json.dumps({
            "progress": f"{index}/{len(assets['tracks'])}",
            "track_id": track["track_id"],
            "name": item["canonical_object_name"],
            "latency_seconds": item["latency_seconds"],
        }, ensure_ascii=False), flush=True)
    estimated_cost_usd = total_input_tokens / 1_000_000 + total_output_tokens * 5 / 1_000_000
    payload = {
        "model": model,
        "runner": "openai_responses_api",
        "input_policy": "two 2x4 contact sheets per SAM2 Track",
        "items": items,
    }
    metrics = {
        "status": "completed",
        "track_count": len(items),
        "total_latency_seconds": round(time.perf_counter() - started, 3),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "estimated_cost_usd": round(estimated_cost_usd, 6),
        "pricing_note": "Estimate uses gpt-6.1-terra $1/M input and $5/M output tokens; verify current pricing before a large run.",
    }
    (output_dir / "raw_response.json").write_text(json.dumps(raw_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "naming.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _load_result(experiment_dir: Path, key: str) -> dict[str, Any]:
    model_dir = experiment_dir / key
    naming_path = model_dir / "naming.json"
    metrics_path = model_dir / "metrics.json"
    if naming_path.exists():
        naming = json.loads(naming_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        return {"status": metrics.get("status", "completed"), "naming": naming, "metrics": metrics}
    failure_path = model_dir / "failure.json"
    if failure_path.exists():
        return {"status": "failed", "failure": json.loads(failure_path.read_text(encoding="utf-8"))}
    return {"status": "not_run"}


def build_comparison(experiment_dir: Path) -> dict[str, Any]:
    assets = json.loads((experiment_dir / "assets.json").read_text(encoding="utf-8"))
    results = {key: _load_result(experiment_dir, key) for key in MODEL_SPECS}
    by_model: dict[str, dict[str, Any]] = {}
    for key, result in results.items():
        if result["status"] == "completed":
            by_model[key] = {item["track_id"]: item for item in result["naming"]["items"]}
    tracks = []
    for asset in assets["tracks"]:
        tracks.append({
            "track_id": asset["track_id"],
            "masked_grid_path": asset["masked_grid_path"],
            "context_grid_path": asset["context_grid_path"],
            "predictions": {key: by_model.get(key, {}).get(asset["track_id"]) for key in MODEL_SPECS},
        })
    summary: dict[str, Any] = {}
    for key, spec in MODEL_SPECS.items():
        result = results[key]
        items = list(by_model.get(key, {}).values())
        summary[key] = {
            "model": spec["model"],
            "kind": spec["kind"],
            "status": result["status"],
            "track_count": len(items),
            "unknown_count": sum(item["canonical_object_name"].strip().lower() == "unknown" for item in items),
            "parse_error_count": sum(bool(item.get("parse_error")) for item in items),
            "mean_confidence": round(sum(float(item["confidence"]) for item in items) / len(items), 3) if items else None,
            "mean_latency_seconds": round(sum(float(item.get("latency_seconds", 0)) for item in items) / len(items), 3) if items else None,
            "metrics": result.get("metrics", {}),
            "failure": result.get("failure"),
        }
    payload = {"models": MODEL_SPECS, "summary": summary, "tracks": tracks}
    (experiment_dir / "comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_report(payload, experiment_dir)
    return payload


def _relative(path: str, base: Path) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


def render_report(comparison: dict[str, Any], experiment_dir: Path) -> None:
    summary_cards = []
    for key, item in comparison["summary"].items():
        cost = item["metrics"].get("estimated_cost_usd")
        cost_text = f" · API estimate ${cost:.6f}" if cost is not None else ""
        summary_cards.append(
            f'<section class="summary"><h3>{html.escape(key)}</h3>'
            f'<div class="model">{html.escape(item["model"])}</div>'
            f'<p>Status <b>{html.escape(item["status"])}</b> · Tracks {item["track_count"]} · '
            f'Unknown {item["unknown_count"]} · Parse errors {item["parse_error_count"]} · '
            f'Mean confidence {item["mean_confidence"]} · Mean latency {item["mean_latency_seconds"]}s{cost_text}</p>'
            f'{"<pre>" + html.escape(json.dumps(item.get("failure"), ensure_ascii=False, indent=2)) + "</pre>" if item.get("failure") else ""}'
            '</section>'
        )
    track_sections = []
    for track in comparison["tracks"]:
        prediction_cards = []
        for key, spec in MODEL_SPECS.items():
            item = track["predictions"].get(key)
            if item is None:
                body = '<p class="missing">결과 없음</p>'
            else:
                attrs = html.escape(json.dumps(item.get("visible_attributes", {}), ensure_ascii=False, indent=2))
                uncertainty = html.escape(item.get("uncertainty_reason", ""))
                parse_error = html.escape(str(item.get("parse_error") or ""))
                body = (
                    f'<h4>{html.escape(item["canonical_object_name"])}</h4>'
                    f'<p>State: {html.escape(item["observable_state"])}<br>'
                    f'Confidence: {item["confidence"]} · Latency: {item.get("latency_seconds", "-")}s</p>'
                    f'<pre>{attrs}</pre><p>Uncertainty: {uncertainty or "—"}</p>'
                    f'<p>Top candidates: {html.escape(json.dumps(item.get("top_candidates", []), ensure_ascii=False))}</p>'
                    f'{f"<p class=error>Parse error: {parse_error}</p>" if parse_error else ""}'
                )
            prediction_cards.append(f'<article class="prediction"><div class="tag">{html.escape(key)}</div>{body}</article>')
        masked_src = html.escape(_relative(track["masked_grid_path"], experiment_dir))
        context_src = html.escape(_relative(track["context_grid_path"], experiment_dir))
        options = ''.join(f'<option value="{key}">{key}</option>' for key in MODEL_SPECS)
        track_sections.append(
            f'<section class="track" data-track="{html.escape(track["track_id"])}">'
            f'<h2>{html.escape(track["track_id"])}</h2>'
            f'<div class="images"><figure><img src="{masked_src}"><figcaption>Masked target</figcaption></figure>'
            f'<figure><img src="{context_src}"><figcaption>Context + BBox</figcaption></figure></div>'
            f'<div class="predictions">{"".join(prediction_cards)}</div>'
            f'<div class="review"><label>Best model <select class="best"><option value="none">none</option>{options}</select></label>'
            f'<label>Corrected name <input class="corrected" placeholder="human verified name"></label>'
            f'<label>Notes <input class="notes" placeholder="why / ambiguity"></label></div></section>'
        )
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAM2 Naming Model Comparison</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#11151a;color:#edf2f7}}main{{max-width:1500px;margin:auto;padding:24px}}
h1{{margin-bottom:4px}}.lead{{color:#aeb9c5}}.summaries,.predictions,.images{{display:grid;gap:14px}}.summaries{{grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}}
.summary,.track,.prediction{{background:#1b222b;border:1px solid #33404d;border-radius:12px;padding:16px}}.model,.tag{{color:#7dd3fc;font:13px ui-monospace,monospace}}
.track{{margin-top:22px}}.images{{grid-template-columns:1fr 1fr}}figure{{margin:0}}img{{width:100%;display:block;border-radius:8px}}figcaption{{color:#aeb9c5;margin-top:6px}}
.predictions{{grid-template-columns:repeat(4,1fr);margin-top:14px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#11151a;padding:10px;border-radius:7px}}
.missing,.error{{color:#fca5a5}}.review{{display:grid;grid-template-columns:1fr 2fr 3fr;gap:12px;margin-top:14px}}label{{display:grid;gap:5px;color:#aeb9c5}}
input,select,button{{font:inherit;border:1px solid #526171;border-radius:7px;padding:9px;background:#11151a;color:#fff}}button{{position:sticky;bottom:16px;margin-top:24px;background:#0369a1;cursor:pointer}}
@media(max-width:1050px){{.predictions{{grid-template-columns:1fr 1fr}}}}@media(max-width:700px){{.images,.predictions,.review{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>SAM2 Naming Model Comparison</h1>
<p class="lead">동일한 SAM2 Track을 비교합니다. VLM은 동일한 2×4 Mask/Context 시트와 동일 프롬프트를 사용합니다. RAM++는 Track crop 태그 빈도, SigLIP2는 고정 후보 사전 유사도, YOLOE는 전체 프레임 탐지 후 SAM2 Mask 매칭을 사용합니다. 서로 다른 score는 직접 비교할 수 없으며 영상을 보고 검수해야 합니다.</p>
<div class="summaries">{"".join(summary_cards)}</div>{"".join(track_sections)}
<button id="download">Human evaluation JSON 다운로드</button></main>
<script>document.getElementById('download').onclick=()=>{{const items=[...document.querySelectorAll('.track')].map(x=>({{track_id:x.dataset.track,best_model:x.querySelector('.best').value,corrected_name:x.querySelector('.corrected').value.trim(),notes:x.querySelector('.notes').value.trim()}}));const blob=new Blob([JSON.stringify({{review_status:'completed',items}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='naming_evaluation.json';a.click();URL.revokeObjectURL(a.href)}};</script>
</body></html>"""
    (experiment_dir / "report.html").write_text(document, encoding="utf-8")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Fair naming benchmark over saved SAM2 Track crops")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--crops-json", type=Path, required=True)
    prepare_parser.add_argument("--experiment-dir", type=Path, required=True)
    prepare_parser.add_argument("--tracks-json", type=Path)
    openai_parser = subparsers.add_parser("openai")
    openai_parser.add_argument("--assets-json", type=Path, required=True)
    openai_parser.add_argument("--output-dir", type=Path, required=True)
    openai_parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-6.1-terra"))
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_assets(args.crops_json, args.experiment_dir, args.tracks_json)
    elif args.command == "openai":
        try:
            run_openai(args.assets_json, args.output_dir, args.model)
        except Exception as exc:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "status": "failed",
                "phase": "api_request",
                "model": args.model,
                "error_type": type(exc).__name__,
                "error": "OpenAI request failed. Check API key validity, model access, billing, and network.",
            }
            (args.output_dir / "failure.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            raise
    else:
        build_comparison(args.experiment_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
