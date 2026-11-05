from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image


def parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def normalize(parsed: dict[str, Any]) -> dict[str, Any]:
    attributes = parsed.get("visible_attributes") or {}
    if not isinstance(attributes, dict):
        attributes = {"description": str(attributes)}
    return {
        "canonical_object_name": str(parsed.get("canonical_object_name") or "unknown").strip(),
        "visible_attributes": {str(key): str(value) for key, value in attributes.items()},
        "observable_state": str(parsed.get("observable_state") or "unknown").strip(),
        "confidence": min(1.0, max(0.0, float(parsed.get("confidence", 0.0)))),
        "uncertainty_reason": str(parsed.get("uncertainty_reason") or "").strip(),
    }


def extract_generated_text(result: Any) -> str:
    value = result
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, dict):
        value = value.get("generated_text", value.get("text", value))
    if isinstance(value, list):
        for message in reversed(value):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = [item.get("text", "") for item in content if isinstance(item, dict)]
                    return "\n".join(text for text in texts if text)
        return str(value[-1]) if value else ""
    return str(value)


def choose_device(torch: Any) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compact_image(source_path: Path, output_path: Path, max_edge: int) -> Path:
    with Image.open(source_path) as source:
        image = source.convert("RGB")
        ratio = min(1.0, max_edge / max(image.size))
        size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
        image = image.resize(size, Image.Resampling.LANCZOS) if size != image.size else image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=92)
    return output_path


def main() -> int:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    parser = argparse.ArgumentParser(description="Run one Hugging Face VLM over normalized SAM2 Track sheets")
    parser.add_argument("--assets-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--max-image-edge", type=int, default=768)
    parser.add_argument("--max-crops", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = json.loads(args.assets_json.read_text(encoding="utf-8"))
    prompt = assets["prompt"]
    (args.output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    started = time.perf_counter()
    try:
        import torch
        from transformers import pipeline

        device = choose_device(torch)
        dtype = torch.bfloat16 if device in {"mps", "cuda"} else torch.float32
        pipe = pipeline(
            "image-text-to-text",
            model=args.model,
            trust_remote_code=True,
            device=device,
            dtype=dtype,
        )
        image_processor = getattr(getattr(pipe, "processor", None), "image_processor", None)
        if image_processor is None:
            image_processor = getattr(pipe, "image_processor", None)
        if args.max_crops is not None and image_processor is not None and hasattr(image_processor, "max_crops"):
            image_processor.max_crops = args.max_crops
        checkpoint_path = args.output_dir / "checkpoint.json"
        checkpoint = {}
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if (
                checkpoint.get("model") != args.model
                or checkpoint.get("max_image_edge") != args.max_image_edge
                or checkpoint.get("max_crops") != args.max_crops
            ):
                checkpoint = {}
        raw_items: list[dict[str, Any]] = checkpoint.get("raw_items", [])
        items: list[dict[str, Any]] = checkpoint.get("items", [])
        completed_track_ids = {item["track_id"] for item in items}
        for index, track in enumerate(assets["tracks"], start=1):
            if track["track_id"] in completed_track_ids:
                print(json.dumps({
                    "progress": f"{index}/{len(assets['tracks'])}",
                    "track_id": track["track_id"],
                    "status": "resumed_from_checkpoint",
                }), flush=True)
                continue
            track_started = time.perf_counter()
            compact_dir = args.output_dir / "input_assets"
            masked_input = compact_image(
                Path(track["masked_grid_path"]), compact_dir / f"{track['track_id']}_masked.jpg", args.max_image_edge
            )
            context_input = compact_image(
                Path(track["context_grid_path"]), compact_dir / f"{track['track_id']}_context.jpg", args.max_image_edge
            )
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "url": str(masked_input.resolve())},
                    {"type": "image", "url": str(context_input.resolve())},
                    {"type": "text", "text": prompt},
                ],
            }]
            result = pipe(text=messages, max_new_tokens=args.max_new_tokens, do_sample=False)
            raw = extract_generated_text(result)
            parse_error = None
            try:
                normalized = normalize(parse_json(raw))
            except Exception as exc:
                normalized = {
                    "canonical_object_name": "unknown",
                    "visible_attributes": {},
                    "observable_state": "unknown",
                    "confidence": 0.0,
                    "uncertainty_reason": "Model output was not valid benchmark JSON.",
                }
                parse_error = f"{type(exc).__name__}: {exc}"
            item = {
                "track_id": track["track_id"],
                **normalized,
                "latency_seconds": round(time.perf_counter() - track_started, 3),
                "parse_error": parse_error,
            }
            items.append(item)
            raw_items.append({"track_id": track["track_id"], "raw_response": raw, "parse_error": parse_error})
            checkpoint_path.write_text(json.dumps({
                "model": args.model,
                "max_image_edge": args.max_image_edge,
                "max_crops": args.max_crops,
                "items": items,
                "raw_items": raw_items,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({
                "progress": f"{index}/{len(assets['tracks'])}",
                "track_id": track["track_id"],
                "name": item["canonical_object_name"],
                "latency_seconds": item["latency_seconds"],
            }, ensure_ascii=False), flush=True)
        payload = {
            "model": args.model,
            "runner": "huggingface_image_text_to_text",
            "input_policy": (
                f"two 2x4 contact sheets per SAM2 Track, max edge {args.max_image_edge}px, "
                f"processor max_crops={args.max_crops if args.max_crops is not None else 'model default'}"
            ),
            "items": items,
        }
        metrics = {
            "status": "completed",
            "track_count": len(items),
            "total_latency_seconds": round(time.perf_counter() - started, 3),
        }
        (args.output_dir / "failure.json").unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
        (args.output_dir / "raw_response.json").write_text(json.dumps(raw_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "naming.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        failure = {
            "status": "failed",
            "phase": "model_load_or_inference",
            "model": args.model,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1200],
        }
        (args.output_dir / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
