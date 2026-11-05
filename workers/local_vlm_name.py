from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


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


def normalized_result(parsed: dict[str, Any]) -> dict[str, Any]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one local MLX VLM over normalized SAM2 Track sheets")
    parser.add_argument("--assets-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = json.loads(args.assets_json.read_text(encoding="utf-8"))
    prompt = assets["prompt"]
    (args.output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    started = time.perf_counter()
    try:
        model, processor = load(args.model)
        config = load_config(args.model)
    except Exception as exc:
        failure = {"status": "failed", "phase": "model_load", "model": args.model, "error": f"{type(exc).__name__}: {exc}"}
        (args.output_dir / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    raw_items: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for index, track in enumerate(assets["tracks"], start=1):
        images = [track["masked_grid_path"], track["context_grid_path"]]
        track_started = time.perf_counter()
        formatted = apply_chat_template(processor, config, prompt, num_images=len(images))
        output = generate(
            model,
            processor,
            formatted,
            images,
            max_tokens=500,
            temperature=0.0,
            verbose=False,
        )
        elapsed = time.perf_counter() - track_started
        raw = getattr(output, "text", str(output))
        parse_error = None
        try:
            result = normalized_result(parse_json(raw))
        except Exception as exc:
            result = {
                "canonical_object_name": "unknown",
                "visible_attributes": {},
                "observable_state": "unknown",
                "confidence": 0.0,
                "uncertainty_reason": "Local model output was not valid JSON",
            }
            parse_error = f"{type(exc).__name__}: {exc}"
        item = {
            "track_id": track["track_id"],
            **result,
            "latency_seconds": round(elapsed, 3),
            "parse_error": parse_error,
        }
        items.append(item)
        raw_items.append({"track_id": track["track_id"], "raw_response": raw, "parse_error": parse_error})
        print(json.dumps({
            "progress": f"{index}/{len(assets['tracks'])}",
            "track_id": track["track_id"],
            "name": item["canonical_object_name"],
            "latency_seconds": item["latency_seconds"],
        }, ensure_ascii=False), flush=True)
    metrics = {
        "status": "completed",
        "track_count": len(items),
        "total_latency_seconds": round(time.perf_counter() - started, 3),
    }
    payload = {
        "model": args.model,
        "runner": "mlx_vlm",
        "input_policy": "two 2x4 contact sheets per SAM2 Track",
        "items": items,
    }
    (args.output_dir / "raw_response.json").write_text(json.dumps(raw_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "naming.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
