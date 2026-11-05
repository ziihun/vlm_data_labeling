from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


PROMPT = """Name every SAM2 Track in this SteelBench clip.

Image 1 is a 2x4 contact sheet of the eight ordered source frames.
Image 2 is the same frames with SAM2 Track ID bounding boxes.
Human GT is semantic context and candidate vocabulary, but a name must still be visually supported.
SAM2 regions can be objects, object parts, background structures, or duplicate fragments.

Return one JSON object only:
{{"items":[{{"track_id":"C000_T001","canonical_object_name":"short specific English noun phrase or unknown","visible_attributes":{{"key":"value"}},"observable_state":"short visible state or unknown","confidence":0.0,"uncertainty_reason":""}}]}}

Requirements:
- Return exactly one item for every supplied Track ID and no invented Track IDs.
- Be specific when visible: worker, safety helmet, red truck trailer, material sack, hand truck, pipe, railing, conveyor, structural column, etc.
- Do not call a region a worker merely because GT says workers exist.
- Use unknown for an unclear, fragmented, or non-object region.
- Do not infer risk, intent, or invisible attributes.
- English values and valid JSON only. No Markdown.

Track data:
{tracks}

Human GT context:
{gt}
"""


def jobs(path: Path) -> list[tuple[str, Path, Path]]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            clip_id, assets, output = line.split("\t")
            result.append((clip_id, Path(assets), Path(output)))
    return result


def parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def normalized(track_id: str, item: dict[str, Any] | None) -> dict[str, Any]:
    item = item or {}
    attributes = item.get("visible_attributes") or {}
    if not isinstance(attributes, dict):
        attributes = {"description": str(attributes)}
    name = str(item.get("canonical_object_name") or "unknown").strip()
    confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
    return {
        "track_id": track_id,
        "canonical_object_name": name,
        "visible_attributes": {str(key): str(value) for key, value in attributes.items()},
        "observable_state": str(item.get("observable_state") or "unknown").strip(),
        "confidence": confidence,
        "uncertainty_reason": str(item.get("uncertainty_reason") or "").strip(),
        "review_status": "auto_candidate" if name.casefold() != "unknown" and confidence >= 0.6 else "needs_review",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-tsv", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cleanup-inputs", action="store_true")
    args = parser.parse_args()
    work = jobs(args.jobs_tsv)
    if not work:
        print(json.dumps({"phase": "naming", "status": "nothing_to_do"}), flush=True)
        return 0

    print(json.dumps({"phase": "naming", "status": "loading_model_once", "model": args.model, "clips": len(work)}), flush=True)
    model, processor = load(args.model)
    config = load_config(args.model)
    for index, (clip_id, assets_path, output_dir) in enumerate(work, 1):
        assets = json.loads(assets_path.read_text(encoding="utf-8"))
        track_ids = [track["track_id"] for track in assets["tracks"]]
        prompt = PROMPT.format(
            tracks=json.dumps(assets["tracks"], ensure_ascii=False, separators=(",", ":")),
            gt=json.dumps(assets["human_gt_context"], ensure_ascii=False, separators=(",", ":")),
        )
        images = [assets["original_contact"], assets["sam_bbox_contact"]]
        print(f"[naming {index}/{len(work)}] {clip_id} · tracks={len(track_ids)}", flush=True)
        started = time.perf_counter()
        formatted = apply_chat_template(processor, config, prompt, num_images=2)
        output = generate(model, processor, formatted, images, max_tokens=4000, temperature=0.0, verbose=False)
        raw = getattr(output, "text", str(output))
        parse_error = None
        try:
            parsed = parse_json(raw)
            returned = {
                str(item.get("track_id")): item
                for item in parsed.get("items", [])
                if isinstance(item, dict) and str(item.get("track_id")) in track_ids
            }
        except Exception as exc:
            returned = {}
            parse_error = f"{type(exc).__name__}: {exc}"
        items = [normalized(track_id, returned.get(track_id)) for track_id in track_ids]
        payload = {
            "model": args.model,
            "runner": "mlx_vlm_clip_batch",
            "input_policy": "one original contact plus one SAM2 BBox contact per clip",
            "clip_id": clip_id,
            "parse_error": parse_error,
            "items": items,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (output_dir / "raw_response.json").write_text(json.dumps({"raw_response": raw, "parse_error": parse_error}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary = output_dir / "naming.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output_dir / "naming.json")
        (output_dir / "metrics.json").write_text(json.dumps({
            "status": "completed",
            "track_count": len(items),
            "named_count": sum(item["canonical_object_name"].casefold() != "unknown" for item in items),
            "latency_seconds": round(time.perf_counter() - started, 3),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.cleanup_inputs:
            shutil.rmtree(output_dir / "inputs", ignore_errors=True)
    del model, processor
    gc.collect()
    mx.clear_cache()
    print(json.dumps({"phase": "naming", "status": "completed", "clips": len(work)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
