from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


MODEL_ID = "mlx-community/Qwen3.5-4B-MLX-4bit"
PROMPT_TEMPLATE = """You are an object annotation assistant.

Identify ONLY the target object highlighted by the segmentation mask.
The images are chronological pairs: masked crop, then context crop.
Use the context crop to understand the target object's appearance, but do not name surrounding objects as the target.

Return a single JSON object with exactly these keys:
- canonical_object_name: short English noun phrase, or \"unknown\"
- visible_attributes: JSON object with English attribute names and English string values
- observable_state: short English phrase, or \"unknown\"
- confidence: number from 0 to 1

Do not infer actions, intentions, hazards, or events that are not directly visible.
If the target object cannot be identified, return \"unknown\".
Return valid JSON only. Do not use Markdown.
"""


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crops = json.loads(args.crops_json.read_text(encoding="utf-8"))
    (args.output_dir / "prompt.txt").write_text(PROMPT_TEMPLATE, encoding="utf-8")
    model, processor = load(MODEL_ID)
    config = load_config(MODEL_ID)
    raw_responses = []
    namings = []
    for index, track in enumerate(crops["tracks"], start=1):
        images = []
        for evidence in track["evidence"]:
            images.extend([evidence["masked_crop_path"], evidence["context_crop_path"]])
        formatted = apply_chat_template(processor, config, PROMPT_TEMPLATE, num_images=len(images))
        output = generate(
            model, processor, formatted, images,
            max_tokens=400, temperature=0.0, verbose=False,
        )
        raw = getattr(output, "text", str(output))
        parse_error = None
        try:
            parsed = parse_json(raw)
            name = str(parsed.get("canonical_object_name") or "unknown").strip()
            attributes = parsed.get("visible_attributes") or {}
            if not isinstance(attributes, dict):
                attributes = {"description": str(attributes)}
            attributes = {str(key): str(value) for key, value in attributes.items()}
            state = str(parsed.get("observable_state") or "unknown").strip()
            confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.5))))
        except Exception as exc:
            name, attributes, state, confidence = "unknown", {}, "unknown", 0.0
            parse_error = f"{type(exc).__name__}: {exc}"
        raw_responses.append({"track_id": track["track_id"], "raw_response": raw, "parse_error": parse_error})
        namings.append({
            "track_id": track["track_id"],
            "canonical_object_name": name,
            "visible_attributes": attributes,
            "observable_state": state,
            "confidence": confidence,
            "raw_response": raw,
            "parse_error": parse_error,
        })
        print(json.dumps({"progress": f"{index}/{len(crops['tracks'])}", "track_id": track["track_id"], "name": name}))
    (args.output_dir / "raw_response.json").write_text(json.dumps(raw_responses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload = {"model": MODEL_ID, "language_policy": "English internal values", "items": namings}
    (args.output_dir / "naming.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
