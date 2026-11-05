from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from local_vlm_name import normalized_result, parse_json


def read_jobs(path: Path) -> list[dict[str, Any]]:
    jobs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        video_id, assets_json, output_dir = line.split("\t")
        jobs.append({
            "video_id": video_id,
            "assets_json": Path(assets_json),
            "output_dir": Path(output_dir),
        })
    return jobs


def cleanup_images(assets: dict[str, Any], assets_json: Path) -> None:
    crops_json = Path(assets["source_crops_json"])
    for name in ("original", "masked", "context"):
        target = crops_json.parent / name
        if target.is_dir():
            shutil.rmtree(target)
    input_dir = assets_json.parent
    if input_dir.is_dir():
        shutil.rmtree(input_dir)


def run_job(
    job: dict[str, Any], model: Any, processor: Any, config: Any,
    model_id: str, video_index: int, video_count: int, cleanup: bool,
) -> None:
    assets_json: Path = job["assets_json"]
    output_dir: Path = job["output_dir"]
    assets = json.loads(assets_json.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = assets["prompt"]
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    raw_items: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    started = time.perf_counter()
    print(json.dumps({
        "phase": "naming", "video": f"{video_index}/{video_count}",
        "video_id": job["video_id"], "tracks": len(assets["tracks"]),
    }, ensure_ascii=False), flush=True)
    for track_index, track in enumerate(assets["tracks"], start=1):
        images = [track["masked_grid_path"], track["context_grid_path"]]
        track_started = time.perf_counter()
        formatted = apply_chat_template(processor, config, prompt, num_images=len(images))
        output = generate(
            model, processor, formatted, images,
            max_tokens=500, temperature=0.0, verbose=False,
        )
        elapsed = time.perf_counter() - track_started
        raw = getattr(output, "text", str(output))
        parse_error = None
        try:
            result = normalized_result(parse_json(raw))
        except Exception as exc:
            result = {
                "canonical_object_name": "unknown", "visible_attributes": {},
                "observable_state": "unknown", "confidence": 0.0,
                "uncertainty_reason": "Local model output was not valid JSON",
            }
            parse_error = f"{type(exc).__name__}: {exc}"
        item = {
            "track_id": track["track_id"], **result,
            "latency_seconds": round(elapsed, 3), "parse_error": parse_error,
        }
        items.append(item)
        raw_items.append({"track_id": track["track_id"], "raw_response": raw, "parse_error": parse_error})
        print(json.dumps({
            "phase": "naming", "video": f"{video_index}/{video_count}",
            "video_id": job["video_id"],
            "track": f"{track_index}/{len(assets['tracks'])}",
            "name": item["canonical_object_name"],
        }, ensure_ascii=False), flush=True)
    payload = {
        "model": model_id, "runner": "mlx_vlm_batch",
        "input_policy": "two 2x4 contact sheets per SAM2 Track", "items": items,
    }
    (output_dir / "raw_response.json").write_text(
        json.dumps(raw_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "naming.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(json.dumps({
        "status": "completed", "track_count": len(items),
        "total_latency_seconds": round(time.perf_counter() - started, 3),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if cleanup:
        cleanup_images(assets, assets_json)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load one MLX VLM once and name Tracks for many videos")
    parser.add_argument("--jobs-tsv", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cleanup-images", action="store_true")
    args = parser.parse_args()
    jobs = read_jobs(args.jobs_tsv)
    if not jobs:
        print(json.dumps({"phase": "naming", "status": "nothing_to_do"}))
        return 0
    print(json.dumps({
        "phase": "naming", "status": "loading_model_once",
        "model": args.model, "videos": len(jobs),
    }, ensure_ascii=False), flush=True)
    model, processor = load(args.model)
    config = load_config(args.model)
    for index, job in enumerate(jobs, start=1):
        run_job(job, model, processor, config, args.model, index, len(jobs), args.cleanup_images)
    del model, processor
    gc.collect()
    mx.clear_cache()
    print(json.dumps({"phase": "naming", "status": "completed", "videos": len(jobs)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
