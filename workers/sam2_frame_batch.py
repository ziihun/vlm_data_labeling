from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from transformers import Sam2VideoModel, Sam2VideoProcessor, pipeline

from sam3_track import COLORS, Sam3PeriodicTracker


MODEL_ID = "facebook/sam2.1-hiera-tiny"


def ordered_frames(frame_dir: Path) -> list[Path]:
    frames = list(frame_dir.glob("frame_*.jpg"))
    return sorted(frames, key=lambda path: int(path.stem.rsplit("_", 1)[1]))


def durations(raw_root: Path) -> dict[str, float]:
    manifest = raw_root / "manifests/batch_manifest.csv"
    if not manifest.exists():
        return {}
    with manifest.open(encoding="utf-8", newline="") as handle:
        return {
            row["clip_id"]: float(row.get("clip_duration_sec") or 15.0)
            for row in csv.DictReader(handle)
        }


def clip_record(clip_id: str, frames: list[Path], duration: float) -> dict[str, Any]:
    interval = duration / max(len(frames) - 1, 1)
    return {
        "clip_id": clip_id,
        "clip_index": 0,
        "tracking_fps": round(1.0 / interval, 6) if interval else 0.0,
        "tracking_frames": [
            {
                "frame_index": index,
                "source_timestamp_seconds": round(index * interval, 6),
                "path": str(path.resolve()),
            }
            for index, path in enumerate(frames)
        ],
    }


def render_contact(frames: list[Path], tracks: list[dict[str, Any]], destination: Path) -> None:
    rendered: list[Image.Image] = []
    for frame_index, path in enumerate(frames):
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for track_index, track in enumerate(tracks):
            observation = next(
                (item for item in track["frames"] if item["frame_index"] == frame_index and item.get("bbox_xyxy")),
                None,
            )
            if observation is None:
                continue
            color = COLORS[track_index % len(COLORS)]
            box = observation["bbox_xyxy"]
            draw.rectangle(box, outline=color, width=max(2, image.width // 500))
            Sam3PeriodicTracker.label(draw, box, track["track_id"], color)
        rendered.append(image)
    cell_width = max(image.width for image in rendered)
    cell_height = max(image.height for image in rendered)
    sheet = Image.new("RGB", (cell_width * 4, cell_height * 2), "white")
    for index, image in enumerate(rendered):
        sheet.paste(image, ((index % 4) * cell_width, (index // 4) * cell_height))
    sheet.save(destination, quality=88, optimize=True)
    sheet.close()
    for image in rendered:
        image.close()


def clear_device(device: str) -> None:
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--start-after", default="")
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--keep-masks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_clips < 0:
        parser.error("--max-clips must be >= 0")

    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    annotations_root = raw_root / "data/annotations"
    duration_map = durations(raw_root)
    selected = None
    if args.selection_file:
        selection = json.loads(args.selection_file.read_text(encoding="utf-8"))
        selected = set(selection.get("clip_ids", [])) if isinstance(selection, dict) else set(selection)
    jobs: list[tuple[str, list[Path], float]] = []
    for frame_dir in sorted((raw_root / "data/frames").iterdir()):
        frames = ordered_frames(frame_dir) if frame_dir.is_dir() else []
        if len(frames) == 8 and (annotations_root / f"{frame_dir.name}.json").exists() and frame_dir.name > args.start_after and (selected is None or frame_dir.name in selected):
            jobs.append((frame_dir.name, frames, duration_map.get(frame_dir.name, 15.0)))
    if args.max_clips:
        jobs = jobs[:args.max_clips]
    print(json.dumps({"clips": len(jobs), "output": str(output_root), "dry_run": args.dry_run}), flush=True)
    if args.dry_run:
        for index, (clip_id, _, _) in enumerate(jobs, 1):
            print(f"[SAM2 {index}/{len(jobs)}] would process · {clip_id}", flush=True)
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    tracker = Sam3PeriodicTracker(
        redetect_interval_frames=2,
        model_id=MODEL_ID,
        video_model_class=Sam2VideoModel,
        video_processor_class=Sam2VideoProcessor,
        stage_name="sam2_discovery_and_tracking",
        official_cuda_required=False,
    )
    pending = [job for job in jobs if not ((output_root / job[0] / "tracks.json").exists() and (output_root / job[0] / "tracking_contact.jpg").exists())]
    resumed = len(jobs) - len(pending)
    if resumed:
        print(f"[SAM2] resume ready · {resumed}/{len(jobs)}", flush=True)
    if not pending:
        print("[SAM2] all outputs already exist", flush=True)
        return 0

    token = os.getenv("HF_TOKEN") or None
    print(f"[SAM2] loading automatic-mask and tracking components once · {MODEL_ID}", flush=True)
    generator = pipeline("mask-generation", model=MODEL_ID, token=token, device=tracker.device)
    model = Sam2VideoModel.from_pretrained(MODEL_ID, token=token).to(tracker.device)
    processor = Sam2VideoProcessor.from_pretrained(MODEL_ID, token=token)
    for index, (clip_id, frame_paths, duration) in enumerate(pending, 1):
        print(f"[SAM2 {index}/{len(pending)}] discovery + tracking · {clip_id}", flush=True)
        destination = output_root / clip_id
        destination.mkdir(parents=True, exist_ok=True)
        clip = clip_record(clip_id, frame_paths, duration)
        records = clip["tracking_frames"]
        images = [Image.open(path).convert("RGB") for path in frame_paths]
        rounds = tracker.discover(
            images, list(range(0, len(images), 2)), destination, token, generator=generator,
        )
        tracks = tracker.track_discoveries(clip, destination, images, records, rounds, model, processor)
        payload = {
            "stage": tracker.stage_name,
            "status": "completed",
            "model": MODEL_ID,
            "device": tracker.device,
            "clip_id": clip_id,
            "source_type": "eight_ordered_sparse_frames",
            "clip_duration_seconds": duration,
            "tracking_frame_count": len(images),
            "redetect_interval_frames": 2,
            "track_count": len(tracks),
            "periodic_new_track_count": sum(1 for track in tracks if track["periodic_discovery"]),
            "mask_files_retained": args.keep_masks,
            "discovery_rounds": rounds,
            "tracks": tracks,
        }
        (destination / "tracks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        render_contact(frame_paths, tracks, destination / "tracking_contact.jpg")
        for image in images:
            image.close()
        if not args.keep_masks:
            shutil.rmtree(destination / "masks", ignore_errors=True)
            shutil.rmtree(destination / "discovery", ignore_errors=True)
        print(json.dumps({"clip_id": clip_id, "tracks": len(tracks), "contact": str(destination / "tracking_contact.jpg")}), flush=True)
    del generator, model, processor
    clear_device(tracker.device)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
