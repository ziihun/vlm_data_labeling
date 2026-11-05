from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


CONTACT_SIZE = (3072, 864)


def ordered_frames(frame_dir: Path) -> list[Path]:
    return sorted(
        frame_dir.glob("frame_*.jpg"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)


def original_contact(frames: list[Path], destination: Path) -> None:
    cell = (CONTACT_SIZE[0] // 4, CONTACT_SIZE[1] // 2)
    sheet = Image.new("RGB", CONTACT_SIZE, "black")
    for index, path in enumerate(frames):
        with Image.open(path) as source:
            image = fit(source, cell)
        sheet.paste(image, ((index % 4) * cell[0], (index // 4) * cell[1]))
        image.close()
    sheet.save(destination, quality=88, optimize=True)
    sheet.close()


def resized_contact(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        resized = fit(image, CONTACT_SIZE)
    resized.save(destination, quality=88, optimize=True)
    resized.close()


def annotation_context(annotation: dict[str, Any]) -> dict[str, Any]:
    people = []
    for person in annotation.get("persons") or []:
        people.append({
            key: person.get(key)
            for key in (
                "person_id", "position", "action_name", "physical_description",
                "spatial_context", "ppe", "interaction",
            )
            if person.get(key) not in (None, "", [], {})
        })
    return {
        "num_workers": annotation.get("num_workers"),
        "visible_equipment": annotation.get("visible_equipment"),
        "scene_description": annotation.get("scene_description"),
        "persons": people,
    }


def track_context(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "track_id": track["track_id"],
            "discovered_at_frame_index": track.get("discovered_at_frame_index"),
            "observations": [
                {
                    "frame_index": item["frame_index"],
                    "bbox_xyxy": item.get("bbox_xyxy"),
                }
                for item in track.get("frames", [])
                if item.get("bbox_xyxy")
            ],
        }
        for track in payload.get("tracks", [])
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--sam-root", type=Path, required=True)
    parser.add_argument("--naming-root", type=Path, required=True)
    parser.add_argument("--jobs-tsv", type=Path, required=True)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--start-after", default="")
    parser.add_argument("--selection-file", type=Path)
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    sam_root = args.sam_root.resolve()
    naming_root = args.naming_root.resolve()
    naming_root.mkdir(parents=True, exist_ok=True)
    selected = None
    if args.selection_file:
        selection = json.loads(args.selection_file.read_text(encoding="utf-8"))
        selected = set(selection.get("clip_ids", [])) if isinstance(selection, dict) else set(selection)
    completed: list[str] = []
    for tracks_path in sorted(sam_root.glob("*/tracks.json")):
        clip_id = tracks_path.parent.name
        contact = tracks_path.parent / "tracking_contact.jpg"
        annotation = raw_root / "data/annotations" / f"{clip_id}.json"
        frames = ordered_frames(raw_root / "data/frames" / clip_id)
        if clip_id > args.start_after and contact.exists() and annotation.exists() and len(frames) == 8 and (selected is None or clip_id in selected):
            completed.append(clip_id)
    if args.max_clips:
        completed = completed[: args.max_clips]

    jobs = []
    resumed = 0
    for index, clip_id in enumerate(completed, 1):
        output_dir = naming_root / clip_id
        naming_json = output_dir / "naming.json"
        if naming_json.exists():
            resumed += 1
            continue
        input_dir = output_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        frames = ordered_frames(raw_root / "data/frames" / clip_id)
        original_path = input_dir / "original_contact.jpg"
        bbox_path = input_dir / "sam_bbox_contact.jpg"
        original_contact(frames, original_path)
        resized_contact(sam_root / clip_id / "tracking_contact.jpg", bbox_path)
        annotation = json.loads((raw_root / "data/annotations" / f"{clip_id}.json").read_text(encoding="utf-8"))
        sam_payload = json.loads((sam_root / clip_id / "tracks.json").read_text(encoding="utf-8"))
        assets = {
            "clip_id": clip_id,
            "original_contact": str(original_path.resolve()),
            "sam_bbox_contact": str(bbox_path.resolve()),
            "track_count": sam_payload.get("track_count", 0),
            "tracks": track_context(sam_payload),
            "human_gt_context": annotation_context(annotation),
        }
        assets_path = output_dir / "assets.json"
        assets_path.write_text(json.dumps(assets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        jobs.append((clip_id, assets_path.resolve(), output_dir.resolve()))
        print(f"[prepare {index}/{len(completed)}] {clip_id} · tracks={assets['track_count']}", flush=True)

    args.jobs_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.jobs_tsv.write_text(
        "".join(f"{clip_id}\t{assets}\t{output}\n" for clip_id, assets, output in jobs),
        encoding="utf-8",
    )
    print(json.dumps({
        "completed_sam_clips": len(completed),
        "pending_naming_clips": len(jobs),
        "resumed_naming_clips": resumed,
        "jobs_tsv": str(args.jobs_tsv.resolve()),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
