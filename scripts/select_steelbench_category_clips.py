from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


CATEGORY_PATTERN = re.compile(r"^(clip_.+)_\d{8}_\d{6}_\d+$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--sam-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    if args.per_category < 1:
        parser.error("--per-category must be >= 1")

    raw_root = args.raw_root.resolve()
    sam_root = args.sam_root.resolve()
    groups: dict[str, list[str]] = defaultdict(list)
    for annotation in sorted((raw_root / "data/annotations").glob("*.json")):
        clip_id = annotation.stem
        match = CATEGORY_PATTERN.match(clip_id)
        frames = raw_root / "data/frames" / clip_id
        if match and frames.is_dir() and len(list(frames.glob("frame_*.jpg"))) == 8:
            groups[match.group(1)].append(clip_id)

    categories = []
    selected_all = []
    for category in sorted(groups):
        available = sorted(groups[category])
        completed = [
            clip_id for clip_id in available
            if (sam_root / clip_id / "tracks.json").exists()
            and (sam_root / clip_id / "tracking_contact.jpg").exists()
        ]
        remaining = [clip_id for clip_id in available if clip_id not in set(completed)]
        rng = random.Random(f"{args.seed}:{category}")
        rng.shuffle(completed)
        rng.shuffle(remaining)
        selected = (completed + remaining)[: args.per_category]
        selected_all.extend(selected)
        categories.append({
            "category": category,
            "available_clip_count": len(available),
            "selected_clip_count": len(selected),
            "reused_sam_count": sum(clip_id in set(completed) for clip_id in selected),
            "selected_clip_ids": selected,
        })

    payload = {
        "schema_version": "1.0.0-steelbench-category-selection",
        "category_rule": "clip prefix before _YYYYMMDD_HHMMSS_NNNN",
        "selection_policy": "random within category, completed SAM clips preferred for reuse",
        "seed": args.seed,
        "per_category": args.per_category,
        "category_count": len(categories),
        "selected_clip_count": len(selected_all),
        "reused_sam_count": sum(item["reused_sam_count"] for item in categories),
        "clip_ids": selected_all,
        "categories": categories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "categories": payload["category_count"],
        "selected_clips": payload["selected_clip_count"],
        "reused_sam": payload["reused_sam_count"],
        "new_sam_needed": payload["selected_clip_count"] - payload["reused_sam_count"],
        "selection": str(args.output.resolve()),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
