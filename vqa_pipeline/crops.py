from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance


class MarginCropGenerator:
    def __init__(self, margin_ratio: float = 0.20, max_frames_per_track: int = 8) -> None:
        self.margin_ratio = margin_ratio
        self.max_frames_per_track = max_frames_per_track

    def _select(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(observations) <= self.max_frames_per_track:
            return observations
        return [
            observations[round(i * (len(observations) - 1) / (self.max_frames_per_track - 1))]
            for i in range(self.max_frames_per_track)
        ]

    def run(self, merged: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        for name in ("original", "masked", "context"):
            (output_dir / name).mkdir(parents=True, exist_ok=True)
        tracks = []
        for track in merged["tracks"]:
            evidence = []
            for evidence_index, item in enumerate(self._select(track["observations"])):
                image = Image.open(item["source_frame_path"]).convert("RGB")
                mask = Image.open(item["mask_path"]).convert("L")
                x1, y1, x2, y2 = [int(value) for value in item["bbox_xyxy"]]
                width, height = x2 - x1, y2 - y1
                mx, my = round(width * self.margin_ratio), round(height * self.margin_ratio)
                margin_box = (
                    max(0, x1 - mx), max(0, y1 - my),
                    min(image.width, x2 + mx), min(image.height, y2 + my),
                )
                prefix = f"{track['track_id']}_{evidence_index:02d}"
                original_path = output_dir / "original" / f"{prefix}.jpg"
                masked_path = output_dir / "masked" / f"{prefix}.jpg"
                context_path = output_dir / "context" / f"{prefix}.jpg"
                image.crop((x1, y1, x2, y2)).save(original_path, quality=95)
                crop = image.crop(margin_box)
                crop_mask = mask.crop(margin_box)
                dimmed = ImageEnhance.Brightness(crop).enhance(0.18)
                dimmed.paste(crop, (0, 0), crop_mask)
                dimmed.save(masked_path, quality=95)
                context = crop.copy()
                draw = ImageDraw.Draw(context)
                local_box = (x1 - margin_box[0], y1 - margin_box[1], x2 - margin_box[0], y2 - margin_box[1])
                draw.rectangle(local_box, outline=(255, 40, 40), width=4)
                context.save(context_path, quality=95)
                evidence.append({
                    "evidence_index": evidence_index,
                    "frame_index": item["frame_index"],
                    "source_timestamp_seconds": item["source_timestamp_seconds"],
                    "bbox_xyxy": item["bbox_xyxy"],
                    "mask_path": item["mask_path"],
                    "original_crop_path": str(original_path.resolve()),
                    "masked_crop_path": str(masked_path.resolve()),
                    "context_crop_path": str(context_path.resolve()),
                })
            tracks.append({
                "track_id": track["track_id"],
                "first_seen_seconds": track["first_seen_seconds"],
                "last_seen_seconds": track["last_seen_seconds"],
                "evidence": evidence,
            })
        payload = {
            "margin_ratio": self.margin_ratio,
            "max_frames_per_track": self.max_frames_per_track,
            "tracks": tracks,
        }
        (output_dir / "crops.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return payload
