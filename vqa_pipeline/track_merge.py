from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops


@dataclass(slots=True)
class LocalTrack:
    clip_id: str
    root: Path
    payload: dict[str, Any]


@dataclass(slots=True)
class GlobalTrack:
    track_id: str
    local_tracks: list[LocalTrack] = field(default_factory=list)


class OverlapTrackMerger:
    def __init__(self, iou_threshold: float = 0.55, comparison_frames: int = 5) -> None:
        self.iou_threshold = iou_threshold
        self.comparison_frames = comparison_frames

    @staticmethod
    def mask_iou(first: Image.Image, second: Image.Image) -> float:
        first = first.convert("L").point(lambda value: 255 if value else 0)
        second = second.convert("L").point(lambda value: 255 if value else 0)
        intersection = ImageChops.multiply(first, second)
        union = ImageChops.lighter(first, second)
        intersection_area = sum(value * count for value, count in enumerate(intersection.histogram())) / 255
        union_area = sum(value * count for value, count in enumerate(union.histogram())) / 255
        return float(intersection_area / union_area) if union_area else 0.0

    @staticmethod
    def _frames(track: LocalTrack) -> dict[float, dict[str, Any]]:
        return {
            round(float(item["source_timestamp_seconds"]), 3): item
            for item in track.payload["frames"]
            if int(item["mask_area_pixels"]) > 0
        }

    def _pair_iou(self, first: LocalTrack, second: LocalTrack) -> float:
        first_frames, second_frames = self._frames(first), self._frames(second)
        common = sorted(set(first_frames) & set(second_frames))
        if not common:
            return 0.0
        if len(common) > self.comparison_frames:
            positions = [round(i * (len(common) - 1) / (self.comparison_frames - 1)) for i in range(self.comparison_frames)]
            common = [common[position] for position in positions]
        scores = []
        for timestamp in common:
            first_mask = Image.open(first.root / first_frames[timestamp]["mask_path"])
            second_mask = Image.open(second.root / second_frames[timestamp]["mask_path"])
            scores.append(self.mask_iou(first_mask, second_mask))
        return sum(scores) / len(scores)

    def merge(self, results: list[tuple[dict[str, Any], Path]]) -> dict[str, Any]:
        global_tracks: list[GlobalTrack] = []
        decisions: list[dict[str, Any]] = []
        for clip_result, root in sorted(results, key=lambda item: int(item[0]["clip_index"])):
            local_tracks = [
                LocalTrack(str(clip_result["clip_id"]), root, track)
                for track in clip_result["tracks"]
            ]
            candidates: list[tuple[float, int, int]] = []
            for local_index, local in enumerate(local_tracks):
                for global_index, global_track in enumerate(global_tracks):
                    score = max(self._pair_iou(existing, local) for existing in global_track.local_tracks)
                    if score >= self.iou_threshold:
                        candidates.append((score, local_index, global_index))
            matched_local: set[int] = set()
            matched_global: set[int] = set()
            for score, local_index, global_index in sorted(candidates, reverse=True):
                if local_index in matched_local or global_index in matched_global:
                    continue
                local = local_tracks[local_index]
                global_tracks[global_index].local_tracks.append(local)
                matched_local.add(local_index)
                matched_global.add(global_index)
                decisions.append({
                    "local_track_id": local.payload["track_id"],
                    "global_track_id": global_tracks[global_index].track_id,
                    "decision": "merged_overlap",
                    "mean_mask_iou": round(score, 6),
                })
            for index, local in enumerate(local_tracks):
                if index in matched_local:
                    continue
                global_track = GlobalTrack(f"G{len(global_tracks) + 1:04d}", [local])
                global_tracks.append(global_track)
                decisions.append({
                    "local_track_id": local.payload["track_id"],
                    "global_track_id": global_track.track_id,
                    "decision": "new_global_track",
                    "mean_mask_iou": None,
                })

        output_tracks = []
        for global_track in global_tracks:
            observations = []
            for local in global_track.local_tracks:
                for frame in local.payload["frames"]:
                    if int(frame["mask_area_pixels"]) <= 0:
                        continue
                    observations.append({
                        "clip_id": local.clip_id,
                        "local_track_id": local.payload["track_id"],
                        "frame_index": int(frame["frame_index"]),
                        "source_timestamp_seconds": float(frame["source_timestamp_seconds"]),
                        "source_frame_path": frame["source_frame_path"],
                        "mask_path": str((local.root / frame["mask_path"]).resolve()),
                        "bbox_xyxy": frame["bbox_xyxy"],
                        "mask_area_pixels": int(frame["mask_area_pixels"]),
                    })
            unique: dict[float, dict[str, Any]] = {}
            for item in sorted(observations, key=lambda value: (value["source_timestamp_seconds"], value["clip_id"])):
                unique.setdefault(round(item["source_timestamp_seconds"], 3), item)
            observations = list(unique.values())
            output_tracks.append({
                "track_id": global_track.track_id,
                "local_track_ids": [item.payload["track_id"] for item in global_track.local_tracks],
                "first_seen_seconds": min(item["source_timestamp_seconds"] for item in observations),
                "last_seen_seconds": max(item["source_timestamp_seconds"] for item in observations),
                "observations": observations,
            })
        return {
            "model": results[0][0]["model"] if results else "unknown",
            "merge_method": "same_source_timestamp_mask_iou",
            "merge_iou_threshold": self.iou_threshold,
            "track_count": len(output_tracks),
            "tracks": output_tracks,
            "merge_decisions": decisions,
        }
