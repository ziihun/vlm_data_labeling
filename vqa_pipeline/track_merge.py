from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops


@dataclass(slots=True)
class LocalTrackRef:
    clip_id: str
    track_id: str
    root: Path
    payload: dict[str, Any]


@dataclass(slots=True)
class GlobalTrack:
    global_track_id: str
    local_tracks: list[LocalTrackRef] = field(default_factory=list)


class OverlapTrackMerger:
    """Merges duplicate tracks in overlapping clips using same-time mask IoU."""

    def __init__(self, iou_threshold: float = 0.55, comparison_frames: int = 5) -> None:
        self.iou_threshold = iou_threshold
        self.comparison_frames = comparison_frames

    @staticmethod
    def _frame_map(track: LocalTrackRef) -> dict[float, dict[str, Any]]:
        return {
            round(float(frame["source_timestamp_seconds"]), 3): frame
            for frame in track.payload["frames"]
            if int(frame["mask_area_pixels"]) > 0
        }

    @staticmethod
    def _load_mask(track: LocalTrackRef, frame: dict[str, Any]) -> Image.Image:
        return Image.open(track.root / frame["mask_path"]).convert("L").point(
            lambda value: 255 if value else 0
        )

    @staticmethod
    def mask_iou(first: Image.Image, second: Image.Image) -> float:
        intersection = ImageChops.multiply(first, second)
        union = ImageChops.lighter(first, second)
        intersection_pixels = sum(
            value * count for value, count in enumerate(intersection.histogram())
        ) / 255
        union_pixels = sum(
            value * count for value, count in enumerate(union.histogram())
        ) / 255
        return float(intersection_pixels / union_pixels) if union_pixels else 0.0

    def _pair_iou(self, first: LocalTrackRef, second: LocalTrackRef) -> float:
        first_frames = self._frame_map(first)
        second_frames = self._frame_map(second)
        common = sorted(set(first_frames) & set(second_frames))
        if not common:
            return 0.0
        if len(common) > self.comparison_frames:
            positions = [
                round(index * (len(common) - 1) / (self.comparison_frames - 1))
                for index in range(self.comparison_frames)
            ]
            common = [common[position] for position in positions]
        scores = [
            self.mask_iou(
                self._load_mask(first, first_frames[timestamp]),
                self._load_mask(second, second_frames[timestamp]),
            )
            for timestamp in common
        ]
        return sum(scores) / len(scores)

    def merge(self, clip_results: list[tuple[dict[str, Any], Path]]) -> dict[str, Any]:
        globals_: list[GlobalTrack] = []
        decisions: list[dict[str, Any]] = []
        for clip_result, root in sorted(
            clip_results, key=lambda item: int(item[0]["clip_index"])
        ):
            locals_ = [
                LocalTrackRef(
                    clip_id=str(clip_result["clip_id"]),
                    track_id=str(track["track_id"]),
                    root=root,
                    payload=track,
                )
                for track in clip_result["tracks"]
            ]
            if not globals_:
                for local in locals_:
                    global_track = GlobalTrack(
                        global_track_id=f"G{len(globals_) + 1:04d}",
                        local_tracks=[local],
                    )
                    globals_.append(global_track)
                    decisions.append(
                        {
                            "local_track_id": local.track_id,
                            "global_track_id": global_track.global_track_id,
                            "decision": "new",
                            "overlap_mean_iou": None,
                        }
                    )
                continue

            candidates: list[tuple[float, int, int]] = []
            for local_index, local in enumerate(locals_):
                for global_index, global_track in enumerate(globals_):
                    score = max(
                        self._pair_iou(existing, local)
                        for existing in global_track.local_tracks
                    )
                    if score >= self.iou_threshold:
                        candidates.append((score, local_index, global_index))
            matched_locals: set[int] = set()
            matched_globals: set[int] = set()
            for score, local_index, global_index in sorted(candidates, reverse=True):
                if local_index in matched_locals or global_index in matched_globals:
                    continue
                local = locals_[local_index]
                global_track = globals_[global_index]
                global_track.local_tracks.append(local)
                matched_locals.add(local_index)
                matched_globals.add(global_index)
                decisions.append(
                    {
                        "local_track_id": local.track_id,
                        "global_track_id": global_track.global_track_id,
                        "decision": "merged_overlap",
                        "overlap_mean_iou": round(score, 6),
                    }
                )
            for local_index, local in enumerate(locals_):
                if local_index in matched_locals:
                    continue
                global_track = GlobalTrack(
                    global_track_id=f"G{len(globals_) + 1:04d}",
                    local_tracks=[local],
                )
                globals_.append(global_track)
                decisions.append(
                    {
                        "local_track_id": local.track_id,
                        "global_track_id": global_track.global_track_id,
                        "decision": "new_no_overlap_match",
                        "overlap_mean_iou": None,
                    }
                )

        global_payloads = []
        for global_track in globals_:
            representatives = []
            for local in global_track.local_tracks:
                for item in local.payload["representative_frames"]:
                    representatives.append(
                        {
                            **item,
                            "clip_id": local.clip_id,
                            "local_track_id": local.track_id,
                            "rgba_path": str((local.root / item["rgba_path"]).resolve()),
                            "mask_path": str((local.root / item["mask_path"]).resolve()),
                        }
                    )
            unique_by_time: dict[float, dict[str, Any]] = {}
            for item in sorted(representatives, key=lambda value: value["source_timestamp_seconds"]):
                unique_by_time.setdefault(round(float(item["source_timestamp_seconds"]), 3), item)
            unique = list(unique_by_time.values())
            if len(unique) > 8:
                positions = [round(index * (len(unique) - 1) / 7) for index in range(8)]
                unique = [unique[position] for position in positions]
            global_payloads.append(
                {
                    "global_track_id": global_track.global_track_id,
                    "local_track_ids": [track.track_id for track in global_track.local_tracks],
                    "clip_ids": [track.clip_id for track in global_track.local_tracks],
                    "first_seen_seconds": min(
                        float(track.payload["first_seen_seconds"])
                        for track in global_track.local_tracks
                        if track.payload["first_seen_seconds"] is not None
                    ),
                    "last_seen_seconds": max(
                        float(track.payload["last_seen_seconds"])
                        for track in global_track.local_tracks
                        if track.payload["last_seen_seconds"] is not None
                    ),
                    "representative_frames": unique,
                }
            )
        return {
            "method": "same_source_timestamp_mask_iou",
            "iou_threshold": self.iou_threshold,
            "global_track_count": len(global_payloads),
            "global_tracks": global_payloads,
            "merge_decisions": decisions,
        }
