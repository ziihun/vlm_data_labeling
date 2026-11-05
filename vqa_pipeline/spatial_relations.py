from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image

from .spec_models import VerifiedAnnotation, VerifiedSpatialRelation


class StableBBoxSpatialRelationBuilder:
    """Derive reviewable spatial facts only from co-visible, stable Track boxes."""

    def __init__(
        self, min_common_frames: int = 5, min_consistency: float = 0.90,
        max_relations: int = 32,
    ) -> None:
        self.min_common_frames = min_common_frames
        self.min_consistency = min_consistency
        self.max_relations = max_relations

    @staticmethod
    def _sample(values: list[float], limit: int = 8) -> list[float]:
        if len(values) <= limit:
            return values
        return [values[round(i * (len(values) - 1) / (limit - 1))] for i in range(limit)]

    @staticmethod
    def _observations(track: dict[str, Any]) -> dict[float, dict[str, Any]]:
        return {
            round(float(item["source_timestamp_seconds"]), 6): item
            for item in track["observations"] if item.get("bbox_xyxy")
        }

    def build(
        self,
        annotation: VerifiedAnnotation,
        tracks_payload: dict[str, Any],
        output_path: Path | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> VerifiedAnnotation:
        allowed_ids = {item.track_id for item in annotation.objects}
        object_names = {item.track_id: item.object_name.strip().casefold() for item in annotation.objects}
        tracks = {
            item["track_id"]: item for item in tracks_payload["tracks"]
            if item["track_id"] in allowed_ids
        }
        if not tracks:
            return annotation
        if frame_size is None:
            first_observation = next(iter(tracks.values()))["observations"][0]
            with Image.open(first_observation["source_frame_path"]) as image:
                frame_width, frame_height = image.size
        else:
            frame_width, frame_height = frame_size

        candidates_found: list[tuple[float, dict[str, Any]]] = []
        for first_id, second_id in combinations(sorted(tracks), 2):
            if object_names[first_id] == object_names[second_id]:
                continue
            first = self._observations(tracks[first_id])
            second = self._observations(tracks[second_id])
            common = sorted(set(first) & set(second))
            if len(common) < self.min_common_frames:
                continue
            horizontal: list[tuple[float, str, float]] = []
            vertical: list[tuple[float, str, float]] = []
            for timestamp in common:
                a = [float(value) for value in first[timestamp]["bbox_xyxy"]]
                b = [float(value) for value in second[timestamp]["bbox_xyxy"]]
                ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
                bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                dx, dy = (ax - bx) / frame_width, (ay - by) / frame_height
                if a[2] < b[0] and (b[0] - a[2]) / frame_width >= 0.02:
                    horizontal.append((timestamp, "left_of", abs(dx)))
                elif b[2] < a[0] and (a[0] - b[2]) / frame_width >= 0.02:
                    horizontal.append((timestamp, "right_of", abs(dx)))
                elif abs(dx) >= 0.15:
                    horizontal.append((timestamp, "left_of" if dx < 0 else "right_of", abs(dx)))
                if a[3] < b[1] and (b[1] - a[3]) / frame_height >= 0.02:
                    vertical.append((timestamp, "above", abs(dy)))
                elif b[3] < a[1] and (a[1] - b[3]) / frame_height >= 0.02:
                    vertical.append((timestamp, "below", abs(dy)))
                elif abs(dy) >= 0.15:
                    vertical.append((timestamp, "above" if dy < 0 else "below", abs(dy)))

            pair_candidates: list[tuple[float, dict[str, Any]]] = []
            for candidates in (horizontal, vertical):
                if len(candidates) < self.min_common_frames:
                    continue
                labels = [label for _, label, _ in candidates]
                relation = max(set(labels), key=labels.count)
                supporting_rows = [row for row in candidates if row[1] == relation]
                supporting = sorted(timestamp for timestamp, _, _ in supporting_rows)
                consistency = len(supporting) / len(common)
                if consistency < self.min_consistency:
                    continue
                separation = median(value for _, _, value in supporting_rows)
                rank_score = consistency + min(separation, 1.0) * 0.25 + min(len(common), 50) * 0.002
                pair_candidates.append((rank_score, {
                    "subject_track_id": first_id,
                    "relation": relation,
                    "reference_track_id": second_id,
                    "evidence_timestamps_seconds": self._sample(supporting),
                    "confidence": round(consistency, 3),
                    "co_visible_frame_count": len(common),
                    "median_axis_separation_ratio": round(separation, 4),
                }))
            if pair_candidates:
                candidates_found.append(max(pair_candidates, key=lambda item: item[0]))

        selected = sorted(candidates_found, key=lambda item: item[0], reverse=True)[:self.max_relations]
        relations = [
            VerifiedSpatialRelation(relation_id=f"SR{index:04d}", **payload)
            for index, (_, payload) in enumerate(selected, start=1)
        ]

        result = annotation.model_copy(update={"verified_spatial_relations": relations})
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps({
                "video_id": result.video_id,
                "method": "co-visible stable BBox geometry",
                "min_common_frames": self.min_common_frames,
                "min_consistency": self.min_consistency,
                "max_relations": self.max_relations,
                "candidate_pair_count": len(candidates_found),
                "frame_size": [frame_width, frame_height],
                "relations": [item.model_dump(mode="json") for item in relations],
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result
