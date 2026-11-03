from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


LABEL_PRIORITY = (
    "Safe Walkway Violation",
    "Safe Walkway",
    "Unauthorized Intervention",
    "Authorized Intervention",
    "Opened Panel Cover",
    "Closed Panel Cover",
    "Carrying Overload with Forklift",
    "Safe Carrying",
)


class DatasetCatalog:
    """Loads FiftyOne metadata and selects a label-diverse pilot subset."""

    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root

    def load_samples(self) -> list[dict[str, Any]]:
        samples_path = self.dataset_root / "samples.json"
        with samples_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        samples = payload.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"No samples found in {samples_path}")
        return samples

    def select_diverse_samples(
        self,
        samples: Iterable[dict[str, Any]],
        count: int,
    ) -> list[dict[str, Any]]:
        if count < 1:
            raise ValueError("count must be at least 1")

        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        all_samples: list[dict[str, Any]] = []
        for sample in samples:
            label = sample.get("ground_truth", {}).get("label", "unknown")
            by_label[label].append(sample)
            all_samples.append(sample)

        selected: list[dict[str, Any]] = []
        selected_paths: set[str] = set()
        for label in LABEL_PRIORITY:
            candidates = sorted(by_label.get(label, []), key=lambda item: item["filepath"])
            if candidates and len(selected) < count:
                selected.append(candidates[0])
                selected_paths.add(candidates[0]["filepath"])

        if len(selected) < count:
            for sample in sorted(all_samples, key=lambda item: item["filepath"]):
                if sample["filepath"] not in selected_paths:
                    selected.append(sample)
                    selected_paths.add(sample["filepath"])
                    if len(selected) == count:
                        break

        if len(selected) != count:
            raise ValueError(f"Requested {count} samples but selected {len(selected)}")
        return selected

    def select(self, count: int) -> list[dict[str, Any]]:
        return self.select_diverse_samples(self.load_samples(), count)
