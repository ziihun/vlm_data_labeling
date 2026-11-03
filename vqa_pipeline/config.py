from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PilotConfig:
    project_root: Path
    dataset_root: Path
    output_root: Path
    model: str
    num_videos: int = 5
    frames_per_video: int = 8
    max_frame_width: int = 768
    image_detail: str = "auto"
    prepare_only: bool = False
