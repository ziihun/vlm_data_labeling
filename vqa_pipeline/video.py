from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FrameSample:
    index: int
    timestamp_seconds: float
    path: Path


@dataclass(frozen=True, slots=True)
class ProcessedVideo:
    duration_seconds: float
    frames: list[FrameSample]
    review_video_path: Path


class VideoProcessor:
    """Creates model evidence frames and full-motion human-review videos."""

    def __init__(self, frame_count: int = 8, max_width: int = 768) -> None:
        if frame_count < 2:
            raise ValueError("frame_count must be at least 2")
        self.frame_count = frame_count
        self.max_width = max_width

    @staticmethod
    def _run_checked(command: list[str]) -> str:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    @staticmethod
    def uniform_timestamps(duration: float, frame_count: int) -> list[float]:
        if frame_count < 2:
            raise ValueError("frame_count must be at least 2")
        safe_end = max(0.0, duration - 0.05)
        if safe_end == 0:
            return [0.0 for _ in range(frame_count)]
        return [round(safe_end * index / (frame_count - 1), 3) for index in range(frame_count)]

    def duration(self, video_path: Path) -> float:
        output = self._run_checked(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(video_path),
            ]
        )
        duration = float(json.loads(output)["format"]["duration"])
        if duration <= 0:
            raise ValueError(f"Invalid duration for {video_path}: {duration}")
        return duration

    def extract_frames(
        self,
        video_path: Path,
        frame_dir: Path,
    ) -> tuple[float, list[FrameSample]]:
        duration = self.duration(video_path)
        timestamps = self.uniform_timestamps(duration, self.frame_count)
        frame_dir.mkdir(parents=True, exist_ok=True)
        frames: list[FrameSample] = []

        for index, timestamp in enumerate(timestamps):
            output_path = frame_dir / f"frame_{index:02d}_{timestamp:07.3f}s.jpg"
            self._run_checked(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-vf",
                    f"scale='min({self.max_width},iw)':-2",
                    "-q:v",
                    "3",
                    str(output_path),
                ]
            )
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError(f"Frame extraction failed: {output_path}")
            frames.append(
                FrameSample(
                    index=index,
                    timestamp_seconds=timestamp,
                    path=output_path,
                )
            )
        return duration, frames

    @staticmethod
    def create_review_video(video_path: Path, output_path: Path) -> Path:
        """Copies the complete MP4 so reviewers can inspect motion, not only frames."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_path, output_path)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Review video creation failed: {output_path}")
        return output_path

    def process(
        self,
        video_path: Path,
        output_root: Path,
        video_id: str,
    ) -> ProcessedVideo:
        duration, frames = self.extract_frames(
            video_path=video_path,
            frame_dir=output_root / "frames" / video_id,
        )
        review_video = self.create_review_video(
            video_path,
            output_root / "review_videos" / f"{video_id}.mp4",
        )
        return ProcessedVideo(
            duration_seconds=duration,
            frames=frames,
            review_video_path=review_video,
        )
