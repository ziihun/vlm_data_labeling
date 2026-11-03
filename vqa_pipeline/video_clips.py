from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VideoInfo:
    duration_seconds: float
    fps: float
    fps_fraction: str
    width: int
    height: int
    frame_count: int | None


@dataclass(frozen=True, slots=True)
class ClipWindow:
    clip_index: int
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    @property
    def clip_id(self) -> str:
        return f"clip_{self.clip_index:03d}_{self.start_seconds:07.3f}_{self.end_seconds:07.3f}"


class OverlappingClipPreprocessor:
    """Creates source-FPS clips and a separate 5 FPS tracking frame stream."""

    def __init__(
        self,
        clip_seconds: float = 10.0,
        stride_seconds: float = 5.0,
        tracking_fps: float = 5.0,
    ) -> None:
        if clip_seconds <= 0 or stride_seconds <= 0 or tracking_fps <= 0:
            raise ValueError("clip_seconds, stride_seconds, and tracking_fps must be positive")
        if stride_seconds > clip_seconds:
            raise ValueError("stride_seconds cannot exceed clip_seconds")
        self.clip_seconds = clip_seconds
        self.stride_seconds = stride_seconds
        self.tracking_fps = tracking_fps

    @staticmethod
    def _run(command: list[str]) -> str:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        return completed.stdout

    @classmethod
    def probe(cls, video_path: Path) -> VideoInfo:
        output = cls._run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,width,height,nb_frames,duration",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(video_path),
            ]
        )
        payload = json.loads(output)
        stream = payload["streams"][0]
        fps_fraction = stream["avg_frame_rate"]
        duration = float(stream.get("duration") or payload["format"]["duration"])
        count = stream.get("nb_frames")
        return VideoInfo(
            duration_seconds=duration,
            fps=float(Fraction(fps_fraction)),
            fps_fraction=fps_fraction,
            width=int(stream["width"]),
            height=int(stream["height"]),
            frame_count=int(count) if count not in (None, "N/A") else None,
        )

    def windows(self, duration_seconds: float) -> list[ClipWindow]:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        windows: list[ClipWindow] = []
        start = 0.0
        while start < duration_seconds - 1e-6:
            end = min(duration_seconds, start + self.clip_seconds)
            if windows and end <= windows[-1].end_seconds + 1e-6:
                break
            windows.append(
                ClipWindow(
                    clip_index=len(windows),
                    start_seconds=round(start, 6),
                    end_seconds=round(end, 6),
                )
            )
            if end >= duration_seconds - 1e-6:
                break
            start += self.stride_seconds
        return windows

    def _extract_source_fps_clip(
        self,
        video_path: Path,
        window: ClipWindow,
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{window.start_seconds:.6f}",
                "-i",
                str(video_path),
                "-t",
                f"{window.duration_seconds:.6f}",
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-fps_mode",
                "passthrough",
                str(output_path),
            ]
        )

    def _extract_tracking_frames(
        self,
        clip_path: Path,
        frame_dir: Path,
        window: ClipWindow,
    ) -> list[dict[str, Any]]:
        frame_dir.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(clip_path),
                "-vf",
                f"fps={self.tracking_fps:g}",
                "-start_number",
                "0",
                str(frame_dir / "%06d.jpg"),
            ]
        )
        paths = sorted(frame_dir.glob("*.jpg"))
        return [
            {
                "frame_index": index,
                "clip_timestamp_seconds": round(index / self.tracking_fps, 6),
                "source_timestamp_seconds": round(
                    window.start_seconds + index / self.tracking_fps, 6
                ),
                "path": str(path.resolve()),
            }
            for index, path in enumerate(paths)
        ]

    def prepare(self, video_path: Path, output_root: Path, video_id: str) -> dict[str, Any]:
        info = self.probe(video_path)
        clips = []
        for window in self.windows(info.duration_seconds):
            clip_root = output_root / "clips" / video_id / window.clip_id
            clip_path = clip_root / "source_fps.mp4"
            self._extract_source_fps_clip(video_path, window, clip_path)
            clip_info = self.probe(clip_path)
            if abs(clip_info.fps - info.fps) > 0.02:
                raise RuntimeError(
                    f"Source FPS changed for {window.clip_id}: {info.fps} -> {clip_info.fps}"
                )
            frames = self._extract_tracking_frames(
                clip_path, clip_root / "tracking_frames_5fps", window
            )
            clips.append(
                {
                    "clip_id": window.clip_id,
                    "clip_index": window.clip_index,
                    "start_seconds": window.start_seconds,
                    "end_seconds": window.end_seconds,
                    "duration_seconds": window.duration_seconds,
                    "clip_path": str(clip_path.resolve()),
                    "source_fps": info.fps,
                    "source_fps_fraction": info.fps_fraction,
                    "tracking_fps": self.tracking_fps,
                    "tracking_frames": frames,
                }
            )
        manifest = {
            "video_id": video_id,
            "source_video": str(video_path.resolve()),
            "video_info": {
                "duration_seconds": info.duration_seconds,
                "fps": info.fps,
                "fps_fraction": info.fps_fraction,
                "width": info.width,
                "height": info.height,
                "frame_count": info.frame_count,
            },
            "preprocessing": {
                "source_fps_preserved_in_clips": True,
                "clip_seconds": self.clip_seconds,
                "stride_seconds": self.stride_seconds,
                "tracking_fps": self.tracking_fps,
                "dam_representative_frames_are_not_tracking_frames": True,
            },
            "clips": clips,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "preprocessing.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest
