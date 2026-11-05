from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor, pipeline


MODEL_ID = "facebook/sam3"
COLORS = ((255, 75, 75), (51, 204, 102), (52, 152, 219), (241, 196, 15), (155, 89, 182), (230, 126, 34), (26, 188, 156), (236, 112, 99))


class Sam3PeriodicTracker:
    def __init__(
        self,
        redetect_interval_frames: int = 10,
        model_id: str = MODEL_ID,
        video_model_class: type = Sam3TrackerVideoModel,
        video_processor_class: type = Sam3TrackerVideoProcessor,
        stage_name: str = "sam3_discovery_and_tracking",
        official_cuda_required: bool = True,
    ) -> None:
        self.redetect_interval_frames = redetect_interval_frames
        self.model_id = model_id
        self.video_model_class = video_model_class
        self.video_processor_class = video_processor_class
        self.stage_name = stage_name
        self.official_cuda_required = official_cuda_required
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.existing_iou_threshold = 0.50
        self.existing_coverage_threshold = 0.80
        self.min_area_ratio = 0.0004
        self.max_area_ratio = 0.90

    @staticmethod
    def mask_array(value: object) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.squeeze(np.asarray(value)).astype(bool)

    @staticmethod
    def bounds(mask: np.ndarray) -> list[int] | None:
        rows, columns = np.where(mask)
        if not len(columns):
            return None
        return [int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1]

    @staticmethod
    def overlap(candidate: np.ndarray, existing: np.ndarray) -> tuple[float, float]:
        intersection = int(np.logical_and(candidate, existing).sum())
        union = int(np.logical_or(candidate, existing).sum())
        area = int(candidate.sum())
        return (intersection / union if union else 0.0, intersection / area if area else 0.0)

    @staticmethod
    def label(draw: ImageDraw.ImageDraw, box: list[int], text: str, color: tuple[int, int, int]) -> None:
        font = ImageFont.load_default()
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        height = bottom - top + 6
        y = max(0, box[1] - height)
        draw.rectangle((box[0], y, box[0] + right - left + 8, y + height), fill=color)
        draw.text((box[0] + 4, y + 2), text, fill=(0, 0, 0), font=font)

    def discover(
        self, frames: list[Image.Image], indices: list[int], output_dir: Path,
        token: str | None, generator: Any | None = None,
    ) -> list[dict[str, Any]]:
        owns_generator = generator is None
        if generator is None:
            generator = pipeline("mask-generation", model=self.model_id, token=token, device=self.device)
        rounds = []
        for round_index, frame_index in enumerate(indices):
            result = generator(frames[frame_index], points_per_batch=64)
            round_dir = output_dir / "discovery" / f"frame_{frame_index:06d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            candidates = []
            for candidate_index, mask_value in enumerate(result["masks"], start=1):
                mask = self.mask_array(mask_value)
                area_ratio = float(mask.mean())
                if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                    continue
                path = Path("discovery") / f"frame_{frame_index:06d}" / f"candidate_{candidate_index:03d}.png"
                Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(output_dir / path)
                scores = result.get("scores", [])
                score = float(scores[candidate_index - 1]) if candidate_index <= len(scores) else None
                candidates.append({
                    "candidate_id": f"R{round_index:02d}_M{candidate_index:03d}",
                    "mask_path": str(path),
                    "bbox_xyxy": self.bounds(mask),
                    "mask_area_ratio": round(area_ratio, 8),
                    "score": round(score, 6) if score is not None else None,
                })
            rounds.append({"round_index": round_index, "frame_index": frame_index, "candidate_count": len(candidates), "candidates": candidates})
        if owns_generator:
            del generator
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
        return rounds

    def existing_masks(self, tracks: list[dict[str, Any]], frame_index: int, output_dir: Path) -> list[np.ndarray]:
        result = []
        for track in tracks:
            item = next((frame for frame in track["frames"] if frame["frame_index"] == frame_index and frame["mask_area_pixels"] > 0), None)
            if item:
                result.append(self.mask_array(Image.open(output_dir / item["mask_path"])))
        return result

    def track_group(self, model: Sam3TrackerVideoModel, processor: Sam3TrackerVideoProcessor, frames: list[Image.Image], records: list[dict[str, Any]], start_index: int, tracks: list[dict[str, Any]], output_dir: Path) -> None:
        if not tracks:
            return
        session = processor.init_video_session(
            video=frames[start_index:], inference_device=self.device,
            inference_state_device="cpu", processing_device="cpu",
            video_storage_device="cpu", dtype=torch.float32,
        )
        masks = [self.mask_array(Image.open(output_dir / track["discovery_mask_path"])).astype(np.uint8) for track in tracks]
        processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=0,
            obj_ids=list(range(1, len(tracks) + 1)), input_masks=masks,
            original_size=(frames[0].height, frames[0].width),
        )
        model(inference_session=session, frame_idx=0)
        for output in model.propagate_in_video_iterator(session, show_progress_bar=False):
            absolute_index = start_index + int(output.frame_idx)
            processed = processor.post_process_masks(
                [output.pred_masks.detach().cpu()],
                original_sizes=[[session.video_height, session.video_width]],
                binarize=True,
            )[0]
            logits = output.object_score_logits.detach().float().cpu().reshape(-1).tolist() if output.object_score_logits is not None else [None] * len(tracks)
            for position, track in enumerate(tracks):
                mask = self.mask_array(processed[position])
                relative = Path("masks") / track["track_id"] / f"{absolute_index:06d}.png"
                (output_dir / relative).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(output_dir / relative)
                record = records[absolute_index]
                track["frames"].append({
                    "frame_index": absolute_index,
                    "source_timestamp_seconds": record["source_timestamp_seconds"],
                    "source_frame_path": record["path"],
                    "mask_path": str(relative),
                    "bbox_xyxy": self.bounds(mask),
                    "mask_area_pixels": int(mask.sum()),
                    "object_score_logit": round(float(logits[position]), 6) if logits[position] is not None else None,
                })

    def render(self, frames: list[Image.Image], tracks: list[dict[str, Any]], output_dir: Path) -> None:
        overlay_dir = output_dir / "overlays"
        overlay_dir.mkdir(exist_ok=True)
        for frame_index, frame in enumerate(frames):
            overlay = frame.convert("RGBA")
            boxes = []
            for index, track in enumerate(tracks):
                item = next((value for value in track["frames"] if value["frame_index"] == frame_index and value["bbox_xyxy"]), None)
                if item is None:
                    continue
                color = COLORS[index % len(COLORS)]
                mask = Image.open(output_dir / item["mask_path"]).convert("L")
                tint = Image.new("RGBA", frame.size, (*color, 0))
                tint.putalpha(mask.point(lambda value: 76 if value else 0))
                overlay = Image.alpha_composite(overlay, tint)
                boxes.append((item["bbox_xyxy"], track["track_id"], color))
            draw = ImageDraw.Draw(overlay)
            for box, track_id, color in boxes:
                draw.rectangle(box, outline=(*color, 255), width=3)
                self.label(draw, box, track_id, color)
            overlay.save(overlay_dir / f"{frame_index:06d}.png")

    def track_discoveries(
        self,
        clip: dict[str, Any],
        output_dir: Path,
        frames: list[Image.Image],
        records: list[dict[str, Any]],
        rounds: list[dict[str, Any]],
        model: Any,
        processor: Any,
    ) -> list[dict[str, Any]]:
        tracks: list[dict[str, Any]] = []
        next_id = 1
        for round_item in rounds:
            frame_index = int(round_item["frame_index"])
            compared = self.existing_masks(tracks, frame_index, output_dir)
            accepted_masks: list[np.ndarray] = []
            new_tracks = []
            rejections = []
            for candidate in round_item["candidates"]:
                mask = self.mask_array(Image.open(output_dir / candidate["mask_path"]))
                scores = [self.overlap(mask, existing) for existing in compared + accepted_masks]
                max_iou = max((score[0] for score in scores), default=0.0)
                max_coverage = max((score[1] for score in scores), default=0.0)
                if max_iou >= self.existing_iou_threshold or max_coverage >= self.existing_coverage_threshold:
                    rejections.append({"candidate_id": candidate["candidate_id"], "reason": "duplicate_existing_track", "max_iou": round(max_iou, 6), "max_candidate_coverage": round(max_coverage, 6)})
                    continue
                track_id = f"C{int(clip['clip_index']):03d}_T{next_id:03d}"
                next_id += 1
                track = {
                    "track_id": track_id,
                    "discovered_at_frame_index": frame_index,
                    "discovered_at_source_timestamp_seconds": records[frame_index]["source_timestamp_seconds"],
                    "periodic_discovery": frame_index > 0,
                    "discovery_mask_path": candidate["mask_path"],
                    "frames": [],
                }
                tracks.append(track)
                new_tracks.append(track)
                accepted_masks.append(mask)
            round_item["new_track_count"] = len(new_tracks)
            round_item["duplicate_rejection_count"] = len(rejections)
            round_item["rejections"] = rejections
            self.track_group(model, processor, frames, records, frame_index, new_tracks, output_dir)
        return tracks

    def run(
        self, clip: dict[str, Any], output_dir: Path, render_outputs: bool = True,
        generator: Any | None = None, model: Any | None = None,
        processor: Any | None = None,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        records = list(clip["tracking_frames"])
        frames = [Image.open(item["path"]).convert("RGB") for item in records]
        indices = list(range(0, len(frames), self.redetect_interval_frames))
        token = os.getenv("HF_TOKEN") or None
        rounds = self.discover(frames, indices, output_dir, token, generator=generator)
        if model is None:
            model = self.video_model_class.from_pretrained(self.model_id, token=token).to(self.device)
        if processor is None:
            processor = self.video_processor_class.from_pretrained(self.model_id, token=token)
        tracks = self.track_discoveries(
            clip, output_dir, frames, records, rounds, model, processor,
        )
        if render_outputs:
            self.render(frames, tracks, output_dir)
        payload = {
            "stage": self.stage_name,
            "status": "completed",
            "model": self.model_id,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "device": self.device,
            "official_cuda_prerequisite_met": (
                self.device == "cuda" if self.official_cuda_required else None
            ),
            "clip_id": clip["clip_id"],
            "clip_index": clip["clip_index"],
            "tracking_fps": clip["tracking_fps"],
            "tracking_frame_count": len(frames),
            "track_count": len(tracks),
            "periodic_new_track_count": sum(1 for item in tracks if item["periodic_discovery"]),
            "discovery_rounds": rounds,
            "tracks": tracks,
        }
        (output_dir / "tracks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"clip_id": clip["clip_id"], "tracks": len(tracks), "device": self.device}))
        return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessing-manifest", type=Path, required=True)
    parser.add_argument("--clip-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--redetect-interval-frames", type=int, default=10)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.preprocessing_manifest.read_text(encoding="utf-8"))
    Sam3PeriodicTracker(args.redetect_interval_frames).run(
        manifest["clips"][args.clip_index], args.output_dir,
        render_outputs=not args.no_render,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
