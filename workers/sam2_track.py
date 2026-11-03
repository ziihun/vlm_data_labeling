from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam2VideoModel, Sam2VideoProcessor, pipeline


MODEL_ID = "facebook/sam2.1-hiera-tiny"
COLORS = (
    (255, 75, 75),
    (51, 204, 102),
    (52, 152, 219),
    (241, 196, 15),
    (155, 89, 182),
    (230, 126, 34),
    (26, 188, 156),
    (236, 112, 99),
)


class Sam2ClipTracker:
    """Automatic first-frame discovery followed by true SAM2 video propagation."""

    def __init__(
        self,
        points_per_crop: int = 16,
        pred_iou_thresh: float = 0.90,
        stability_score_thresh: float = 0.92,
        max_dam_frames: int = 8,
    ) -> None:
        self.points_per_crop = points_per_crop
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.max_dam_frames = max_dam_frames
        self.device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @staticmethod
    def _mask_array(value: object) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        return np.squeeze(array).astype(bool)

    @staticmethod
    def _bounds(mask: np.ndarray) -> list[int] | None:
        rows, columns = np.where(mask)
        if len(columns) == 0:
            return None
        return [
            int(columns.min()),
            int(rows.min()),
            int(columns.max()) + 1,
            int(rows.max()) + 1,
        ]

    @staticmethod
    def _representative_indices(indices: list[int], limit: int) -> list[int]:
        if len(indices) <= limit:
            return indices
        positions = np.linspace(0, len(indices) - 1, limit, dtype=int)
        return [indices[int(position)] for position in positions]

    @staticmethod
    def _label(draw: ImageDraw.ImageDraw, box: list[int], text: str, color: tuple[int, int, int]) -> None:
        font = ImageFont.load_default()
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        width = right - left + 8
        height = bottom - top + 6
        x, y = box[0], max(0, box[1] - height)
        draw.rectangle((x, y, x + width, y + height), fill=color)
        draw.text((x + 4, y + 2), text, fill=(0, 0, 0), font=font)

    def _discover(self, first_frame: Image.Image) -> tuple[list[np.ndarray], list[float]]:
        generator = pipeline("mask-generation", model=MODEL_ID, device=self.device)
        output = generator(
            first_frame,
            points_per_batch=64,
            points_per_crop=self.points_per_crop,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            output_bboxes_mask=True,
        )
        masks = [self._mask_array(mask) for mask in output["masks"]]
        scores = [round(float(score), 6) for score in output["scores"]]
        del generator, output
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        if not masks:
            raise RuntimeError("SAM2 automatic discovery returned no initial masks")
        return masks, scores

    def run(self, clip: dict[str, object], output_dir: Path) -> dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_records = list(clip["tracking_frames"])  # type: ignore[arg-type]
        frames = [Image.open(record["path"]).convert("RGB") for record in frame_records]
        if not frames:
            raise RuntimeError("Clip has no 5 FPS tracking frames")
        width, height = frames[0].size
        initial_masks, initial_scores = self._discover(frames[0])
        # Keep our immutable copy because the processor owns/may mutate its obj_ids input.
        object_ids = tuple(range(1, len(initial_masks) + 1))

        model = Sam2VideoModel.from_pretrained(MODEL_ID).to(self.device)
        processor = Sam2VideoProcessor.from_pretrained(MODEL_ID)
        session = processor.init_video_session(
            video=frames,
            inference_device=self.device,
            inference_state_device="cpu",
            video_storage_device="cpu",
            dtype=torch.float32,
        )
        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=0,
            obj_ids=list(object_ids),
            input_masks=[mask.astype(np.uint8) for mask in initial_masks],
            original_size=(height, width),
        )
        model(inference_session=session, frame_idx=0)

        clip_index = int(clip["clip_index"])
        tracks: dict[int, dict[str, object]] = {}
        for object_id, score in zip(object_ids, initial_scores, strict=True):
            track_id = f"C{clip_index:03d}_T{object_id:03d}"
            track_dir = output_dir / "tracks" / track_id
            (track_dir / "masks").mkdir(parents=True, exist_ok=True)
            (track_dir / "dam_frames").mkdir(exist_ok=True)
            tracks[object_id] = {
                "track_id": track_id,
                "clip_id": clip["clip_id"],
                "initial_sam_score": score,
                "frames": [],
                "representative_frames": [],
            }

        overlays_dir = output_dir / "overlays"
        overlays_dir.mkdir(exist_ok=True)
        for result in model.propagate_in_video_iterator(session, show_progress_bar=False):
            frame_index = int(result.frame_idx)
            masks = processor.post_process_masks(
                [result.pred_masks.detach().cpu()],
                original_sizes=[[session.video_height, session.video_width]],
                binarize=True,
            )[0]
            base = frames[frame_index].convert("RGBA")
            overlay = base.copy()
            draw_boxes: list[tuple[list[int], str, tuple[int, int, int]]] = []
            scores = (
                result.object_score_logits.detach().float().cpu().reshape(-1).tolist()
                if result.object_score_logits is not None
                else [None] * len(object_ids)
            )
            for position, object_id in enumerate(object_ids):
                track = tracks[object_id]
                track_id = str(track["track_id"])
                mask = self._mask_array(masks[position])
                mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
                relative_mask = Path("tracks") / track_id / "masks" / f"{frame_index:06d}.png"
                mask_image.save(output_dir / relative_mask)
                box = self._bounds(mask)
                color = COLORS[(object_id - 1) % len(COLORS)]
                if box is not None:
                    tint = Image.new("RGBA", base.size, (*color, 0))
                    tint.putalpha(mask_image.point(lambda value: 72 if value else 0))
                    overlay = Image.alpha_composite(overlay, tint)
                    draw_boxes.append((box, f"T{object_id:03d}", color))
                record = frame_records[frame_index]
                track["frames"].append(  # type: ignore[union-attr]
                    {
                        "frame_index": frame_index,
                        "clip_timestamp_seconds": record["clip_timestamp_seconds"],
                        "source_timestamp_seconds": record["source_timestamp_seconds"],
                        "mask_path": str(relative_mask),
                        "bbox_xyxy": box,
                        "mask_area_pixels": int(mask.sum()),
                        "mask_area_ratio": round(float(mask.mean()), 8),
                        "object_score_logit": (
                            round(float(scores[position]), 6)
                            if scores[position] is not None
                            else None
                        ),
                    }
                )
            drawer = ImageDraw.Draw(overlay)
            for box, label, color in draw_boxes:
                drawer.rectangle(box, outline=(*color, 255), width=3)
                self._label(drawer, box, label, color)
            overlay.save(overlays_dir / f"{frame_index:06d}.png")

        for track in tracks.values():
            present_indices = [
                int(record["frame_index"])
                for record in track["frames"]  # type: ignore[union-attr]
                if int(record["mask_area_pixels"]) > 0
            ]
            selected = self._representative_indices(present_indices, self.max_dam_frames)
            frame_by_index = {
                int(record["frame_index"]): record
                for record in track["frames"]  # type: ignore[union-attr]
            }
            track_id = str(track["track_id"])
            for representative_index, frame_index in enumerate(selected):
                record = frame_by_index[frame_index]
                mask = Image.open(output_dir / record["mask_path"]).convert("L")
                rgba = frames[frame_index].convert("RGBA")
                rgba.putalpha(mask)
                relative_path = (
                    Path("tracks")
                    / track_id
                    / "dam_frames"
                    / f"rep_{representative_index:02d}_frame_{frame_index:06d}.png"
                )
                rgba.save(output_dir / relative_path)
                track["representative_frames"].append(  # type: ignore[union-attr]
                    {
                        "frame_index": frame_index,
                        "source_timestamp_seconds": record["source_timestamp_seconds"],
                        "rgba_path": str(relative_path),
                        "mask_path": record["mask_path"],
                    }
                )
            frame_list = track["frames"]  # type: ignore[assignment]
            track["first_seen_seconds"] = next(
                (item["source_timestamp_seconds"] for item in frame_list if item["mask_area_pixels"] > 0),
                None,
            )
            track["last_seen_seconds"] = next(
                (item["source_timestamp_seconds"] for item in reversed(frame_list) if item["mask_area_pixels"] > 0),
                None,
            )

        output = {
            "stage": "sam2_automatic_discovery_and_tracking",
            "status": "executed",
            "model": MODEL_ID,
            "device": self.device,
            "clip_id": clip["clip_id"],
            "clip_index": clip_index,
            "source_start_seconds": clip["start_seconds"],
            "source_end_seconds": clip["end_seconds"],
            "tracking_fps": clip["tracking_fps"],
            "tracking_frame_count": len(frames),
            "initial_object_count": len(tracks),
            "initial_discovery": {
                "points_per_crop": self.points_per_crop,
                "pred_iou_thresh": self.pred_iou_thresh,
                "stability_score_thresh": self.stability_score_thresh,
                "extra_object_cap": None,
            },
            "tracks": list(tracks.values()),
            "artifacts": {"overlays_dir": "overlays"},
        }
        (output_dir / "tracking.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "clip_id": clip["clip_id"],
                    "tracking_frames": len(frames),
                    "tracks": len(tracks),
                    "device": self.device,
                },
                ensure_ascii=False,
            )
        )
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM2.1 automatic discovery and 5 FPS tracking")
    parser.add_argument("--preprocessing-manifest", type=Path, required=True)
    parser.add_argument("--clip-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--points-per-crop", type=int, default=16)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.90)
    parser.add_argument("--stability-score-thresh", type=float, default=0.92)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.preprocessing_manifest.read_text(encoding="utf-8"))
    clip = payload["clips"][args.clip_index]
    tracker = Sam2ClipTracker(
        points_per_crop=args.points_per_crop,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
    )
    tracker.run(clip, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
