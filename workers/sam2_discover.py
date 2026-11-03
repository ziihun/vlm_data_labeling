from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline


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


class Sam2AutomaticMaskWorker:
    """Runs SAM 2 automatic mask generation once and keeps every accepted proposal."""

    def __init__(
        self,
        points_per_crop: int,
        points_per_batch: int,
        pred_iou_thresh: float,
        stability_score_thresh: float,
    ) -> None:
        self.points_per_crop = points_per_crop
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.device = self._choose_device()
        self.generator = pipeline(
            "mask-generation",
            model=MODEL_ID,
            device=self.device,
        )

    @staticmethod
    def _choose_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _as_bool_mask(value: object) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        return np.squeeze(array).astype(bool)

    @staticmethod
    def _box(value: object, width: int, height: int) -> list[int]:
        if isinstance(value, torch.Tensor):
            values = value.detach().cpu().tolist()
        else:
            values = list(value)  # type: ignore[arg-type]
        x1, y1, x2, y2 = [int(round(float(item))) for item in values]
        return [
            max(0, min(width - 1, x1)),
            max(0, min(height - 1, y1)),
            max(1, min(width, x2)),
            max(1, min(height, y2)),
        ]

    @staticmethod
    def _label(draw: ImageDraw.ImageDraw, box: list[int], label: str, color: tuple[int, int, int]) -> None:
        font = ImageFont.load_default()
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_width = max(44, right - left + 8)
        text_height = max(16, bottom - top + 6)
        x = box[0]
        y = max(0, box[1] - text_height)
        draw.rectangle((x, y, x + text_width, y + text_height), fill=color)
        draw.text((x + 4, y + 2), label, fill=(0, 0, 0), font=font)

    def run_frame(
        self,
        image_path: Path,
        frame_index: int,
        timestamp_seconds: float,
        output_dir: Path,
    ) -> dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = output_dir / "masks"
        crops_dir = output_dir / "crops"
        masks_dir.mkdir(exist_ok=True)
        crops_dir.mkdir(exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        generated = self.generator(
            image,
            points_per_batch=self.points_per_batch,
            points_per_crop=self.points_per_crop,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            output_bboxes_mask=True,
        )
        masks = generated["masks"]
        scores = generated["scores"]
        boxes = generated["bounding_boxes"]

        boxed = image.copy()
        box_draw = ImageDraw.Draw(boxed)
        overlay = image.convert("RGBA")
        proposals: list[dict[str, object]] = []

        for proposal_index, (mask_value, score_value, box_value) in enumerate(
            zip(masks, scores, boxes, strict=True), start=1
        ):
            region_id = f"F{frame_index:02d}_M{proposal_index:03d}"
            color = COLORS[(proposal_index - 1) % len(COLORS)]
            mask = self._as_bool_mask(mask_value)
            box = self._box(box_value, width, height)
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            mask_name = f"{region_id}.png"
            crop_name = f"{region_id}.jpg"
            mask_image.save(masks_dir / mask_name)

            padding = 8
            crop_box = (
                max(0, box[0] - padding),
                max(0, box[1] - padding),
                min(width, box[2] + padding),
                min(height, box[3] + padding),
            )
            image.crop(crop_box).save(crops_dir / crop_name, quality=93)

            tint = Image.new("RGBA", image.size, (*color, 0))
            tint.putalpha(mask_image.point(lambda value: 80 if value else 0))
            overlay = Image.alpha_composite(overlay, tint)
            box_draw.rectangle(box, outline=color, width=3)
            self._label(box_draw, box, region_id, color)

            proposals.append(
                {
                    "region_id": region_id,
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "bbox_xyxy": box,
                    "score": round(float(score_value), 6),
                    "mask_area_pixels": int(mask.sum()),
                    "mask_area_ratio": round(float(mask.mean()), 7),
                    "mask_path": f"masks/{mask_name}",
                    "crop_path": f"crops/{crop_name}",
                }
            )

        overlay_draw = ImageDraw.Draw(overlay)
        for proposal_index, proposal in enumerate(proposals, start=1):
            color = COLORS[(proposal_index - 1) % len(COLORS)]
            box = proposal["bbox_xyxy"]
            overlay_draw.rectangle(box, outline=(*color, 255), width=3)
            self._label(overlay_draw, box, str(proposal["region_id"]), color)

        boxed.save(output_dir / "all_bboxes.jpg", quality=95)
        overlay.save(output_dir / "all_masks_overlay.png")
        result = {
            "frame_index": frame_index,
            "timestamp_seconds": timestamp_seconds,
            "source_image": str(image_path),
            "image_size": [width, height],
            "raw_proposal_count": len(masks),
            "retained_proposal_count": len(proposals),
            "filtering": {
                "note": "All proposals returned by SAM2 after its score, stability, and built-in crop NMS thresholds are retained; no extra cap or custom filter is applied.",
                "pred_iou_thresh": self.pred_iou_thresh,
                "stability_score_thresh": self.stability_score_thresh,
                "crops_nms_thresh": 0.7,
            },
            "proposals": proposals,
            "artifacts": {
                "all_bboxes": "all_bboxes.jpg",
                "all_masks_overlay": "all_masks_overlay.png",
            },
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    def run_manifest(self, manifest_path: Path, output_dir: Path) -> dict[str, object]:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame_results = []
        for frame in manifest["frames"]:
            frame_results.append(
                self.run_frame(
                    Path(frame["path"]),
                    int(frame["index"]),
                    float(frame["timestamp_seconds"]),
                    output_dir / f"frame_{int(frame['index']):02d}",
                )
            )
        result = {
            "stage": "sam2_automatic_mask_discovery",
            "status": "executed",
            "requested_model": MODEL_ID,
            "executed_model": MODEL_ID,
            "device": self.device,
            "frame_count": len(frame_results),
            "total_raw_proposals": sum(int(item["raw_proposal_count"]) for item in frame_results),
            "total_retained_proposals": sum(
                int(item["retained_proposal_count"]) for item in frame_results
            ),
            "frames": frame_results,
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover all accepted SAM 2 masks in sampled frames")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--points-per-crop", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.82)
    parser.add_argument("--stability-score-thresh", type=float, default=0.90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker = Sam2AutomaticMaskWorker(
        points_per_crop=args.points_per_crop,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
    )
    result = worker.run_manifest(args.manifest, args.output_dir)
    print(json.dumps({key: result[key] for key in ("status", "device", "frame_count", "total_retained_proposals")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
