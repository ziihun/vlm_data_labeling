from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import Sam2Model, Sam2Processor


MODEL_ID = "facebook/sam2.1-hiera-tiny"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM 2.1 Tiny on one image and box")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--box", type=float, nargs=4, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def mask_bounds(mask: np.ndarray, fallback: list[int]) -> list[int]:
    rows, columns = np.where(mask)
    if len(columns) == 0 or len(rows) == 0:
        return fallback
    return [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())]


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    width, height = image.size
    box = [
        max(0, min(width - 1, int(round(args.box[0])))),
        max(0, min(height - 1, int(round(args.box[1])))),
        max(1, min(width, int(round(args.box[2])))),
        max(1, min(height, int(round(args.box[3])))),
    ]
    device = choose_device()

    model = Sam2Model.from_pretrained(MODEL_ID).to(device)
    processor = Sam2Processor.from_pretrained(MODEL_ID)
    inputs = processor(images=image, input_boxes=[[box]], return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
    )[0]
    scores = outputs.iou_scores[0, 0].detach().float().cpu()
    best_index = int(torch.argmax(scores).item())
    mask = masks[0, best_index].detach().cpu().numpy()
    if mask.dtype != np.bool_:
        mask = mask > 0

    mask_box = mask_bounds(mask, box)
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_image.save(args.output_dir / "mask.png")

    overlay = image.convert("RGBA")
    red = Image.new("RGBA", image.size, (255, 50, 50, 0))
    red.putalpha(mask_image.point(lambda value: 110 if value else 0))
    overlay = Image.alpha_composite(overlay, red)
    drawer = ImageDraw.Draw(overlay)
    drawer.rectangle(box, outline=(255, 220, 0, 255), width=4)
    drawer.rectangle(mask_box, outline=(0, 255, 120, 255), width=3)
    overlay.save(args.output_dir / "overlay.png")

    padding = 12
    crop_box = [
        max(0, mask_box[0] - padding),
        max(0, mask_box[1] - padding),
        min(width, mask_box[2] + padding + 1),
        min(height, mask_box[3] + padding + 1),
    ]
    rgba = image.convert("RGBA")
    alpha = Image.new("L", image.size, 35)
    alpha.paste(mask_image, (0, 0))
    rgba.putalpha(alpha)
    rgba.crop(crop_box).save(args.output_dir / "roi_crop.png")
    image.crop(crop_box).save(args.output_dir / "roi_crop_context.jpg", quality=95)

    result = {
        "stage": "sam2_segmentation",
        "status": "executed",
        "requested_model": MODEL_ID,
        "executed_model": MODEL_ID,
        "device": device,
        "input_box_pixels": box,
        "selected_mask_index": best_index,
        "predicted_iou_scores": [round(float(score), 6) for score in scores.tolist()],
        "selected_score": round(float(scores[best_index]), 6),
        "mask_box_pixels": mask_box,
        "crop_box_pixels": crop_box,
        "mask_area_pixels": int(mask.sum()),
        "mask_area_ratio": round(float(mask.mean()), 6),
        "artifacts": {
            "mask": "mask.png",
            "overlay": "overlay.png",
            "roi_crop": "roi_crop.png",
            "roi_crop_context": "roi_crop_context.jpg",
        },
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
