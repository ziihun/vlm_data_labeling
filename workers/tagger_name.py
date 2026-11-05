from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


NON_OBJECT_TAGS = {
    "attach", "attached", "black", "blue", "brown", "carry", "catch", "ceiling",
    "dark", "electronic", "facility", "factory", "fill", "floor", "gray", "green",
    "hang", "illuminate", "job", "light", "material", "night view", "orange", "pink",
    "purple", "red", "ride", "room", "sing", "sit", "skate", "square", "stand",
    "trick", "walk", "warehouse", "wear", "white", "workshop", "writing", "yellow",
}


def choose_device(torch: Any) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_outputs(
    output_dir: Path,
    *,
    model: str,
    runner: str,
    items: list[dict[str, Any]],
    raw: list[dict[str, Any]],
    started: float,
    score_note: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "failure.json").unlink(missing_ok=True)
    payload = {"model": model, "runner": runner, "items": items}
    metrics = {
        "status": "completed",
        "track_count": len(items),
        "total_latency_seconds": round(time.perf_counter() - started, 3),
        "score_note": score_note,
    }
    (output_dir / "naming.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "raw_response.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def top_candidates(scores: dict[str, float], count: int = 5) -> list[dict[str, Any]]:
    return [
        {"label": label, "score": round(float(score), 4)}
        for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:count]
    ]


def run_ram_plus(assets: dict[str, Any], args: argparse.Namespace) -> None:
    import torch
    from ram import get_transform
    from ram.models import ram_plus

    started = time.perf_counter()
    device = choose_device(torch)
    transform = get_transform(image_size=384)
    model = ram_plus(pretrained=str(args.checkpoint), image_size=384, vit="swin_l").eval().to(device)
    items: list[dict[str, Any]] = []
    raw_items: list[dict[str, Any]] = []
    for index, track in enumerate(assets["tracks"], start=1):
        track_started = time.perf_counter()
        tensors = []
        for evidence in track["evidence"]:
            with Image.open(evidence["masked_crop_path"]) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            english_tags, _ = model.generate_tag(batch)
        per_frame = [[tag.strip() for tag in text.split(" | ") if tag.strip()] for text in english_tags]
        counts = Counter(tag for tags in per_frame for tag in set(tags))
        scores = {tag: count / len(per_frame) for tag, count in counts.items()}
        all_candidates = top_candidates(scores, count=10)
        object_scores = {tag: score for tag, score in scores.items() if tag.lower() not in NON_OBJECT_TAGS}
        candidates = top_candidates(object_scores)
        name = candidates[0]["label"] if candidates else "unknown"
        confidence = candidates[0]["score"] if candidates else 0.0
        item = {
            "track_id": track["track_id"],
            "canonical_object_name": name,
            "visible_attributes": {},
            "observable_state": "unknown",
            "confidence": confidence,
            "uncertainty_reason": "RAM++ returns image tags, not a target-grounded single object name.",
            "top_candidates": candidates,
            "latency_seconds": round(time.perf_counter() - track_started, 3),
            "parse_error": None,
        }
        items.append(item)
        raw_items.append({"track_id": track["track_id"], "per_frame_tags": per_frame, "all_top_tags": all_candidates})
        print(json.dumps({"progress": f"{index}/{len(assets['tracks'])}", "track_id": track["track_id"], "name": name}), flush=True)
    save_outputs(
        args.output_dir,
        model="RAM++ Swin-Large 14M",
        runner="recognize_anything_plus",
        items=items,
        raw=raw_items,
        started=started,
        score_note="confidence is Track crop support ratio after a conservative non-object tag filter, not a calibrated probability",
    )


def load_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def run_siglip2(assets: dict[str, Any], args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModel, AutoProcessor

    started = time.perf_counter()
    device = choose_device(torch)
    labels = load_labels(args.labels_file)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().to(device)
    text_prompts = [f"This is a photo of {label}." for label in labels]
    text_feature_batches = []
    text_batch_size = 256
    for start in range(0, len(text_prompts), text_batch_size):
        prompt_batch = text_prompts[start : start + text_batch_size]
        text_inputs = processor(
            text=prompt_batch,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        with torch.no_grad():
            batch_features = model.get_text_features(**text_inputs)
            batch_features = batch_features / batch_features.norm(p=2, dim=-1, keepdim=True)
        text_feature_batches.append(batch_features)
        print(json.dumps({
            "text_progress": f"{min(start + text_batch_size, len(text_prompts))}/{len(text_prompts)}"
        }), flush=True)
    text_features = torch.cat(text_feature_batches, dim=0)
    items: list[dict[str, Any]] = []
    raw_items: list[dict[str, Any]] = []
    for index, track in enumerate(assets["tracks"], start=1):
        track_started = time.perf_counter()
        images = []
        for evidence in track["evidence"]:
            with Image.open(evidence["original_crop_path"]) as image:
                images.append(image.convert("RGB").copy())
        image_inputs = processor(images=images, return_tensors="pt")
        image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            logits = logits * model.logit_scale.exp() + model.logit_bias
            means = logits.sigmoid().mean(dim=0).float().cpu()
        best_values, best_indices = torch.topk(means, k=min(5, len(labels)))
        candidates = [
            {"label": labels[int(label_index)], "score": round(float(value), 4)}
            for value, label_index in zip(best_values, best_indices, strict=True)
        ]
        item = {
            "track_id": track["track_id"],
            "canonical_object_name": candidates[0]["label"],
            "visible_attributes": {},
            "observable_state": "unknown",
            "confidence": candidates[0]["score"],
            "uncertainty_reason": f"Zero-shot selection is restricted to a fixed {len(labels):,}-label vocabulary.",
            "top_candidates": candidates,
            "latency_seconds": round(time.perf_counter() - track_started, 3),
            "parse_error": None,
        }
        items.append(item)
        raw_items.append({"track_id": track["track_id"], "top_candidates": candidates})
        print(json.dumps({"progress": f"{index}/{len(assets['tracks'])}", "track_id": track["track_id"], "name": item["canonical_object_name"]}), flush=True)
    save_outputs(
        args.output_dir,
        model=args.model,
        runner="siglip2_zero_shot",
        items=items,
        raw=raw_items,
        started=started,
        score_note="confidence is mean sigmoid similarity within the fixed label bank, not VLM confidence",
    )


def binary_mask_iou_and_coverage(first: Any, second: Any) -> tuple[float, float]:
    intersection = (first & second).sum()
    union = (first | second).sum()
    first_area = first.sum()
    iou = float(intersection / union) if union else 0.0
    coverage = float(intersection / first_area) if first_area else 0.0
    return iou, coverage


def run_yoloe(assets: dict[str, Any], args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from ultralytics import YOLOE

    started = time.perf_counter()
    device = choose_device(torch)
    source_paths = sorted({
        evidence["source_frame_path"]
        for track in assets["tracks"]
        for evidence in track["evidence"]
        if evidence.get("source_frame_path")
    })
    if not source_paths:
        raise RuntimeError("assets.json has no source_frame_path; rerun naming_experiment prepare with SAM2 tracks.json")
    model = YOLOE(args.model)
    result_by_path: dict[str, Any] = {}
    for index, source_path in enumerate(source_paths, start=1):
        result_by_path[source_path] = model.predict(
            source=source_path,
            imgsz=640,
            conf=0.05,
            device=device,
            retina_masks=True,
            verbose=False,
        )[0]
        print(json.dumps({"frame_progress": f"{index}/{len(source_paths)}", "source": Path(source_path).name}), flush=True)
    items: list[dict[str, Any]] = []
    raw_items: list[dict[str, Any]] = []
    per_track_latency = (time.perf_counter() - started) / max(1, len(assets["tracks"]))
    for track_index, track in enumerate(assets["tracks"], start=1):
        label_scores: dict[str, float] = defaultdict(float)
        label_counts: Counter[str] = Counter()
        matches = []
        for evidence in track["evidence"]:
            source_path = evidence.get("source_frame_path")
            if not source_path:
                continue
            result = result_by_path[source_path]
            if result.masks is None or result.boxes is None:
                matches.append({"timestamp": evidence["source_timestamp_seconds"], "match": None})
                continue
            sam_mask = np.asarray(Image.open(evidence["mask_path"]).convert("L")) > 127
            predicted_masks = result.masks.data.cpu().numpy() > 0.5
            best = None
            for detection_index, predicted_mask in enumerate(predicted_masks):
                if predicted_mask.shape != sam_mask.shape:
                    predicted_mask = np.asarray(
                        Image.fromarray(predicted_mask.astype("uint8") * 255).resize(
                            (sam_mask.shape[1], sam_mask.shape[0]), Image.Resampling.NEAREST
                        )
                    ) > 127
                iou, coverage = binary_mask_iou_and_coverage(sam_mask, predicted_mask)
                confidence = float(result.boxes.conf[detection_index].cpu())
                class_id = int(result.boxes.cls[detection_index].cpu())
                label = str(result.names[class_id])
                match_score = iou * confidence
                candidate = {
                    "label": label,
                    "detector_confidence": round(confidence, 4),
                    "mask_iou": round(iou, 4),
                    "sam_coverage": round(coverage, 4),
                    "match_score": round(match_score, 4),
                }
                if best is None or match_score > best["match_score"]:
                    best = candidate
            accepted = best if best and best["mask_iou"] >= 0.10 else None
            matches.append({"timestamp": evidence["source_timestamp_seconds"], "match": accepted, "best_rejected": best if accepted is None else None})
            if accepted:
                label_scores[accepted["label"]] += accepted["match_score"]
                label_counts[accepted["label"]] += 1
        normalized_scores = {
            label: score / label_counts[label]
            for label, score in label_scores.items()
        }
        candidates = top_candidates(normalized_scores)
        name = candidates[0]["label"] if candidates else "unknown"
        winning_confidences = [
            entry["match"]["detector_confidence"]
            for entry in matches
            if entry.get("match") and entry["match"]["label"] == name
        ]
        confidence = sum(winning_confidences) / len(winning_confidences) if winning_confidences else 0.0
        item = {
            "track_id": track["track_id"],
            "canonical_object_name": name,
            "visible_attributes": {},
            "observable_state": "unknown",
            "confidence": round(confidence, 4),
            "uncertainty_reason": "unknown" if candidates else "No prompt-free detection passed the SAM2 mask match threshold.",
            "top_candidates": candidates,
            "latency_seconds": round(per_track_latency, 3),
            "parse_error": None,
            "matched_evidence_count": sum(entry.get("match") is not None for entry in matches),
        }
        items.append(item)
        raw_items.append({"track_id": track["track_id"], "matches": matches})
        print(json.dumps({"progress": f"{track_index}/{len(assets['tracks'])}", "track_id": track["track_id"], "name": name}), flush=True)
    save_outputs(
        args.output_dir,
        model=args.model,
        runner="yoloe_prompt_free_sam2_mask_match",
        items=items,
        raw=raw_items,
        started=started,
        score_note="confidence is YOLOE detector confidence after SAM2 mask IoU/coverage matching",
    )


def main() -> int:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    parser = argparse.ArgumentParser(description="Non-generative naming baselines for saved SAM2 Tracks")
    parser.add_argument("--method", choices=("ram_plus", "siglip2", "yoloe_prompt_free"), required=True)
    parser.add_argument("--assets-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--labels-file", type=Path)
    parser.add_argument("--model")
    args = parser.parse_args()
    assets = json.loads(args.assets_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.method == "ram_plus":
            if args.checkpoint is None:
                parser.error("ram_plus requires --checkpoint")
            run_ram_plus(assets, args)
        elif args.method == "siglip2":
            if args.labels_file is None:
                parser.error("siglip2 requires --labels-file")
            args.model = args.model or "google/siglip2-base-patch16-256"
            run_siglip2(assets, args)
        else:
            args.model = args.model or "models/yoloe-26s-seg-pf.pt"
            run_yoloe(assets, args)
    except Exception as exc:
        failure = {
            "status": "failed",
            "phase": "inference",
            "method": args.method,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        }
        (args.output_dir / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
