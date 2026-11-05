from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import Sam2VideoModel, Sam2VideoProcessor

from sam3_track import Sam3PeriodicTracker


MODEL_ID = "facebook/sam2.1-hiera-tiny"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessing-manifest", type=Path, required=True)
    parser.add_argument("--clip-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--redetect-interval-frames", type=int, default=10)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.preprocessing_manifest.read_text(encoding="utf-8"))
    tracker = Sam3PeriodicTracker(
        redetect_interval_frames=args.redetect_interval_frames,
        model_id=MODEL_ID,
        video_model_class=Sam2VideoModel,
        video_processor_class=Sam2VideoProcessor,
        stage_name="sam2_discovery_and_tracking",
        official_cuda_required=False,
    )
    tracker.run(
        manifest["clips"][args.clip_index], args.output_dir,
        render_outputs=not args.no_render,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
