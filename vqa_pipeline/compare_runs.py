from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def load_run(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    segmenter = manifest["segmentation_model"]
    stage = f"01_{segmenter}"
    clip_results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / stage).glob("clip_*/tracks.json"))]
    merged = json.loads((root / stage / "tracks.json").read_text(encoding="utf-8"))
    naming = json.loads((root / "03_qwen_naming" / "naming.json").read_text(encoding="utf-8"))
    return {"root": root, "manifest": manifest, "segmenter": segmenter, "stage": stage, "clips": clip_results, "merged": merged, "naming": naming}


def write_comparison(sam2: dict[str, Any], sam3: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    def summary(run: dict[str, Any]) -> dict[str, Any]:
        return {
            "local": sum(item["track_count"] for item in run["clips"]),
            "periodic": sum(item["periodic_new_track_count"] for item in run["clips"]),
            "global": run["merged"]["track_count"],
            "names": [item["canonical_object_name"] for item in run["naming"]["items"]],
            "device": sorted({item["device"] for item in run["clips"]}),
        }
    s2, s3 = summary(sam2), summary(sam3)
    video_rows = ""
    for index in range(min(len(sam2["clips"]), len(sam3["clips"]))):
        video_rows += f"""<section><h2>Clip {index}</h2><div class='grid'><figure><video controls src='../{sam2['root'].name}/{sam2['stage']}/clip_{index:03d}/tracking_overlay.mp4'></video><figcaption>SAM 2.1 · BBox/Mask/Track ID</figcaption></figure><figure><video controls src='../{sam3['root'].name}/{sam3['stage']}/clip_{index:03d}/tracking_overlay.mp4'></video><figcaption>SAM 3 · BBox/Mask/Track ID</figcaption></figure></div></section>"""
    def names(values: list[str]) -> str:
        return "<br>".join(html.escape(value) for value in values)
    document = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>SAM 2 vs SAM 3</title><style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f3f5f7;max-width:1500px;margin:auto;padding:24px}}section{{background:#fff;padding:20px;border-radius:14px;margin:18px 0;box-shadow:0 2px 14px #0001}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #d5d8dc;padding:9px;vertical-align:top;text-align:left}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}video{{width:100%;background:#111}}figure{{margin:0}}</style></head><body><section><h1>SAM 2.1 vs SAM 3 · 동일 VQA 전처리 비교</h1><p>동일한 10초 Clip / 5초 Stride / 5 FPS / 2초 주기 재탐지 / 20% Crop / Qwen3.5 Naming을 적용했다.</p><table><thead><tr><th>Metric</th><th>SAM 2.1</th><th>SAM 3</th></tr></thead><tbody><tr><td>Device</td><td>{s2['device']}</td><td>{s3['device']}</td></tr><tr><td>Local Tracks</td><td>{s2['local']}</td><td>{s3['local']}</td></tr><tr><td>중간 재탐지 신규 Tracks</td><td>{s2['periodic']}</td><td>{s3['periodic']}</td></tr><tr><td>Overlap 병합 후 Global Tracks</td><td>{s2['global']}</td><td>{s3['global']}</td></tr><tr><td>Qwen Names</td><td>{names(s2['names'])}</td><td>{names(s3['names'])}</td></tr></tbody></table><p><a href='../{sam2['root'].name}/report.html'>SAM 2 전체 검수 Report</a> · <a href='../{sam3['root'].name}/report.html'>SAM 3 전체 검수 Report</a></p></section>{video_rows}</body></html>"""
    path = output_dir / "report.html"
    path.write_text(document, encoding="utf-8")
    (output_dir / "comparison.json").write_text(json.dumps({"sam2": s2, "sam3": s3}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2", type=Path, default=Path("outputs/0_te1_sam2"))
    parser.add_argument("--sam3", type=Path, default=Path("outputs/0_te1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison_0_te1"))
    args = parser.parse_args()
    print(write_comparison(load_run(args.sam2), load_run(args.sam3), args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
