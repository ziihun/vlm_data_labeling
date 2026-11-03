from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


class ArtifactWriter:
    """Writes machine-readable artifacts and a video-first human review page."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)

    def write_json(self, relative_path: str | Path, payload: Any) -> None:
        self._write_json(self.output_root / relative_path, payload)

    def write_annotation(self, video_id: str, wrapper: dict[str, Any]) -> Path:
        relative = Path("annotations") / f"{video_id}.json"
        self.write_json(relative, wrapper)
        return relative

    @staticmethod
    def flatten_vqa(wrapper: dict[str, Any]) -> list[dict[str, Any]]:
        source = wrapper["source"]
        return [
            {
                "video_id": source["video_id"],
                "video_path": source["video_path"],
                "review_video_path": source["review_video_path"],
                "dataset_split": source["dataset_split"],
                "dataset_label": source["dataset_label"],
                **item,
                "annotation_path": wrapper["annotation_path"],
            }
            for item in wrapper["result"]["vqa"]
        ]

    def write_jsonl(self, records: Iterable[dict[str, Any]]) -> None:
        path = self.output_root / "vqa.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_review_csv(self, records: list[dict[str, Any]]) -> None:
        columns = [
            "video_id",
            "review_video_path",
            "dataset_label",
            "category",
            "question",
            "answer",
            "polarity",
            "timestamp_start",
            "timestamp_end",
            "evidence_frame_indices",
            "evidence_timestamps_seconds",
            "evidence_rationale",
            "confidence",
            "requires_human_review",
            "review_status",
            "reviewer_notes",
        ]
        with (self.output_root / "human_review.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for record in records:
                evidence = record["evidence"]
                row = {
                    **record,
                    "evidence_frame_indices": json.dumps(evidence["frame_indices"]),
                    "evidence_timestamps_seconds": json.dumps(
                        evidence["timestamps_seconds"]
                    ),
                    "evidence_rationale": evidence["rationale"],
                    "review_status": "pending",
                    "reviewer_notes": "",
                }
                writer.writerow({column: row.get(column, "") for column in columns})

    def write_review_html(self, wrappers: list[dict[str, Any]]) -> None:
        sections: list[str] = []
        for wrapper in wrappers:
            source = wrapper["source"]
            video_id = html.escape(source["video_id"])
            video_path = html.escape(source["review_video_path"])
            label = html.escape(str(source["dataset_label"]))
            summary = html.escape(wrapper["result"]["scene_summary"])
            rows: list[str] = []
            for item in wrapper["result"]["vqa"]:
                timestamp_buttons = " ".join(
                    (
                        f'<button type="button" onclick="seekVideo(\'{video_id}\', '
                        f'{float(timestamp):.3f})">{float(timestamp):.2f}s</button>'
                    )
                    for timestamp in item["evidence"]["timestamps_seconds"]
                )
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(item['category'])}</td>"
                    f"<td>{html.escape(item['question'])}</td>"
                    f"<td>{html.escape(item['answer'])}</td>"
                    f"<td>{html.escape(item['polarity'])}</td>"
                    f"<td>{timestamp_buttons}<br>{html.escape(item['evidence']['rationale'])}</td>"
                    f"<td>{float(item['confidence']):.2f}</td>"
                    "</tr>"
                )
            sections.append(
                f"""
                <section>
                  <h2>{video_id} <span>{label}</span></h2>
                  <video id="{video_id}" controls preload="metadata" src="{video_path}"></video>
                  <p><strong>장면 요약:</strong> {summary}</p>
                  <table>
                    <thead><tr><th>Category</th><th>Question</th><th>Answer</th><th>Polarity</th><th>Evidence</th><th>Confidence</th></tr></thead>
                    <tbody>{''.join(rows)}</tbody>
                  </table>
                </section>
                """
            )

        document = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VQA Pilot Video Review</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0 auto; max-width: 1440px; padding: 24px; color: #17202a; background: #f4f6f7; }}
    section {{ background: white; border-radius: 12px; padding: 20px; margin: 0 0 28px; box-shadow: 0 2px 12px #0001; }}
    h2 span {{ color: #566573; font-size: 0.65em; font-weight: 500; }}
    video {{ display: block; width: min(100%, 960px); max-height: 70vh; background: #111; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border: 1px solid #d5dbdb; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eaf2f8; }}
    button {{ margin: 0 4px 4px 0; cursor: pointer; }}
  </style>
</head>
<body>
  <h1>5-video VQA Pilot Review</h1>
  <p>버튼을 누르면 해당 증거 시점으로 영상이 이동한다. Risk Reasoning은 포함하지 않았다.</p>
  {''.join(sections)}
  <script>
    function seekVideo(id, seconds) {{
      const video = document.getElementById(id);
      video.currentTime = seconds;
      video.play();
      video.scrollIntoView({{behavior: 'smooth', block: 'center'}});
    }}
  </script>
</body>
</html>
"""
        (self.output_root / "review.html").write_text(document, encoding="utf-8")

    def finalize(
        self,
        wrappers: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        records = [record for wrapper in wrappers for record in self.flatten_vqa(wrapper)]
        self.write_jsonl(records)
        self.write_review_csv(records)
        self.write_review_html(wrappers)
        category_counts: dict[str, int] = defaultdict(int)
        for record in records:
            category_counts[record["category"]] += 1
        return records, dict(category_counts)
