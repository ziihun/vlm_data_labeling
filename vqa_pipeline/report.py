from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


class StageReport:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def rel(self, path: str | Path) -> str:
        return str(Path(path).resolve().relative_to(self.output_root.resolve()))

    def write(
        self,
        manifest: dict[str, Any],
        preprocessing: dict[str, Any] | None = None,
        clip_tracks: list[dict[str, Any]] | None = None,
        pending_annotation: dict[str, Any] | None = None,
        draft_vqa: dict[str, Any] | None = None,
        spatial_relations: list[dict[str, Any]] | None = None,
    ) -> Path:
        segmenter = str(manifest.get("segmentation_model", "sam3"))
        stage01 = f"01_{segmenter}"
        segmenter_label = segmenter.upper()
        naming_model = str(manifest.get("naming_model", "Qwen VLM"))
        openai_model = str(manifest.get("openai_model", "OpenAI API"))
        statuses = "".join(
            f"<tr><td>{key}</td><td><span class='status {value['status']}'>{value['status']}</span></td><td>{html.escape(value.get('detail',''))}</td></tr>"
            for key, value in manifest["stages"].items()
        )
        videos = ""
        if preprocessing and clip_tracks:
            for clip, result in zip(preprocessing["clips"], clip_tracks, strict=False):
                index = int(clip["clip_index"])
                videos += f"""<article><h3>{html.escape(clip['clip_id'])}</h3><p>Source {clip['source_fps']:.3f} FPS / Tracking 5 FPS / Tracks {result.get('track_count', 0)} / Device {result.get('device','unknown')}</p><div class='videos'><video controls src='{html.escape(self.rel(clip['clip_path']))}'></video><video controls src='{stage01}/clip_{index:03d}/tracking_overlay.mp4'></video></div><p><a href='{stage01}/clip_{index:03d}/tracks.json'>tracks.json</a></p></article>"""
        object_cards = ""
        if pending_annotation:
            for obj in pending_annotation["objects"]:
                evidence_images = "".join(
                    f"""<div class='triplet'><img src='{html.escape(self.rel(ev['original_crop_path']))}'><img src='{html.escape(self.rel(ev['masked_crop_path']))}'><img src='{html.escape(self.rel(ev['context_crop_path']))}'><small>Original / Masked / Context · evidence {ev['evidence_index']} · {ev['source_timestamp_seconds']:.3f}s</small></div>"""
                    for ev in obj["evidence"]
                )
                attrs = html.escape(json.dumps(obj["visible_attributes"], ensure_ascii=False))
                gate_status = str(obj.get("verification_status", "pending"))
                approved_selected = " selected" if gate_status == "auto_accepted" else ""
                modified_selected = " selected" if gate_status != "auto_accepted" else ""
                object_cards += f"""<article class='object-card' data-track='{obj['track_id']}'><h3>{obj['track_id']} · {html.escape(obj['object_name'])}</h3><p>BBox {obj['bbox_xyxy']} · Naming confidence {obj['naming_confidence']:.2f} · <span class='status {html.escape(gate_status)}'>{html.escape(gate_status)}</span></p><div class='evidence'>{evidence_images}</div><label>Decision<select class='decision'><option value='approved'{approved_selected}>approved</option><option value='modified'{modified_selected}>modified</option><option value='rejected'>rejected</option></select></label><label>Object name<input class='object-name' value='{html.escape(obj['object_name'])}'></label><label>Attributes JSON<textarea class='attributes'>{attrs}</textarea></label><label>Observable state<input class='state' value='{html.escape(obj['observable_state'])}'></label><label>Reviewer notes<textarea class='notes'></textarea></label><details><summary>Qwen raw response</summary><pre>{html.escape(obj.get('raw_response',''))}</pre></details></article>"""
        vqa_cards = ""
        if draft_vqa:
            pending_by_track = {
                obj["track_id"]: obj for obj in (pending_annotation or {}).get("objects", [])
            }
            for item in draft_vqa.get("items", []):
                evidence_previews = ""
                requested_timestamps = {
                    round(float(value), 6) for value in item["evidence"]["timestamps_seconds"]
                }
                for track_id in item["evidence"]["track_ids"]:
                    obj = pending_by_track.get(track_id)
                    if not obj:
                        continue
                    matches = [
                        ev for ev in obj["evidence"]
                        if round(float(ev["source_timestamp_seconds"]), 6) in requested_timestamps
                    ] or obj["evidence"][:1]
                    for ev in matches:
                        evidence_previews += f"""<div class='triplet'><img src='{html.escape(self.rel(ev['original_crop_path']))}'><img src='{html.escape(self.rel(ev['masked_crop_path']))}'><img src='{html.escape(self.rel(ev['context_crop_path']))}'><small>{html.escape(track_id)} · Original / Masked / Context · evidence metadata {ev['source_timestamp_seconds']:.3f}s</small></div>"""
                vqa_cards += f"""<article class='vqa-card' data-item-id='{html.escape(item['item_id'])}' data-category='{item['category']}'><h3>{html.escape(item['item_id'])} · {item['category']}</h3><label>Decision<select class='vqa-decision'><option value='approved'>approved</option><option value='modified'>modified</option><option value='rejected'>rejected</option></select></label><label>Question<input class='question' value='{html.escape(item['question'])}'></label><label>Answer<input class='answer' value='{html.escape(item['answer'])}'></label><div class='evidence'>{evidence_previews}</div><p>Evidence metadata: {html.escape(str(item['evidence']))}</p><label>Notes<textarea class='vqa-notes'></textarea></label></article>"""
            skipped = "".join(f"<li>{item['category']}: {html.escape(item['reason'])}</li>" for item in draft_vqa.get("skipped_categories", []))
            vqa_cards += f"<h3>Skipped</h3><ul>{skipped}</ul>"
        spatial_rows = "".join(
            f"<tr><td>{html.escape(item['relation_id'])}</td><td>{html.escape(item['subject_track_id'])}</td><td>{html.escape(item['relation'])}</td><td>{html.escape(item['reference_track_id'])}</td><td>{item['confidence']:.2f}</td><td>{html.escape(str(item['evidence_timestamps_seconds']))}</td></tr>"
            for item in (spatial_relations or [])
        )
        video_id = manifest["video_id"]
        document = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>VQA Pipeline Final Verification</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f3f5f7;color:#18202a;max-width:1500px;margin:auto;padding:24px}}section,article{{background:white;border-radius:14px;padding:18px;margin:16px 0;box-shadow:0 2px 14px #0001}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #d8dee6;padding:8px;text-align:left}}.status{{padding:3px 8px;border-radius:99px;background:#ddd}}.completed,.auto_accepted{{background:#bbf7d0}}.failed,.needs_human_review{{background:#fecaca}}.pending_review{{background:#fde68a}}.skipped{{background:#e5e7eb}}.videos{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}video{{width:100%;background:#111}}.evidence{{display:grid;gap:10px}}.triplet{{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;background:#eef2f7;padding:6px}}.triplet img{{width:100%;height:180px;object-fit:contain;background:#111}}.triplet small{{grid-column:1/-1}}label{{display:block;margin:10px 0;font-weight:600}}input,select,textarea{{display:block;width:100%;box-sizing:border-box;padding:8px;margin-top:4px}}textarea{{min-height:60px}}button{{padding:10px 16px;font-weight:700;cursor:pointer}}pre{{white-space:pre-wrap;word-break:break-word}}</style></head><body>
<section><h1>VQA Dataset Pipeline · {html.escape(video_id)} · {segmenter_label}</h1><p><b>{segmenter_label} → Margin Crop → {html.escape(naming_model)} → Structured Annotation Gate → {html.escape(openai_model)} VQA → Human Verification</b></p><p>GPU 모델은 직렬 실행 후 종료한다. VQA는 Naming 프로세스가 끝난 뒤 저장된 JSON만 입력받는다.</p></section>
<section><h2>Stage Status</h2><table><thead><tr><th>Stage</th><th>Status</th><th>Detail</th></tr></thead><tbody>{statuses}</tbody></table><p><a href='manifest.json'>manifest.json</a></p></section>
<section><h2>00 · Video Preprocessing</h2><p><a href='00_preprocessing/manifest.json'>Input / FPS / Clip manifest</a></p></section>
<section><h2>01 · {segmenter_label} Discovery & Tracking</h2><p>오른쪽 영상에서 인식한 모든 Track의 Mask, BBox, Track ID를 확인한다.</p>{videos}</section>
<section><h2>02 · 20% Margin Crops</h2><p>각 묶음은 Original / Masked / Context 순서다. <a href='02_crops/crops.json'>crops.json</a></p></section>
<section><h2>03 · Qwen Object Naming</h2><p><a href='03_qwen_naming/prompt.txt'>실제 Prompt</a> · <a href='03_qwen_naming/raw_response.json'>Raw Response</a> · <a href='03_qwen_naming/naming.json'>Parsed Naming</a> · <a href='05_human_review/naming_gate.json'>Confidence Gate</a></p></section>
<section><h2>04–05 · Structured Annotation / Human Verification</h2><p>모든 객체를 승인·수정·거절한 뒤 review.json을 내려받아 <code>05_human_review/review.json</code>에 넣고 파이프라인을 다시 실행한다.</p>{object_cards}<label>Verified events JSON<textarea id='events'>[]</textarea></label><label>Verified negative facts JSON<textarea id='negativeFacts'>[]</textarea></label><button onclick='downloadObjectReview()'>review.json 다운로드</button><p><a href='04_annotation/objects.json'>Pending objects.json</a></p></section>
<section><h2>04B · Stable Spatial Relations</h2><p>같은 Timestamp에서 함께 보이는 BBox의 방향이 여러 프레임에 걸쳐 일관될 때만 생성한다. 최종 Benchmark 전에는 영상으로 검수해야 한다. <a href='04_annotation/spatial_relations.json'>spatial_relations.json</a></p><table><thead><tr><th>ID</th><th>Subject</th><th>Relation</th><th>Reference</th><th>Confidence</th><th>Evidence timestamps</th></tr></thead><tbody>{spatial_rows}</tbody></table></section>
<section><h2>06–07 · OpenAI VQA / Final Review</h2><p>API에는 Gate를 통과한 Structured Annotation JSON만 전달한다. 생성 문항은 draft이며 마지막 사람 검수 전에는 Benchmark에 포함되지 않는다.</p><p><a href='06_vqa/api_input_annotation.json'>API 입력 Annotation</a> · <a href='06_vqa/prompt.txt'>실제 VQA Prompt</a> · <a href='06_vqa/raw_response.json'>OpenAI Raw Response</a> · <a href='06_vqa/generated_vqa.json'>Parsed VQA</a></p>{vqa_cards}<button {'style="display:none"' if not draft_vqa else ''} onclick='downloadVQAReview()'>final_review.json 다운로드</button></section>
<script>
const videoId={json.dumps(video_id)};
function download(name,data){{const blob=new Blob([JSON.stringify(data,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();URL.revokeObjectURL(a.href);}}
function downloadObjectReview(){{const reviews=[...document.querySelectorAll('.object-card')].map(card=>({{track_id:card.dataset.track,decision:card.querySelector('.decision').value,object_name:card.querySelector('.object-name').value,visible_attributes:JSON.parse(card.querySelector('.attributes').value),observable_state:card.querySelector('.state').value,reviewer_notes:card.querySelector('.notes').value}}));download('review.json',{{video_id:videoId,review_status:'completed',object_reviews:reviews,verified_events:JSON.parse(document.getElementById('events').value),verified_negative_facts:JSON.parse(document.getElementById('negativeFacts').value)}});}}
function downloadVQAReview(){{const items=[...document.querySelectorAll('.vqa-card')].map(card=>({{item_id:card.dataset.itemId,category:card.dataset.category,decision:card.querySelector('.vqa-decision').value,question:card.querySelector('.question').value,answer:card.querySelector('.answer').value,reviewer_notes:card.querySelector('.vqa-notes').value}}));download('final_review.json',{{video_id:videoId,review_status:'completed',items}});}}
</script></body></html>"""
        path = self.output_root / "report.html"
        path.write_text(document, encoding="utf-8")
        return path
