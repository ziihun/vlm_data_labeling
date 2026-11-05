from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter
from pathlib import Path


CATEGORY_LABELS = {
    "scene.object_attribute_recognition": "Scene · Object & Attribute",
    "scene.action_spatial_understanding": "Scene · Action & Spatial",
    "scene.temporal_understanding": "Scene · Temporal",
    "scene.task_state_completion_understanding": "Scene · Task State & Completion",
    "alignment.object_attribute_alignment": "Alignment · Object & Attribute",
    "alignment.action_relation_alignment": "Alignment · Action & Relation",
    "alignment.temporal_alignment": "Alignment · Temporal",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def load_gt(project_root: Path) -> dict[str, dict]:
    path = project_root / "raw_datasets/Safe_and_Unsafe_Behaviours/samples.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {Path(item["filepath"]).stem: item for item in payload.get("samples", [])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video-dir", type=Path)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    project_root = Path(__file__).resolve().parents[1]
    output = (args.output or dataset_dir / "human_review.html").resolve()
    video_dir = args.video_dir.resolve() if args.video_dir else None
    gt_by_id = load_gt(project_root)
    records = []
    for path in sorted(dataset_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("items", [])
        usage = payload.get("usage") or {}
        temporal_usage = (payload.get("temporal_generation") or {}).get("usage") or {}
        records.append({
            "path": path,
            "payload": payload,
            "items": items,
            "counts": Counter(item["category"] for item in items),
            "size_bytes": path.stat().st_size,
            "tokens": int(usage.get("total_tokens") or 0) + int(temporal_usage.get("total_tokens") or 0),
            "gt": gt_by_id.get(payload.get("video_id", ""), {}),
        })

    total_items = sum(len(record["items"]) for record in records)
    total_bytes = sum(record["size_bytes"] for record in records)
    total_tokens = sum(record["tokens"] for record in records)
    overall = Counter(item["category"] for record in records for item in record["items"])

    summary_rows = []
    sections = []
    for record in records:
        payload = record["payload"]
        video_id = payload["video_id"]
        gt = record["gt"]
        metadata = gt.get("metadata") or {}
        gt_label = (gt.get("ground_truth") or {}).get("label", "—")
        split = ", ".join(gt.get("tags") or []) or "—"
        source = Path(payload.get("source_video", ""))
        if video_dir is not None:
            video_href = Path(os.path.relpath(video_dir / f"{video_id}.mp4", output.parent))
        else:
            try:
                video_href = Path("../../") / source.relative_to(project_root)
            except (ValueError, TypeError):
                video_href = source
        category_cells = "".join(
            f"<td>{record['counts'].get(category, 0)}</td>" for category in CATEGORY_LABELS
        )
        summary_rows.append(
            f"<tr><td><a href='#{esc(video_id)}'>{esc(video_id)}</a></td>"
            f"<td>{esc(gt_label)}</td><td>{len(record['items'])}</td>{category_cells}"
            f"<td>{record['tokens']:,}</td><td>{record['size_bytes'] / 1024:.1f} KB</td></tr>"
        )

        cards = []
        for item in record["items"]:
            evidence = item.get("evidence") or {}
            cards.append(
                f"<article class='qa' data-video='{esc(video_id)}' data-item='{esc(item['item_id'])}' data-category='{esc(item['category'])}'>"
                f"<div class='qa-head'><b>{esc(item['item_id'])}</b>"
                f"<span>{esc(CATEGORY_LABELS.get(item['category'], item['category']))}</span>"
                f"<select class='decision'><option value='approved'>Approve</option>"
                f"<option value='modified'>Modify</option><option value='rejected'>Reject</option></select></div>"
                f"<label>Question<input class='question' value='{esc(item['question'])}'></label>"
                f"<label>Answer<input class='answer' value='{esc(item['answer'])}'></label>"
                f"<div class='meta'>type={esc(item.get('answer_type'))} · polarity={esc(item.get('polarity'))} "
                f"· confidence={esc(item.get('confidence'))}</div>"
                f"<details><summary>Evidence</summary><pre>{esc(json.dumps(evidence, ensure_ascii=False, indent=2))}</pre></details>"
                f"<label>Review note<textarea class='note'></textarea></label></article>"
            )

        skipped = "".join(
            f"<li><b>{esc(CATEGORY_LABELS.get(item['category'], item['category']))}</b>: {esc(item['reason'])}</li>"
            for item in payload.get("skipped_categories", [])
        ) or "<li>None</li>"
        count_badges = "".join(
            f"<span class='badge'>{esc(CATEGORY_LABELS[category])}: {record['counts'].get(category, 0)}</span>"
            for category in CATEGORY_LABELS
        )
        sections.append(
            f"<section id='{esc(video_id)}' class='video-section'>"
            f"<h2>{esc(video_id)} <small>{esc(gt_label)}</small></h2>"
            f"<div class='video-grid'><video controls preload='metadata' src='{esc(video_href.as_posix())}'></video>"
            f"<div><p><b>Source:</b> {esc(source.name)}</p><p><b>Split:</b> {esc(split)}</p>"
            f"<p><b>Video:</b> {metadata.get('duration', 0):.2f}s · {metadata.get('frame_rate', 0):.2f} FPS "
            f"· {metadata.get('frame_width', '—')}×{metadata.get('frame_height', '—')}</p>"
            f"<p><b>VQA:</b> {len(record['items'])} · <b>Tokens:</b> {record['tokens']:,} "
            f"· <b>JSON:</b> {record['size_bytes'] / 1024:.1f} KB</p>{count_badges}"
            f"<details><summary>Skipped categories</summary><ul>{skipped}</ul></details></div></div>"
            f"<div class='qa-list'>{''.join(cards)}</div></section>"
        )

    headers = "".join(f"<th>{esc(label)}</th>" for label in CATEGORY_LABELS.values())
    overall_badges = "".join(
        f"<span class='badge'>{esc(label)}: {overall.get(category, 0)}</span>"
        for category, label in CATEGORY_LABELS.items()
    )
    document = f"""<!doctype html>
<html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>VQA Human Review</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1500px;margin:auto;padding:24px;background:#f4f6f8;color:#17202a}}
h1,h2{{margin-bottom:8px}} small{{font-weight:400;color:#566573}} .panel,.video-section{{background:#fff;border-radius:14px;padding:20px;margin-bottom:22px;box-shadow:0 2px 12px #0001}}
.totals{{display:flex;gap:20px;flex-wrap:wrap;font-size:18px}} .badge{{display:inline-block;background:#eaf2f8;border-radius:20px;padding:5px 10px;margin:3px;font-size:12px}}
table{{border-collapse:collapse;width:100%;font-size:12px;overflow:auto;display:block}} th,td{{border:1px solid #d5d8dc;padding:7px;white-space:nowrap}} th{{background:#eef2f3}}
.video-grid{{display:grid;grid-template-columns:minmax(420px,1.3fr) 1fr;gap:18px}} video{{width:100%;max-height:520px;background:#111}}
.qa-list{{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:12px;margin-top:18px}} .qa{{border:1px solid #d5d8dc;border-radius:10px;padding:12px;background:#fbfcfc}}
.qa-head{{display:flex;gap:10px;align-items:center;margin-bottom:10px}} .qa-head span{{flex:1;color:#566573;font-size:12px}} label{{display:block;font-size:12px;margin:8px 0}}
input,textarea,select{{box-sizing:border-box;width:100%;padding:8px;border:1px solid #aeb6bf;border-radius:6px;background:#fff}} .qa-head select{{width:110px}} textarea{{min-height:48px}}
.meta{{font-size:11px;color:#5d6d7e}} pre{{white-space:pre-wrap;word-break:break-word;font-size:11px;background:#f2f3f4;padding:8px}} .sticky{{position:sticky;top:8px;z-index:3}}
button{{background:#1f618d;color:#fff;border:0;border-radius:8px;padding:11px 16px;cursor:pointer}} @media(max-width:800px){{.video-grid{{grid-template-columns:1fr}}.qa-list{{grid-template-columns:1fr}}}}
</style></head><body>
<section class='panel'><h1>VQA Human Review</h1><div class='totals'><b>Videos {len(records)}</b><b>Questions {total_items}</b><b>Tokens {total_tokens:,}</b><b>JSON {total_bytes / 1024:.1f} KB</b></div>{overall_badges}</section>
<section class='panel'><h2>Rejected Questions · <span id='rejectCount'>0</span></h2><div id='rejectList'>None</div></section>
<section class='panel'><h2>Video / Dataset Volume</h2><table><thead><tr><th>Video</th><th>Original GT</th><th>Total</th>{headers}<th>Tokens</th><th>JSON</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></section>
<div class='sticky'><button onclick='exportReview()'>Export human_review.json</button></div>
{''.join(sections)}
<script>
const stateKey='vqa-human-review-state-v2';
function reviewState(){{return [...document.querySelectorAll('.qa')].map(x=>({{video_id:x.dataset.video,item_id:x.dataset.item,decision:x.querySelector('.decision').value,question:x.querySelector('.question').value,answer:x.querySelector('.answer').value,reviewer_notes:x.querySelector('.note').value}}));}}
function updateRejected(){{const rows=[...document.querySelectorAll('.qa')].filter(x=>x.querySelector('.decision').value==='rejected').map(x=>`<div><b>${{x.dataset.video}} · ${{x.dataset.item}}</b> · ${{x.dataset.category}}<br>${{x.querySelector('.question').value}}</div>`);document.getElementById('rejectCount').textContent=rows.length;document.getElementById('rejectList').innerHTML=rows.length?rows.join('<hr>'):'None';}}
function saveState(){{localStorage.setItem(stateKey,JSON.stringify(reviewState()));updateRejected();}}
function restoreState(){{try{{const saved=JSON.parse(localStorage.getItem(stateKey)||'[]');const map=new Map(saved.map(x=>[x.video_id+'|'+x.item_id,x]));document.querySelectorAll('.qa').forEach(x=>{{const v=map.get(x.dataset.video+'|'+x.dataset.item);if(!v)return;x.querySelector('.decision').value=v.decision;x.querySelector('.question').value=v.question;x.querySelector('.answer').value=v.answer;x.querySelector('.note').value=v.reviewer_notes;}});}}catch(e){{}}updateRejected();}}
function exportReview(){{
 const items=reviewState();
 const blob=new Blob([JSON.stringify({{review_status:'completed',items}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='human_review.json';a.click();URL.revokeObjectURL(a.href);
}}
document.querySelectorAll('.decision,.question,.answer,.note').forEach(x=>x.addEventListener('change',saveState));restoreState();
</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
