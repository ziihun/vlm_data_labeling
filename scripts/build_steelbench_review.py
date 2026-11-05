from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output = (args.output or dataset_dir / "human_review.html").resolve()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(dataset_dir.glob("clip_*.json"))]
    counts = Counter(item["category"] for record in records for item in record.get("items", []))
    total = sum(len(record.get("items", [])) for record in records)
    sections = []
    for record in records:
        clip_id = record["video_id"]
        frames = "".join(
            f"<figure><img src='{esc(Path(frame).as_uri())}'><figcaption>frame_{index}</figcaption></figure>"
            for index, frame in enumerate(record.get("source_frames", []))
        )
        sam_contact = record.get("source_sam_contact")
        if sam_contact and Path(sam_contact).exists():
            frames += (
                f"<figure class='sam-contact'><img src='{esc(Path(sam_contact).as_uri())}'>"
                f"<figcaption>SAM2 all Tracks · BBox contact</figcaption></figure>"
            )
        cards = []
        for item in record.get("items", []):
            cards.append(
                f"<article class='qa' data-video='{esc(clip_id)}' data-item='{esc(item['item_id'])}' data-category='{esc(item['category'])}'>"
                f"<div class='head'><b>{esc(item['item_id'])}</b><span>{esc(item['category'])}</span>"
                f"<select class='decision'><option value='approved'>Approve</option><option value='modified'>Modify</option><option value='rejected'>Reject</option></select></div>"
                f"<label>Question<input class='question' value='{esc(item['question'])}'></label>"
                f"<label>Answer<input class='answer' value='{esc(item['answer'])}'></label>"
                f"<details><summary>Evidence</summary><pre>{esc(json.dumps(item.get('evidence'), ensure_ascii=False, indent=2))}</pre></details>"
                f"<label>Note<textarea class='note'></textarea></label></article>"
            )
        sections.append(f"<section><h2>{esc(clip_id)} · {len(cards)} questions</h2><div class='frames'>{frames}</div><div class='qas'>{''.join(cards)}</div></section>")
    badges = "".join(f"<span>{esc(key)}: {value}</span>" for key, value in sorted(counts.items()))
    document = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>SteelBench VQA Review</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1500px;margin:auto;padding:22px;background:#f3f5f7}}section,.panel{{background:white;padding:18px;border-radius:12px;margin:18px 0;box-shadow:0 2px 12px #0001}}.stats span{{display:inline-block;background:#eaf2f8;padding:5px 9px;border-radius:16px;margin:3px;font-size:12px}}.frames{{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}}figure{{margin:0}}.sam-contact{{grid-column:1/-1}}img{{width:100%}}figcaption{{font-size:11px}}.qas{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:10px;margin-top:14px}}.qa{{border:1px solid #ccd1d1;border-radius:9px;padding:10px}}.head{{display:flex;gap:8px;align-items:center}}.head span{{flex:1;font-size:11px}}label{{display:block;font-size:12px;margin-top:7px}}input,textarea,select{{box-sizing:border-box;width:100%;padding:7px}}.head select{{width:105px}}pre{{white-space:pre-wrap;font-size:11px}}button{{padding:10px 14px;background:#1f618d;color:white;border:0;border-radius:7px}}@media(max-width:700px){{.frames{{grid-template-columns:repeat(2,1fr)}}.qas{{grid-template-columns:1fr}}}}
</style></head><body><div class='panel'><h1>SteelBench VQA Human Review</h1><p>Clips {len(records)} · Questions {total}</p><div class='stats'>{badges}</div></div><div class='panel'><h2>Rejected Questions · <span id='rejectCount'>0</span></h2><div id='rejectList'>None</div></div><button onclick='exportReview()'>Export human_review.json</button>{''.join(sections)}<script>
function state(){{return [...document.querySelectorAll('.qa')].map(x=>({{video_id:x.dataset.video,item_id:x.dataset.item,category:x.dataset.category,decision:x.querySelector('.decision').value,question:x.querySelector('.question').value,answer:x.querySelector('.answer').value,reviewer_notes:x.querySelector('.note').value}}))}}function rejected(){{const r=[...document.querySelectorAll('.qa')].filter(x=>x.querySelector('.decision').value==='rejected');document.getElementById('rejectCount').textContent=r.length;document.getElementById('rejectList').innerHTML=r.length?r.map(x=>`<p><b>${{x.dataset.video}} · ${{x.dataset.item}}</b><br>${{x.querySelector('.question').value}}</p>`).join(''):'None'}}function save(){{localStorage.setItem('steelbench-vqa-review',JSON.stringify(state()));rejected()}}function restore(){{try{{const m=new Map(JSON.parse(localStorage.getItem('steelbench-vqa-review')||'[]').map(x=>[x.video_id+'|'+x.item_id,x]));document.querySelectorAll('.qa').forEach(x=>{{const v=m.get(x.dataset.video+'|'+x.dataset.item);if(v){{x.querySelector('.decision').value=v.decision;x.querySelector('.question').value=v.question;x.querySelector('.answer').value=v.answer;x.querySelector('.note').value=v.reviewer_notes}}}})}}catch(e){{}}rejected()}}function exportReview(){{const b=new Blob([JSON.stringify({{review_status:'completed',items:state()}},null,2)],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='human_review.json';a.click()}}document.querySelectorAll('select,input,textarea').forEach(x=>x.addEventListener('change',save));restore();
</script></body></html>"""
    output.write_text(document, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
