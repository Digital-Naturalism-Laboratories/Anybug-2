"""
review_missed.py

Tiny local web app for reviewing the unmatched detections found by
flatbug_find_missed.py. Decisions are saved to <dir>/decisions.json after
every page, so you can stop and resume any time — and you can start
reviewing while the detection run is still going (reload to pick up more).

Usage:
    python3 review_missed.py --dir /Users/automeris/Desktop/Megabug/flatbug_missed
    → open http://127.0.0.1:8765

Controls:
    click          creature (green) — click again to clear
    right-click    not a creature / don't care (red) — right-click again to clear
    shift-click    big context view (also: hover + Z)
    Enter / Space  save marks — marked tiles leave the unreviewed view,
                   unmarked tiles stay unreviewed
    A / N          mark every unmarked tile on the page creature / not
    U              undo the last save
    ← / →          previous / next page (marks are kept until you save)
"""

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ARGS = None


def load_ugliness():
    """{id: {"ugly", "ugly_winner", "ugly_type"}} from score_ugliness.py, if it has run."""
    scores = {}
    path = ARGS.dir / "ugliness.jsonl"
    if path.exists():
        for line in open(path):
            try:
                scores.update(json.loads(line)["scores"])
            except json.JSONDecodeError:
                continue  # last line mid-write
    return scores


def load_candidates():
    out = []
    path = ARGS.dir / "detections.jsonl"
    if not path.exists():
        return out
    ugly = load_ugliness()
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # last line mid-write
        for d in r["dets"]:
            if not d["matched"]:
                out.append({"id": d["id"], "code": r["code"], "subset": r["subset"],
                            "file": r["file"], "conf": d["conf"], "long": d["long"],
                            "short": d["short"], "caption": d.get("caption"), **ugly.get(d["id"], {})})
    return out


def load_decisions():
    p = ARGS.dir / "decisions.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_decisions(dec):
    p = ARGS.dir / "decisions.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dec, indent=0))
    os.replace(tmp, p)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            page = (PAGE.replace("__COLS__", str(ARGS.cols)).replace("__MAXROWS__", str(ARGS.max_rows))
                    .replace("__MAXSIZE__", str(ARGS.max_size)).replace("__TITLE__", ARGS.title).replace("__HELP__", ARGS.legend or DEFAULT_HELP))
            self.send(200, page.encode(), "text/html; charset=utf-8")
        elif path == "/api/data":
            body = json.dumps({"candidates": load_candidates(), "decisions": load_decisions()})
            self.send(200, body.encode(), "application/json")
        elif path.startswith("/crops/"):
            f = ARGS.dir / "crops" / Path(unquote(path[7:])).name
            if f.exists():
                self.send(200, f.read_bytes(), "image/jpeg")
            else:
                self.send(404, b"", "text/plain")
        else:
            self.send(404, b"", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/api/decide":
            return self.send(404, b"", "text/plain")
        changes = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        dec = load_decisions()
        now = int(time.time())
        for cid, label in changes.items():
            if label is None:
                dec.pop(cid, None)
            else:
                dec[cid] = {"label": label, "t": now}
        save_decisions(dec)
        self.send(200, b"{}", "application/json")


DEFAULT_HELP = """Judge the <b style="color:#ff28c8">pink box</b> only — <b style="color:#00e66e">green outlines</b> are flat-bug's existing labels, shown for context.
 click = creature (green) · right-click = not (red) · click again to clear · unmarked = not reviewed yet · shift-click or hover+Z = wider view · Enter = save marks · A / N = mark rest creature / not · U = undo · ←/→ pages"""

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
body{margin:0;background:#1b1b1b;color:#ddd;font:13px -apple-system,Helvetica,sans-serif}
header{position:sticky;top:0;background:#111;padding:8px 12px;display:flex;flex-wrap:wrap;gap:10px;align-items:center;z-index:5;border-bottom:1px solid #333}
header label{display:flex;gap:4px;align-items:center}
input,select,button{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:4px;padding:3px 6px;font:inherit}
input[type=number]{width:60px}
button{cursor:pointer}button.primary{background:#2d6a3e;border-color:#3c8a52}
#stats{margin-left:auto;color:#aaa}
#grid{display:grid;grid-template-columns:repeat(__COLS__,var(--tile,200px));gap:8px;padding:8px;justify-content:center}
.t{position:relative;border:4px solid #444;border-radius:5px;cursor:pointer;background:#000}
.t img{width:100%;aspect-ratio:1;display:block}
.t .c{font-size:12px;padding:3px 5px;color:#bbb;white-space:nowrap;overflow:hidden}
.t .b{position:absolute;top:6px;left:6px;font:bold 15px sans-serif;padding:2px 8px;border-radius:4px;display:none}
.t.insect{border-color:#2ecc71}.t.insect .b{display:block;background:#2ecc71;color:#000}
.t.not{border-color:#e74c3c}.t.not img{opacity:.45}.t.not .b{display:block;background:#e74c3c;color:#fff}
.t.saved::after{content:"saved";position:absolute;top:6px;right:6px;font-size:11px;color:#fff;background:#000a;padding:1px 4px}
#modal{display:none;position:fixed;inset:0;background:#000c;z-index:10;align-items:center;justify-content:center;flex-direction:column}
#modal img{max-width:90vw;max-height:85vh}#modal div{margin-top:6px}
.help{color:#888;font-size:12px;padding:0 12px 6px}
</style></head><body>
<header>
 <label>subset <select id="subset"><option value="">all</option></select></label>
 <label>size px <input id="minL" type="number" value="0"> – <input id="maxL" type="number" value="__MAXSIZE__"></label>
 <label>min conf <input id="minC" type="number" step="0.05" value="0.15"></label>
 <label>show <select id="show"><option value="none">unreviewed</option><option value="insect">creature</option><option value="not">not</option><option value="all">all</option></select></label>
 <label>sort <select id="sort"><option value="conf">confidence ↓</option><option value="size">size ↑</option><option value="ugly">ugly ↑ (sharpest first)</option><option value="uglyd">ugly ↓ (ugliest first)</option><option value="image">by image</option></select></label>
 <button id="reload">reload data</button>
 <button class="primary" id="commit">save marks ⏎</button>
 <button id="undo">undo (U)</button>
 <span id="stats"></span>
</header>
<div class="help">__HELP__</div>
<div id="grid"></div>
<div id="modal"><img><div></div></div>
<script>
let PAGE=15,C=[],D={},view=[],page=0,pending={},history=[],hover=null;
const $=s=>document.querySelector(s);
async function load(){const r=await (await fetch('/api/data')).json();C=r.candidates;D=r.decisions;
 const subs=[...new Set(C.map(c=>c.subset))].sort(),sel=$('#subset'),cur=sel.value;
 sel.innerHTML='<option value="">all</option>'+subs.map(s=>`<option>${s}</option>`).join('');sel.value=cur;refilter();}
// pending holds unsaved marks (null = clear a saved mark); kept across pages until saved.
const saved=id=>D[id]?.label??null;
function label(id){return id in pending?pending[id]:saved(id)}
function refilter(keepPage){const sub=$('#subset').value,a=+$('#minL').value,b=+$('#maxL').value||1e9,mc=+$('#minC').value,show=$('#show').value;
 view=C.filter(c=>(!sub||c.subset==sub)&&c.long>=a&&c.long<=b&&c.conf>=mc&&
  (show=='all'||(show=='none'?!D[c.id]:saved(c.id)==show)));
 const s=$('#sort').value;
 const U=x=>x.ugly??1e9;  // unscored last
 view.sort(s=='conf'?(x,y)=>y.conf-x.conf:s=='size'?(x,y)=>x.long-y.long:s=='ugly'?(x,y)=>U(x)-U(y):s=='uglyd'?(x,y)=>(y.ugly??-1)-(x.ugly??-1):(x,y)=>(x.subset+x.file).localeCompare(y.subset+y.file)||y.conf-x.conf);
 if(!keepPage)page=0;page=Math.min(page,Math.max(0,Math.ceil(view.length/PAGE)-1));render();}
function pageItems(){return view.slice(page*PAGE,(page+1)*PAGE)}
function render(){const g=$('#grid');g.innerHTML='';
 for(const c of pageItems()){const t=document.createElement('div');t.className='t';t.dataset.id=c.id;
  t.innerHTML=`<img loading="lazy" src="/crops/${c.id}_s.jpg"><span class="b"></span><div class="c">${c.caption?c.caption:c.conf.toFixed(2)} · ${Math.round(c.long)}px${c.ugly!=null?` · ugly ${Math.round(c.ugly)}${c.ugly_winner=='motion'?'m':''}`:''} · ${c.code} · ${c.file}</div>`;
  t.onclick=e=>{if(e.shiftKey)return zoom(c);toggle(c.id,'insect');paint(t)};
  t.oncontextmenu=e=>{e.preventDefault();toggle(c.id,'not');paint(t)};
  t.onmouseenter=()=>hover=c;t.onmouseleave=()=>hover=null;paint(t);g.appendChild(t);}
 for(const c of view.slice((page+1)*PAGE,(page+2)*PAGE))new Image().src=`/crops/${c.id}_s.jpg`;
 stats();window.scrollTo(0,0);}
function setMark(id,l){if(l===saved(id))delete pending[id];else pending[id]=l}
function toggle(id,want){setMark(id,label(id)==want?null:want);stats()}
const BADGE={insect:'✓ creature',not:'✗ not'};
function paint(t){const id=t.dataset.id,l=label(id);t.className='t'+(l?' '+l:'')+(D[id]&&!(id in pending)?' saved':'');
 t.querySelector('.b').textContent=BADGE[l]||''}
// --cols columns, up to --max-rows rows, sized to fill the window.
function layout(){const top=$('header').offsetHeight+$('.help').offsetHeight,cap=26,gap=8;
 const availW=innerWidth-16-(__COLS__-1)*gap,availH=innerHeight-top-16;
 let w=Math.floor(availW/__COLS__),rows=Math.floor((availH+gap)/(w+cap+gap));
 if(rows<2){rows=2;w=Math.floor((availH-gap)/2-cap)}
 rows=Math.min(rows,__MAXROWS__);document.body.style.setProperty('--tile',Math.max(w,120)+'px');
 const first=page*PAGE;PAGE=__COLS__*rows;page=Math.floor(first/PAGE);}
let rt;addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(()=>{layout();render()},150)});
addEventListener('beforeunload',e=>{if(Object.keys(pending).length)e.preventDefault()});
function stats(){const n=C.length,cnt={insect:0,not:0};for(const k in D)cnt[D[k].label]=(cnt[D[k].label]||0)+1;
 const u=Object.keys(pending).length;
 $('#stats').textContent=`page ${page+1}/${Math.max(1,Math.ceil(view.length/PAGE))} · ${view.length} in view · reviewed ${Object.keys(D).length}/${n} · creature ${cnt.insect} · not ${cnt.not}`+(u?` · ${u} unsaved`:'')}
async function post(ch){await fetch('/api/decide',{method:'POST',body:JSON.stringify(ch)});for(const k in ch){if(ch[k]===null)delete D[k];else D[k]={label:ch[k]}}}
async function commit(){const ch={...pending};if(!Object.keys(ch).length)return;const prev={};
 for(const id in ch)prev[id]=saved(id);
 await post(ch);history.push(prev);pending={};refilter(true)}
async function undo(){const prev=history.pop();if(!prev)return;await post(prev);refilter(true)}
function markRest(l){for(const c of pageItems())if(!label(c.id))setMark(c.id,l);render()}
function zoom(c){const m=$('#modal');m.querySelector('img').src=`/crops/${c.id}_b.jpg`;
 m.querySelector('div').textContent=`${c.subset}/${c.file} · conf ${c.conf} · ${Math.round(c.long)}×${Math.round(c.short)}px`+(c.ugly!=null?` · ugly ${c.ugly} (${c.ugly_winner})`:'');m.style.display='flex'}
$('#modal').onclick=()=>$('#modal').style.display='none';
for(const id of ['subset','minL','maxL','minC','show','sort'])$('#'+id).onchange=()=>refilter();
$('#reload').onclick=load;$('#commit').onclick=commit;$('#undo').onclick=undo;
document.onkeydown=e=>{if(e.target.tagName=='INPUT'||e.target.tagName=='SELECT')return;
 if(e.key=='Escape')$('#modal').style.display='none';
 else if(e.key=='Enter'||e.key==' '){e.preventDefault();commit()}
 else if(e.key=='a')markRest('insect');
 else if(e.key=='n')markRest('not');
 else if(e.key=='u')undo();
 else if(e.key=='z'&&hover)zoom(hover);
 else if(e.key=='ArrowRight'){page=Math.min(page+1,Math.max(0,Math.ceil(view.length/PAGE)-1));render()}
 else if(e.key=='ArrowLeft'){page=Math.max(0,page-1);render()}};
layout();load();
</script></body></html>"""


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path, help="output folder of flatbug_find_missed.py")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--cols", type=int, default=5, help="tiles per row")
    ap.add_argument("--max-rows", type=int, default=3, help="most rows per page (fewer if the window is short)")
    ap.add_argument("--max-size", type=float, default=48, help="initial upper size filter, px")
    ap.add_argument("--title", default="Flatbug missed-creature review")
    ap.add_argument("--legend", help="HTML for the help line under the toolbar (default: missed-creature review)")
    ARGS = ap.parse_args()
    print(f"Reviewing {ARGS.dir}\n→ open http://127.0.0.1:{ARGS.port}   (Ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
