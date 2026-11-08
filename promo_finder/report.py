"""Write the outputs folder: promos.json, promos.csv, report.md, report.html, run_meta.json.

Design choices (short version):
- promos.json is the complete, machine-readable result (every score, reason
  and flag). The CSV is the same list flattened for a spreadsheet.
- report.md is for a human: top 15 with hook / cliffhanger / why, coverage by
  episode, and an automatic "where it picks badly" section built from the run
  itself (rule-vs-LLM disagreements, LLM 'skip' verdicts, weak episodes, flags).
- report.html is one self-contained file (no internet needed): a filterable,
  sortable table of every promo.
- Timestamps are always written as MM:SS.ss, the same style as the scripts.
"""

from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path

from .candidates import Candidate
from .parser import fmt_time
from .pipeline import LOW_SCORE, RunResult

DISAGREE = 0.20   # |rule - llm| at least this much = worth a human look


def segments_text(c: Candidate) -> str:
    return " + ".join(f"Ep{s.episode} {fmt_time(s.start)}-{fmt_time(s.end)}" for s in c.segments)


def promo_record(c: Candidate) -> dict:
    d = c.to_dict()
    d["segments_text"] = segments_text(c)
    d["verdict"] = (c.llm or {}).get("verdict")
    return d


# ---------------------------------------------------------------- json / csv


def write_json(result: RunResult, meta: dict, path: Path) -> None:
    data = {"meta": meta, "promos": [promo_record(c) for c in result.promos]}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


CSV_COLUMNS = ["rank", "id", "shape", "duration", "segments", "hook", "cliffhanger", "why", "final", "rule",
               "llm", "hook_score", "cliffhanger_score", "standalone_score", "intensity_score", "llm_verdict",
               "llm_adjusted", "flags"]


def write_csv(result: RunResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # -sig so Excel shows accents/dashes right
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for c in result.promos:
            s = c.scores
            w.writerow([c.rank, c.id, c.shape, c.duration, segments_text(c), c.hook_text, c.cliffhanger_text,
                        c.why, s.get("final"), s.get("rule"), s.get("llm"), s.get("hook"), s.get("cliffhanger"),
                        s.get("standalone"), s.get("intensity"), (c.llm or {}).get("verdict", ""),
                        c.llm_adjusted, "; ".join(c.flags)])


# ---------------------------------------------------------------- markdown


def _md(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ")


def picks_badly(result: RunResult) -> list[str]:
    """Automatic quality notes from this run."""
    out: list[str] = []
    judged = [c for c in result.promos if c.scores.get("llm") is not None]
    dis = sorted((c for c in judged if abs(c.scores["rule"] - c.scores["llm"]) >= DISAGREE),
                 key=lambda c: -abs(c.scores["rule"] - c.scores["llm"]))
    if judged:
        out.append(f"**Rules vs LLM disagree (≥{DISAGREE:.2f}) on {len(dis)} of {len(judged)} judged promos.** "
                   "These are where the keyword rules are most likely wrong:")
        for c in dis[:8]:
            out.append(f"- `{c.id}` (rank {c.rank}) rule {c.scores['rule']:.2f} vs LLM {c.scores['llm']:.2f}: "
                       f"{_md(c.llm.get('why', ''))}")
        skips = [c for c in judged if c.llm.get("verdict") in ("skip", "maybe")]
        out.append(f"- LLM verdicts among judged promos: "
                   + ", ".join(f"{v} {n}" for v, n in Counter(c.llm['verdict'] for c in judged).most_common())
                   + (f" (not 'post': {', '.join('`' + c.id + '`' for c in skips[:8])})" if skips else ""))
    else:
        out.append("- **No LLM was used** (or every call failed): ranking is rules only, so subtle drama "
                   "without loud keywords is under-scored and calm endings with dramatic words can slip through.")
    weak = sorted({c.segments[0].episode for c in result.promos} - {
        c.segments[0].episode for c in result.promos if c.scores.get("final", 0) >= LOW_SCORE})
    no_promo = sorted({e.number for e in result.episodes} - {c.segments[0].episode for c in result.promos})
    if weak or no_promo:
        out.append(f"- Episodes with no promo scoring ≥ {LOW_SCORE}: {weak or 'none'}; "
                   f"episodes with no promo at all: {no_promo or 'none'}.")
    flags = Counter(f.split(" (")[0] for c in result.promos for f in c.flags)
    if flags:
        out.append("- Warning flags across all promos: " + ", ".join(f"{k} ×{v}" for k, v in flags.most_common(8)))
    st = result.llm_stats
    if st.get("failed") or st.get("skipped"):
        out.append(f"- LLM failures: {st.get('failed', 0)} failed, {st.get('skipped', 0)} skipped "
                   f"(last error: {_md(st.get('last_error', ''))[:160]}). Those promos use rule scores only.")
    out += [
        "- Known blind spots (found while building it): keyword rules miss quiet reveals with no loud words; "
        "an episode's last beat is not always a cliffhanger (Ep 4 ends on a triumphant exit); stitched promos "
        "score high on rules but the LLM rates them harder to follow; the LLM judge is generous (mostly 'post'), "
        "so it is better at catching bad promos than at ordering good ones.",
    ]
    return out


def write_markdown(result: RunResult, meta: dict, path: Path, top: int = 15) -> None:
    L: list[str] = []
    L.append(f"# Promo report: {meta['show']}\n")
    L.append(f"Run {meta['finished_at']} · {meta['episodes']} episodes ({meta['runtime_minutes']} min of show) · "
             f"{meta['candidates']} candidate cuts → {len(result.promos)} distinct promos · "
             f"LLM: **{meta['llm']['label']}** ({result.llm_stats.get('judged', 0)} judged)\n")
    L.append("Scores are 0–1. `final` = 0.5 × rules + 0.5 × LLM when the LLM judged it, otherwise rules only. "
             "Posting order: LLM-checked promos first (an LLM 'skip' goes to the end of that group), then by "
             "final score plus a small early-episode bonus (up to +0.05), never repeating an episode in the "
             "top 10.\n")

    L.append(f"## Top {top} promos (post in this order)\n")
    for c in result.promos[:top]:
        s = c.scores
        llm = f"{s['llm']:.2f}" if s.get("llm") is not None else "—"
        verdict = f" · LLM verdict **{c.llm['verdict']}**" if c.llm else ""
        L.append(f"### {c.rank}. `{c.id}` · {c.shape} · {c.duration:.1f}s{verdict}")
        L.append(f"- **Segments:** {segments_text(c)}")
        L.append(f"- **Hook (first 3 s):** {_md(c.hook_text)}")
        L.append(f"- **Cliffhanger:** {_md(c.cliffhanger_text)}")
        L.append(f"- **Why it works:** {_md(c.why)}")
        L.append(f"- **Scores:** final {s.get('final', s['rule']):.2f} (rules {s['rule']:.2f}, LLM {llm}) · "
                 f"hook {s['hook']:.2f} · cliffhanger {s['cliffhanger']:.2f} · standalone {s['standalone']:.2f}")
        if c.llm_adjusted:
            L.append(f"- **Cut adjusted by LLM suggestion:** {c.llm_adjusted}")
        if c.flags:
            L.append(f"- **Flags:** {'; '.join(_md(f) for f in c.flags)}")
        L.append("")

    L.append("## Coverage by episode\n")
    L.append("| Ep | Promos | Best rank | Best final | Best promo hook |")
    L.append("|---:|---:|---:|---:|---|")
    by_ep: dict[int, list[Candidate]] = {}
    for c in result.promos:
        by_ep.setdefault(c.segments[0].episode, []).append(c)
    for ep in result.episodes:
        cs = by_ep.get(ep.number, [])
        if not cs:
            L.append(f"| {ep.number} | 0 | — | — | *(no promo)* |")
            continue
        best = min(cs, key=lambda c: c.rank)
        top_final = max(c.scores.get("final", c.scores["rule"]) for c in cs)
        mark = " ⚠" if top_final < LOW_SCORE else ""
        L.append(f"| {ep.number} | {len(cs)} | {best.rank} | {top_final:.2f}{mark} | {_md(best.hook_text)[:70]} |")
    L.append("")

    L.append("## Where it picks badly\n")
    L += picks_badly(result)
    L.append("")
    L.append("## Files\n")
    L.append("`promos.json` (everything) · `promos.csv` (spreadsheet) · `report.html` (filterable table) · "
             "`llm_raw.jsonl` (unedited LLM replies) · `run_meta.json` (settings, timings, versions)")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- html


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Promos: __SHOW__</title>
<style>
:root{--bg:#fbfaf8;--fg:#1d1c1a;--mute:#6b675f;--line:#e4e0d8;--card:#fff;--accent:#b4462a;--good:#2f7d4f;--warn:#a26a00}
@media (prefers-color-scheme: dark){:root{--bg:#161514;--fg:#ece9e3;--mute:#a19c92;--line:#33302b;--card:#1e1d1b;--accent:#e07a5a;--good:#6cc08e;--warn:#e0b04c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1400px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--mute);margin:0 0 18px}
.controls{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;margin-bottom:12px;padding:12px;background:var(--card);border:1px solid var(--line);border-radius:8px}
.controls label{color:var(--mute);font-size:13px;display:flex;gap:6px;align-items:center}
input[type=search]{min-width:220px;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
select{padding:5px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
.count{margin-left:auto;color:var(--mute);font-variant-numeric:tabular-nums}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;min-width:1100px}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{position:sticky;top:0;background:var(--card);cursor:pointer;user-select:none;font-size:12px;text-transform:uppercase;letter-spacing:.03em;color:var(--mute);white-space:nowrap}
th:hover{color:var(--fg)}td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.seg{white-space:nowrap;font-variant-numeric:tabular-nums;font-size:13px}
.q{max-width:280px}.why{max-width:300px;color:var(--mute)}
.pill{display:inline-block;padding:1px 7px;border-radius:999px;font-size:12px;border:1px solid var(--line)}
.post{color:var(--good);border-color:currentColor}.maybe{color:var(--warn);border-color:currentColor}.skip{color:var(--accent);border-color:currentColor}
.flag{display:block;font-size:12px;color:var(--warn)}
.bar{display:inline-block;height:6px;border-radius:3px;background:var(--accent);vertical-align:middle;margin-left:6px}
</style></head><body><main>
<h1>Promo candidates: __SHOW__</h1>
<p class="sub">__SUB__</p>
<div class="controls">
  <input type="search" id="q" placeholder="Search hook, cliffhanger, why, id…" aria-label="Search">
  <label>Shape <select id="shape"><option value="">all</option><option>continuous</option><option>stitched</option></select></label>
  <label>LLM verdict <select id="verdict"><option value="">any</option><option>post</option><option>maybe</option><option>skip</option><option value="none">not judged</option></select></label>
  <label>Min final <input type="range" id="min" min="0" max="1" step="0.05" value="0"> <span id="minv">0.00</span></label>
  <label><input type="checkbox" id="noflags"> no flags only</label>
  <span class="count" id="count"></span>
</div>
<div class="wrap"><table><thead><tr>
<th data-k="rank">#</th><th data-k="id">Promo</th><th data-k="duration">Len</th><th data-k="segments_text">Segments</th>
<th data-k="hook">Hook (first 3 s)</th><th data-k="cliffhanger">Cliffhanger</th><th data-k="why">Why</th>
<th data-k="final">Final</th><th data-k="rule">Rules</th><th data-k="llm">LLM</th><th data-k="verdict">Verdict</th>
</tr></thead><tbody id="rows"></tbody></table></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById('data').textContent);
const rows = data.map(p => ({...p, final: p.scores.final ?? p.scores.rule, rule: p.scores.rule, llm: p.scores.llm}));
let sortKey = 'rank', sortDir = 1;
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const num = v => v == null ? '—' : v.toFixed(2);
function render(){
  const q = document.getElementById('q').value.toLowerCase(), shape = document.getElementById('shape').value,
        verdict = document.getElementById('verdict').value, min = +document.getElementById('min').value,
        noflags = document.getElementById('noflags').checked;
  document.getElementById('minv').textContent = min.toFixed(2);
  const list = rows.filter(p => (!shape || p.shape === shape)
      && (!verdict || (verdict === 'none' ? !p.verdict : p.verdict === verdict))
      && p.final >= min && (!noflags || !p.flags.length)
      && (!q || [p.id, p.hook, p.cliffhanger, p.why, p.segments_text].join(' ').toLowerCase().includes(q)))
    .sort((a, b) => { const x = a[sortKey] ?? -1, y = b[sortKey] ?? -1; return (x > y ? 1 : x < y ? -1 : 0) * sortDir; });
  document.getElementById('count').textContent = list.length + ' of ' + rows.length + ' promos';
  document.getElementById('rows').innerHTML = list.map(p => `<tr>
    <td class="num">${p.rank}</td><td>${esc(p.id)}<br><span class="pill">${p.shape}</span></td>
    <td class="num">${p.duration.toFixed(1)}s</td>
    <td class="seg">${esc(p.segments_text).replaceAll(' + ', '<br>')}</td>
    <td class="q">${esc(p.hook)}</td><td class="q">${esc(p.cliffhanger)}</td>
    <td class="why">${esc(p.why)}${p.flags.map(f => `<span class="flag">⚠ ${esc(f)}</span>`).join('')}</td>
    <td class="num">${num(p.final)}<span class="bar" style="width:${Math.round(p.final * 40)}px"></span></td>
    <td class="num">${num(p.rule)}</td><td class="num">${num(p.llm)}</td>
    <td>${p.verdict ? `<span class="pill ${p.verdict}">${p.verdict}</span>` : '—'}</td></tr>`).join('');
}
document.querySelectorAll('th').forEach(th => th.addEventListener('click', () => {
  const k = th.dataset.k; sortDir = (sortKey === k) ? -sortDir : (['final','rule','llm','duration'].includes(k) ? -1 : 1);
  sortKey = k; render(); }));
['q','shape','verdict','min','noflags'].forEach(id => document.getElementById(id).addEventListener('input', render));
render();
</script></body></html>
"""


def write_html(result: RunResult, meta: dict, path: Path) -> None:
    records = [promo_record(c) for c in result.promos]
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")  # can't close the <script> early
    sub = (f"{meta['episodes']} episodes · {len(records)} promos · LLM: {meta['llm']['label']} · "
           f"run {meta['finished_at']}. Click a column to sort.")
    page = (HTML_TEMPLATE.replace("__SHOW__", html.escape(meta["show"]))
            .replace("__SUB__", html.escape(sub)).replace("__DATA__", payload))
    path.write_text(page, encoding="utf-8")


# ---------------------------------------------------------------- all


def write_outputs(result: RunResult, meta: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / name for name in
             ("promos.json", "promos.csv", "report.md", "report.html", "run_meta.json")}
    write_json(result, meta, paths["promos.json"])
    write_csv(result, paths["promos.csv"])
    write_markdown(result, meta, paths["report.md"])
    write_html(result, meta, paths["report.html"])
    paths["run_meta.json"].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths
