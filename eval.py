"""Check a finished run against the original scripts.

  python eval.py                                   # checks outputs/promos.json
  python eval.py --out outputs --gold gold.csv     # + compare with hand-picked promo moments

Hard checks (any failure -> exit code 1). Each one re-reads the scripts and
verifies the promo independently, so a bug in the pipeline can't hide:
  1. length        every promo is 20-60 s (segments of stitched promos add up)
  2. real cuts     every segment starts on a DIALOGUE/ACTION beat's start and ends on a story beat's end
  3. hook window   at least one story beat starts in the first 3 s
  4. mid-line      no cut lands inside a spoken line
  5. order         stitched segments run forward in story time (same episode later, or next episode)
  6. overlap       no two promos share more than 35% of their footage
  7. quotes        every hook / cliffhanger line and every LLM quote appears in the script
  8. ranking       ranks are 1..N and the top 10 don't repeat an episode while an unused one is available

Soft checks (reported, never fail the run):
  - episode coverage (promos per episode, episodes with no strong promo)
  - rules vs LLM agreement (correlation, biggest disagreements)
  - LLM verdict mix and score spread (is the judge actually discriminating?)
  - optional: precision / recall against a hand-labelled gold file
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from promo_finder.candidates import TOLERANCE
from promo_finder.parser import fmt_time, load_show, parse_time

STORY_KINDS = {"ACTION", "DIALOGUE", "SFX"}
START_KINDS = {"ACTION", "DIALOGUE"}


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", s.lower().replace("’", "'")).split())


class Script:
    """Independent lookups over the parsed scripts (does not reuse candidate code)."""

    def __init__(self, episodes):
        self.eps = {e.number: e for e in episodes}

    def beats(self, ep):
        return [e for e in self.eps[ep].events if e.kind in STORY_KINDS and not e.is_end_marker]

    def lines(self, ep):
        return [e for e in self.eps[ep].events if e.kind == "DIALOGUE"]

    def inside_line(self, ep, t):
        return any(d.start < t - TOLERANCE and d.end > t + TOLERANCE for d in self.lines(ep))

    def text_in_episode(self, ep, text):
        q = norm(text)
        return bool(q) and any(q in norm(e.text) for e in self.eps[ep].events)


def overlap(a, b):
    shared = sum(max(0.0, min(x["end_sec"], y["end_sec"]) - max(x["start_sec"], y["start_sec"]))
                 for x in a["segments"] for y in b["segments"] if x["episode"] == y["episode"])
    shorter = min(a["duration"], b["duration"])
    return shared / shorter if shorter else 0.0


def said(line: str) -> str:
    """'Anna: "No!"' -> 'No!'; '[action] She slaps him.' -> 'She slaps him.'"""
    m = re.match(r'^[^:"]+:\s*"(.*)"$', line.strip())
    if m:
        return m.group(1)
    return re.sub(r"^\[\w+\]\s*", "", line.strip())


def corr(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy), 2) if sx and sy else None


# ---------------------------------------------------------------- checks


def hard_checks(promos, script, max_overlap):
    fails: dict[str, list[str]] = {k: [] for k in
                                   ("length", "real cuts", "hook window", "mid-line", "order", "overlap",
                                    "quotes", "ranking")}
    for p in promos:
        pid = p["id"]
        total = sum(s["end_sec"] - s["start_sec"] for s in p["segments"])
        if not (20 - TOLERANCE <= total <= 60 + TOLERANCE) or abs(total - p["duration"]) > 0.05:
            fails["length"].append(f"{pid}: {total:.2f}s (reported {p['duration']})")
        for i, s in enumerate(p["segments"]):
            ep, a, b = s["episode"], s["start_sec"], s["end_sec"]
            if ep not in script.eps:
                fails["real cuts"].append(f"{pid}: episode {ep} not in scripts")
                continue
            beats = script.beats(ep)
            if not any(abs(e.start - a) <= TOLERANCE and e.kind in START_KINDS for e in beats):
                fails["real cuts"].append(f"{pid} seg{i + 1}: starts at {fmt_time(a)}, not on a dialogue/action beat")
            if not any(abs(e.end - b) <= TOLERANCE for e in beats):
                fails["real cuts"].append(f"{pid} seg{i + 1}: ends at {fmt_time(b)}, not on a story beat's end")
            if script.inside_line(ep, a) or script.inside_line(ep, b):
                fails["mid-line"].append(f"{pid} seg{i + 1}: {fmt_time(a)}-{fmt_time(b)} cuts inside a spoken line")
            if i == 0 and not any(a - TOLERANCE <= e.start < a + 3.0 for e in beats):
                fails["hook window"].append(f"{pid}: no story beat in the first 3 s")
            if i > 0:
                prev = p["segments"][i - 1]
                ok = (ep == prev["episode"] and a >= prev["end_sec"] - TOLERANCE) or ep == prev["episode"] + 1
                if not ok:
                    fails["order"].append(f"{pid}: segment {i + 1} (Ep{ep}) does not follow segment {i}")
        home = p["segments"][0]["episode"]
        last = p["segments"][-1]["episode"]
        eps_in = {s["episode"] for s in p["segments"]}
        if p.get("hook") and not any(script.text_in_episode(e, said(p["hook"])) for e in eps_in):
            fails["quotes"].append(f"{pid}: hook text not found in Ep{home}: {p['hook'][:50]!r}")
        if p.get("cliffhanger") and not script.text_in_episode(last, said(p["cliffhanger"])):
            fails["quotes"].append(f"{pid}: cliffhanger text not found in Ep{last}: {p['cliffhanger'][:50]!r}")
        for key in ("hook_quote", "cliffhanger_quote"):
            q = (p.get("llm") or {}).get(key)
            if q and not any(script.text_in_episode(e, q) for e in eps_in):
                fails["quotes"].append(f"{pid}: LLM {key} not in script: {q[:50]!r}")

    for i, a in enumerate(promos):
        for b in promos[i + 1:]:
            if {s["episode"] for s in a["segments"]} & {s["episode"] for s in b["segments"]}:
                r = overlap(a, b)
                if r > max_overlap + 0.005:
                    fails["overlap"].append(f"{a['id']} & {b['id']}: {r:.0%} shared")

    ranks = [p["rank"] for p in promos]
    if ranks != list(range(1, len(promos) + 1)):
        fails["ranking"].append("ranks are not 1..N in file order")
    # Top 10: a repeated episode is only wrong if a promo from an unused episode was available
    # (a 3-episode show can't fill 10 slots with distinct episodes).
    seen: set[int] = set()
    for i, p in enumerate(promos[:10]):
        eps = {s["episode"] for s in p["segments"]}
        if eps & seen:
            fresh = next((q for q in promos[i + 1:] if not ({s["episode"] for s in q["segments"]} & seen)), None)
            if fresh:
                fails["ranking"].append(f"{p['id']} (rank {p['rank']}) repeats an episode in the top 10 "
                                        f"while {fresh['id']} (unused episode) ranks lower")
        seen |= eps
    return fails


def soft_checks(promos, episodes):
    out: list[str] = []
    by_ep = Counter(p["segments"][0]["episode"] for p in promos)
    strong = Counter(p["segments"][0]["episode"] for p in promos
                     if (p["scores"].get("final") or p["scores"]["rule"]) >= 0.6)
    all_eps = [e.number for e in episodes]
    out.append(f"- **Coverage:** {sum(1 for e in all_eps if by_ep[e])}/{len(all_eps)} episodes have a promo; "
               f"{sum(1 for e in all_eps if strong[e])}/{len(all_eps)} have one scoring ≥ 0.6. "
               f"Promos per episode: min {min(by_ep[e] for e in all_eps)}, max {max(by_ep[e] for e in all_eps)}.")
    weak = [e for e in all_eps if not strong[e]]
    if weak:
        out.append(f"- **Episodes with no strong promo:** {weak}")

    judged = [p for p in promos if p["scores"].get("llm") is not None]
    if not judged:
        out.append("- **LLM:** not used in this run - no agreement checks possible.")
        return out
    rules = [p["scores"]["rule"] for p in judged]
    llms = [p["scores"]["llm"] for p in judged]
    out.append(f"- **Rules vs LLM:** correlation {corr(rules, llms)} over {len(judged)} judged promos "
               f"(1 = always agree, 0 = unrelated). Mean rules {st.mean(rules):.2f}, mean LLM {st.mean(llms):.2f}.")
    for part in ("hook", "cliffhanger", "standalone"):
        r = [p["scores"][part] for p in judged]
        l = [p["llm"][part] / 10 for p in judged]
        out.append(f"  - {part}: rules mean {st.mean(r):.2f} vs LLM mean {st.mean(l):.2f} (corr {corr(r, l)})")
    dis = sorted(judged, key=lambda p: -abs(p["scores"]["rule"] - p["scores"]["llm"]))[:5]
    out.append("- **Biggest disagreements** (where the rules are most likely wrong):")
    for p in dis:
        out.append(f"  - `{p['id']}` rules {p['scores']['rule']:.2f} vs LLM {p['scores']['llm']:.2f}: {p['llm'].get('why', '')}")
    verdicts = Counter(p["llm"]["verdict"] for p in judged)
    spread = max(llms) - min(llms)
    out.append(f"- **LLM verdicts:** {dict(verdicts)}; LLM score spread {min(llms):.2f}-{max(llms):.2f}. "
               + ("A narrow spread or almost-all 'post' means the judge isn't discriminating much."
                  if spread < 0.3 or verdicts.get("post", 0) / len(judged) > 0.85 else ""))
    by_shape = {}
    for shape in ("continuous", "stitched"):
        s = [p["llm"]["standalone"] for p in judged if p["shape"] == shape]
        if s:
            by_shape[shape] = f"{st.mean(s):.1f}/10 (n={len(s)})"
    out.append(f"- **LLM 'makes sense alone' by shape:** {by_shape}")
    return out


# ---------------------------------------------------------------- gold labels


def gold_check(promos, gold_path: Path, k: int) -> list[str]:
    """Gold CSV columns: episode,start,end[,note]  (times as MM:SS.ss). A hit = ≥50% overlap."""
    gold = []
    with gold_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            gold.append({"episode": int(row["episode"]), "start": parse_time(row["start"]),
                         "end": parse_time(row["end"]), "note": row.get("note", "")})

    def hit(p, g):
        shared = sum(max(0.0, min(s["end_sec"], g["end"]) - max(s["start_sec"], g["start"]))
                     for s in p["segments"] if s["episode"] == g["episode"])
        return shared / max(1e-6, min(p["duration"], g["end"] - g["start"])) >= 0.5

    top = promos[:k]
    precision = sum(any(hit(p, g) for g in gold) for p in top) / max(1, len(top))
    found = [g for g in gold if any(hit(p, g) for p in top)]
    found_any = [g for g in gold if any(hit(p, g) for p in promos)]
    out = [f"- **Gold set:** {len(gold)} hand-picked moments from `{gold_path.name}`.",
           f"- Precision@{k}: {precision:.0%} of the top {k} promos overlap a gold moment.",
           f"- Recall@{k}: {len(found)}/{len(gold)} gold moments are in the top {k}; "
           f"{len(found_any)}/{len(gold)} appear anywhere in the {len(promos)} promos."]
    missed = [g for g in gold if g not in found_any]
    for g in missed[:10]:
        out.append(f"  - missed: Ep{g['episode']} {fmt_time(g['start'])}-{fmt_time(g['end'])} {g['note']}")
    return out


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check a finished promo run against the scripts.")
    ap.add_argument("--out", default="outputs", help="run folder containing promos.json (default: outputs)")
    ap.add_argument("--input", help="scripts folder/zip (default: the one recorded in promos.json)")
    ap.add_argument("--gold", help="optional CSV of hand-picked promo moments: episode,start,end,note")
    ap.add_argument("--k", type=int, default=20, help="top-K for gold precision/recall")
    ap.add_argument("--max-overlap", type=float, default=None, help="default: the value used by the run")
    args = ap.parse_args(argv)

    out = Path(args.out)
    data = json.loads((out / "promos.json").read_text(encoding="utf-8"))
    meta, promos = data["meta"], data["promos"]
    episodes = load_show(args.input or meta["input"])
    script = Script(episodes)
    max_overlap = args.max_overlap if args.max_overlap is not None else meta["settings"]["max_overlap"]

    fails = hard_checks(promos, script, max_overlap)
    n_fail = sum(len(v) for v in fails.values())
    L = [f"# Evaluation: {meta['show']}", "",
         f"Run {meta['finished_at']} · {len(promos)} promos · LLM: {meta['llm']['label']}", "",
         "## Hard checks (verified against the scripts)", "",
         "| Check | Result |", "|---|---|"]
    for name, items in fails.items():
        L.append(f"| {name} | {'PASS' if not items else f'**FAIL** ({len(items)})'} |")
    for name, items in fails.items():
        if items:
            L += ["", f"**{name} failures:**"] + [f"- {x}" for x in items[:15]]
    L += ["", "## Soft checks", ""] + soft_checks(promos, episodes)
    if args.gold:
        L += ["", "## Against hand-picked moments", ""] + gold_check(promos, Path(args.gold), args.k)
    report = "\n".join(L) + "\n"
    (out / "eval.md").write_text(report, encoding="utf-8")
    (out / "eval.json").write_text(json.dumps({"failures": fails, "promos": len(promos)}, indent=2), encoding="utf-8")

    print(f"Hard checks: {'ALL PASS' if not n_fail else f'{n_fail} FAILURE(S)'} "
          + " | ".join(f"{k}: {'ok' if not v else len(v)}" for k, v in fails.items()))
    print(f"Full report: {(out / 'eval.md').resolve()}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
