"""A quick BLIND human check of the tool's ranking (about 10-12 minutes).

  python human_check.py make      # writes outputs/human_rating.md (clips to read) + human_rating.csv (to fill in)
  python human_check.py score     # after you fill the CSV: reveals the key and reports the result

How it works:
- 12 clips: the tool's top 8 picks + 4 controls from the bottom half of its list,
  shuffled and labelled A-L. Scores, ranks and LLM opinions are hidden.
- You read each clip as plain script text and mark would_post (y/n) and score (1-5).
- 'score' then compares: do you rate the tool's top picks higher than its controls?
  That tests the ranking directly, and the hidden controls stop either of us from
  rating what we expect instead of what we read.
"""

from __future__ import annotations

import csv
import json
import random
import statistics as st
import sys
from pathlib import Path

from promo_finder.parser import load_show

OUT = Path("outputs")
SHEET, FORM, KEY, RESULT = (OUT / "human_rating.md", OUT / "human_rating.csv",
                            OUT / "human_rating_key.json", OUT / "human_check.md")
N_TOP, N_CONTROL, SEED = 8, 4, 2026


def clip_text(promo: dict, episodes: dict) -> list[str]:
    lines = []
    for i, seg in enumerate(promo["segments"]):
        if i:
            lines.append("   --- (cut to a later moment) ---")
        for ev in episodes[seg["episode"]].events:
            if seg["start_sec"] - 0.05 <= ev.start < seg["end_sec"] and not ev.is_end_marker:
                if ev.kind == "DIALOGUE":
                    lines.append(f'   {ev.speaker.title()}: "{ev.text}"')
                elif ev.kind in ("ACTION", "SFX"):
                    lines.append(f"   [{ev.kind.lower()}] {ev.text}")
    return lines


def make() -> None:
    data = json.loads((OUT / "promos.json").read_text(encoding="utf-8"))
    promos = data["promos"]
    episodes = {e.number: e for e in load_show(data["meta"]["input"])}
    rng = random.Random(SEED)
    top = promos[:N_TOP]
    bottom_half = promos[len(promos) // 2:]
    controls = rng.sample(bottom_half, N_CONTROL)
    picked = [("top", p) for p in top] + [("control", p) for p in controls]
    rng.shuffle(picked)

    labels = [chr(ord("A") + i) for i in range(len(picked))]
    key = {lab: {"group": grp, "id": p["id"], "rank": p["rank"],
                 "final": p["scores"].get("final") or p["scores"]["rule"],
                 "llm_verdict": (p.get("llm") or {}).get("verdict")}
           for lab, (grp, p) in zip(labels, picked)}
    KEY.write_text(json.dumps(key, indent=2), encoding="utf-8")

    md = ["# Blind promo rating", "",
          "Read each clip as if it popped up on TikTok and you've never seen the show.",
          "For each one, fill in `outputs/human_rating.csv`:",
          "- **would_post**: `y` if you'd post it as an ad for the show, else `n`",
          "- **score**: 1 (would scroll past) to 5 (must watch the next episode)",
          "- **note**: optional, a few words on why", "",
          "Don't look at report.md or the scores while rating. About 1 minute per clip.", ""]
    for lab, (_, p) in zip(labels, picked):
        md += [f"## Clip {lab}  ({p['duration']:.0f} s)", ""] + clip_text(p, episodes) + [""]
    SHEET.write_text("\n".join(md), encoding="utf-8")
    with FORM.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "would_post", "score", "note"])
        for lab in labels:
            w.writerow([lab, "", "", ""])
    print(f"1. Read the clips in   {SHEET.resolve()}")
    print(f"2. Fill in             {FORM.resolve()}  (would_post y/n, score 1-5)")
    print("3. Then run            python human_check.py score")


def score() -> None:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(FORM.open(encoding="utf-8-sig")))
    rated = []
    for r in rows:
        lab = r["clip"].strip().upper()
        post, sc = r["would_post"].strip().lower(), r["score"].strip()
        if lab in key and post in ("y", "n") and sc.isdigit() and 1 <= int(sc) <= 5:
            rated.append({**key[lab], "label": lab, "post": post == "y", "score": int(sc),
                          "note": r.get("note", "").strip()})
    if len(rated) < len(key):
        missing = sorted(set(key) - {r["label"] for r in rated})
        print(f"Note: clips not (validly) rated yet: {missing} - fill would_post (y/n) and score (1-5).")
    if not rated:
        return

    def summary(group: str) -> tuple[int, int, float]:
        g = [r for r in rated if r["group"] == group]
        return sum(r["post"] for r in g), len(g), (st.mean(r["score"] for r in g) if g else 0.0)

    tp, tn, ts = summary("top")
    cp, cn, cs = summary("control")
    llm_rows = [r for r in rated if r["llm_verdict"]]
    agree = sum((r["llm_verdict"] == "post") == r["post"] for r in llm_rows)
    L = ["# Blind human check", "",
         f"{len(rated)} clips rated blind (tool's top {N_TOP} picks mixed with {N_CONTROL} low-ranked controls).", "",
         "| Group | Would post | Mean score (1-5) |", "|---|---|---|",
         f"| Tool's top picks | {tp}/{tn} | {ts:.1f} |",
         f"| Low-ranked controls | {cp}/{cn} | {cs:.1f} |", ""]
    if tn and cn:
        verdict = ("The ranking matches human taste: top picks were preferred over controls."
                   if ts > cs and tp / tn > cp / max(1, cn) else
                   "The ranking does NOT clearly match human taste on this sample.")
        L += [f"**{verdict}** (small sample - treat as a sanity check, not proof.)", ""]
    if llm_rows:
        L += [f"LLM verdict ('post' vs not) agreed with the human on {agree}/{len(llm_rows)} judged clips.", ""]
    L += ["| Clip | Group | Tool rank | Human: post? | Human score | LLM verdict | Note |",
          "|---|---|---:|---|---:|---|---|"]
    for r in sorted(rated, key=lambda r: r["rank"]):
        L.append(f"| {r['label']} | {r['group']} | {r['rank']} | {'y' if r['post'] else 'n'} | {r['score']} | "
                 f"{r['llm_verdict'] or '—'} | {r['note']} |")
    RESULT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:9]))
    print(f"\nFull result: {RESULT.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("make", "score"):
        print(__doc__)
        sys.exit(2)
    make() if sys.argv[1] == "make" else score()
