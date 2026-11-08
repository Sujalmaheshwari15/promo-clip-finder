"""LLM judge: a second opinion on the top rule-scored promos.

Design choices (short version):
- The model sees ONLY the clip - exactly what a new viewer would see. No
  logline, no show summary. That is the honest test of "makes sense on its own".
- Every line it sees carries a beat ID like [E07-B031]. If it wants to move a
  cut it must answer with one of those IDs; it is told never to write
  timestamps. Code turns IDs back into times, so a timestamp can't be invented.
- Nothing the model says is trusted blindly:
    * scores are clamped to 0-10, verdict must be post/maybe/skip
    * a suggested cut must be a listed ID, a legal cut point (not mid-line),
      and keep the promo 20-60 s - otherwise it is ignored and flagged
    * a "quote" that doesn't appear in the clip's script is dropped
    * any timestamp the model writes anyway is removed from its text
- Only the top N (default 60) promos are judged: that's where an editor will
  look, and it keeps a free-tier run to a few minutes.
- If the call fails for any reason, the promo keeps its rule score and gets
  an "llm failed" flag. Every raw reply is logged unedited to llm_raw.jsonl.
- final score = 0.5 * rules + 0.5 * LLM (LLM part = 0.4 hook + 0.4 cliff + 0.2 standalone).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

from .candidates import CONTINUOUS_MAX, CONTINUOUS_MIN, TOLERANCE, Candidate
from .features import Scorer
from .llm import LLMClient, LLMError
from .parser import Event, fmt_time

LLM_WEIGHTS = {"hook": 0.4, "cliffhanger": 0.4, "standalone": 0.2}
RULE_SHARE = 0.5
MAX_CONSECUTIVE_FAILURES = 3   # after this many failed calls in a row, stop calling the LLM
VERDICTS = {"post", "maybe", "skip"}
TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?:[.:]\d{1,2})?\b")

SYSTEM_PROMPT = """You are a promo editor for short vertical drama series. You pick 20-60 second clips \
that are posted on TikTok, Instagram Reels and YouTube Shorts to pull NEW viewers into a show.

You will see one candidate clip as a list of script beats, in order. Each beat starts with an ID in \
square brackets like [E07-B031]. CAMERA lines describe what is on screen; other lines are actions, \
sounds and dialogue. You see ONLY what the viewer of the promo would see - judge it as someone who has \
never watched the show.

Score three things from 0 to 10:
- hook: do the first 3 seconds make a scrolling viewer stop? 10 = a slap, a shocking reveal or a line \
you can't scroll past. 5 = mildly interesting. 0 = calm, scenery, or confusing.
- cliffhanger: does it cut at an unresolved moment so the viewer must know what happens next? 10 = a \
question, threat or reveal left hanging. 5 = some tension left. 0 = resolved, calm or romantic ending.
- standalone: does it make sense with zero context? 10 = clear who is in conflict and why it matters. \
0 = needs earlier episodes to follow.

Rules - follow them exactly:
1. Never write timestamps or times. Refer to moments only by the beat IDs shown.
2. You may suggest a tighter cut with new_start_beat and/or new_end_beat. Use only IDs from the list. \
Leave them null if the cut is already right.
3. hook_quote and cliffhanger_quote must be copied word for word from the dialogue or action text shown \
(a short fragment is fine). Use "" if there is nothing to quote.
4. Reply with ONE JSON object and nothing else:
{"hook": int, "cliffhanger": int, "standalone": int, "verdict": "post" | "maybe" | "skip", \
"why": "one sentence on why it works or fails", "hook_quote": str, "cliffhanger_quote": str, \
"new_start_beat": str | null, "new_end_beat": str | null}"""


# ---------------------------------------------------------------- prompt


def clip_events(scorer: Scorer, cand: Candidate) -> list[list[Event]]:
    """Every event (camera included) inside each segment, in order."""
    out = []
    for seg in cand.segments:
        evs = [e for e in scorer.indexes[seg.episode].ep.events
               if e.start >= seg.start - TOLERANCE and e.start < seg.end and not e.is_end_marker]
        out.append(evs)
    return out


def render_event(ev: Event) -> str:
    if ev.kind == "DIALOGUE":
        emo = f" ({', '.join(ev.emotion)})" if ev.emotion else ""
        how = f" [{ev.delivery}]" if ev.delivery else ""
        return f'[{ev.beat_id}] {ev.speaker}{how}{emo}: "{ev.text}"'
    return f"[{ev.beat_id}] {ev.kind}: {ev.text}"


def build_prompt(scorer: Scorer, cand: Candidate) -> str:
    parts = [f"Candidate promo, {cand.duration:.0f} seconds"
             + (f", made of {len(cand.segments)} pieces joined together" if len(cand.segments) > 1 else "")
             + ".", ""]
    for i, evs in enumerate(clip_events(scorer, cand)):
        if i:
            parts += ["", "--- (cut to a later moment) ---", ""]
        parts += [render_event(e) for e in evs]
    parts += ["", "The first 3 seconds are the first few lines above. The clip ends after the last line.",
              "Reply with the JSON object only."]
    return "\n".join(parts)


# ---------------------------------------------------------------- reply validation


def extract_json(text: str) -> dict:
    """Pull the first {...} object out of a reply that may have ``` fences or chatter."""
    cleaned = re.sub(r"```(?:json)?", "", text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    obj = json.loads(cleaned[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("reply JSON is not an object")
    return obj


def _norm_text(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", s.lower().replace("’", "'")).split())


def quote_in_clip(quote: str, evs: list[Event]) -> bool:
    q = _norm_text(quote)
    return bool(q) and any(q in _norm_text(e.text) for e in evs)


def _score(value, name: str, problems: list[str]) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        problems.append(f"{name} not a number ({value!r}) - set to 0")
        return 0.0
    if not 0 <= v <= 10:
        problems.append(f"{name}={v} out of range - clamped")
    return max(0.0, min(10.0, v))


def validate_reply(obj: dict, scorer: Scorer, cand: Candidate) -> tuple[dict, list[str]]:
    """Return (clean judgement, list of problems found). Never raises on content."""
    problems: list[str] = []
    evs = [e for seg in clip_events(scorer, cand) for e in seg]
    clean = {k: _score(obj.get(k), k, problems) for k in LLM_WEIGHTS}

    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        problems.append(f"verdict {verdict!r} not post/maybe/skip - set to 'maybe'")
        verdict = "maybe"
    clean["verdict"] = verdict

    why = str(obj.get("why") or "").strip()
    if TIMESTAMP_RE.search(why):
        problems.append("model wrote a timestamp in 'why' - removed")
        why = re.sub(r"\s*\b(?:at|around|by|near)?\s*" + TIMESTAMP_RE.pattern, "", why)
        why = re.sub(r"\s{2,}", " ", why).strip()
    clean["why"] = why[:240]

    for key in ("hook_quote", "cliffhanger_quote"):
        q = str(obj.get(key) or "").strip().strip('"“”')
        if q and not quote_in_clip(q, evs):
            problems.append(f"{key} not found in the script - dropped: {q[:50]!r}")
            q = ""
        clean[key] = q

    clean["new_start_beat"] = obj.get("new_start_beat") or None
    clean["new_end_beat"] = obj.get("new_end_beat") or None
    return clean, problems


def apply_cut_suggestion(clean: dict, scorer: Scorer, cand: Candidate) -> tuple[bool, list[str]]:
    """Move the first segment's start and/or last segment's end to the suggested beats, if legal."""
    new_s, new_e = clean.get("new_start_beat"), clean.get("new_end_beat")
    if not new_s and not new_e:
        return False, []
    problems: list[str] = []
    first, last = cand.segments[0], cand.segments[-1]
    in_first = {e.beat_id for e in clip_events(scorer, cand)[0]}
    in_last = {e.beat_id for e in clip_events(scorer, cand)[-1]}

    start_t, start_ev = first.start, scorer.events[first.start_beat]
    end_t, end_ev = last.end, scorer.events[last.end_beat]
    if new_s:
        idx = scorer.indexes[first.episode]
        legal = {b.beat_id: t for t, b in idx.starts}
        if new_s not in in_first:
            problems.append(f"suggested start {new_s!r} is not a beat in the clip's opening piece - ignored")
        elif new_s not in legal:
            problems.append(f"suggested start {new_s} is not a legal cut (camera note or mid-line) - ignored")
        else:
            start_t, start_ev = legal[new_s], scorer.events[new_s]
    if new_e:
        idx = scorer.indexes[last.episode]
        legal = {b.beat_id: t for t, b in idx.ends}
        if new_e not in in_last:
            problems.append(f"suggested end {new_e!r} is not a beat in the clip's last piece - ignored")
        elif new_e not in legal:
            problems.append(f"suggested end {new_e} is not a legal cut (camera note or mid-line) - ignored")
        else:
            end_t, end_ev = legal[new_e], scorer.events[new_e]

    if (start_t, end_t) == (first.start, last.end):
        return False, problems
    if len(cand.segments) == 1:
        lengths = [end_t - start_t]
    else:
        lengths = [first.end - start_t] + [s.duration for s in cand.segments[1:-1]] + [end_t - last.start]
    total = sum(lengths)
    if any(x <= 1.0 for x in lengths) or not (CONTINUOUS_MIN - TOLERANCE <= total <= CONTINUOUS_MAX + TOLERANCE):
        problems.append(f"suggested cut gives {total:.1f}s (must stay 20-60 s) - ignored")
        return False, problems

    before = f"{fmt_time(first.start)}-{fmt_time(last.end)}"
    if len(cand.segments) == 1:
        cand.segments[0] = scorer.indexes[first.episode].make_segment(start_t, start_ev, end_t, end_ev)
    else:
        cand.segments[0] = scorer.indexes[first.episode].make_segment(
            start_t, start_ev, first.end, scorer.events[first.end_beat])
        cand.segments[-1] = scorer.indexes[last.episode].make_segment(
            last.start, scorer.events[last.start_beat], end_t, end_ev)
    cand.ends_episode = abs(cand.segments[-1].end - scorer.indexes[last.episode].last_beat_end) <= TOLERANCE
    cand.llm_adjusted = f"{before} -> {fmt_time(cand.segments[0].start)}-{fmt_time(cand.segments[-1].end)}"
    return True, problems


def rank_key(c: Candidate) -> tuple:
    """Sort key: LLM-checked promos first (by final score), then unchecked ones (by rule score).

    Without this, a promo the LLM marked down would fall below promos it never looked
    at, simply because those kept their (unverified) rule score.
    """
    return (c.scores.get("llm") is None, -c.scores.get("final", c.scores.get("rule", 0.0)), c.id)


# ---------------------------------------------------------------- main loop


def cache_key(client: LLMClient, prompt: str) -> str:
    """Same model + same prompts = same question; reuse the earlier answer."""
    return hashlib.sha256(f"{client.label}\n{SYSTEM_PROMPT}\n{prompt}".encode("utf-8")).hexdigest()


def load_cache(path: str | Path | None) -> dict[str, str]:
    cache: dict[str, str] = {}
    if path and Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                cache[rec["key"]] = rec["reply"]
            except (json.JSONDecodeError, KeyError):
                continue  # a half-written line from an interrupted run
    return cache


def judge_candidates(cands: list[Candidate], scorer: Scorer, client: LLMClient | None, top_n: int = 60,
                     log_path: str | Path | None = None, progress=None,
                     cache_path: str | Path | None = None) -> dict:
    """Judge the top_n candidates (by rule score) in place. Returns run stats.

    With cache_path, replies that parsed correctly are saved, and a re-run asks the
    model only about promos it hasn't answered before (free tiers allow ~20 calls/day).
    """
    stats = {"judged": 0, "cached": 0, "failed": 0, "skipped": 0, "adjusted": 0, "problems": 0, "seconds": 0.0}
    for c in cands:  # everyone starts with final = rule score
        c.scores["final"] = c.scores["rule"]
        c.scores.setdefault("llm", None)
    if client is None:
        return stats
    targets = sorted(cands, key=lambda c: -c.scores["rule"])[:top_n]
    log = open(log_path, "a", encoding="utf-8") if log_path else None
    cache = load_cache(cache_path)
    cache_out = open(cache_path, "a", encoding="utf-8") if cache_path else None
    started = time.monotonic()
    consecutive_failures, last_error = 0, ""
    try:
        for i, cand in enumerate(targets, 1):
            prompt = build_prompt(scorer, cand)
            key = cache_key(client, prompt)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES and key not in cache:
                # The provider is clearly down / misconfigured: stop spending time on it.
                cand.flags.append("llm skipped (provider kept failing) - rule score only")
                stats["skipped"] += 1
                if progress and stats["skipped"] == 1:  # one line, not one per promo
                    progress(i, len(targets), cand,
                             f"provider keeps failing - skipping the remaining {len(targets) - i + 1} promos")
                continue
            record = {"candidate": cand.id, "provider": client.label,
                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                      "prompt": prompt, "reply": None, "error": None, "cached": key in cache}
            status = "judged"
            try:
                if key in cache:
                    status = "from cache"
                    reply_text = cache[key]
                    record.update(reply=reply_text)
                    stats["cached"] += 1
                else:
                    reply = client.complete(SYSTEM_PROMPT, prompt)
                    reply_text = reply.text
                    record.update(reply=reply_text, model=reply.model, latency_s=reply.latency_s,
                                  attempts=reply.attempts)
                parsed = extract_json(reply_text)
                if cache_out and key not in cache:
                    cache_out.write(json.dumps({"key": key, "candidate": cand.id, "reply": reply_text},
                                               ensure_ascii=False) + "\n")
                    cache_out.flush()
                    cache[key] = reply_text
                clean, problems = validate_reply(parsed, scorer, cand)
                adjusted, cut_problems = apply_cut_suggestion(clean, scorer, cand)
                problems += cut_problems
                if adjusted:
                    stats["adjusted"] += 1
                    scorer.score(cand)  # re-score rules on the new cut
                llm = sum(LLM_WEIGHTS[k] * clean[k] for k in LLM_WEIGHTS) / 10
                cand.llm = clean
                cand.llm_problems = problems
                cand.scores["llm"] = round(llm, 3)
                cand.scores["final"] = round(RULE_SHARE * cand.scores["rule"] + (1 - RULE_SHARE) * llm, 3)
                if problems:
                    stats["problems"] += len(problems)
                    cand.flags.append(f"llm reply had {len(problems)} problem(s) - fixed/ignored")
                stats["judged"] += 1
                consecutive_failures = 0
            except (LLMError, ValueError) as e:
                record["error"] = str(e)
                cand.scores["llm"] = None
                cand.scores["final"] = cand.scores["rule"]
                cand.flags.append(f"llm failed - rule score only ({str(e)[:160]})")
                stats["failed"] += 1
                status = f"FAILED: {str(e)[:70]}"
                consecutive_failures += 1
                last_error = str(e)
                if getattr(e, "fatal", False):  # bad key / unknown model / SSL: every call would fail
                    consecutive_failures = MAX_CONSECUTIVE_FAILURES
            if log:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
            if progress:
                progress(i, len(targets), cand, status)
    finally:
        if log:
            log.close()
        if cache_out:
            cache_out.close()
    stats["seconds"] = round(time.monotonic() - started, 1)
    if last_error:
        stats["last_error"] = last_error[:300]
    return stats


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    import argparse
    import tempfile

    from .candidates import deduplicate, generate_continuous, generate_stitched
    from .llm import add_llm_args, client_from_args
    from .parser import load_show

    ap = argparse.ArgumentParser(description="Judge the top rule-scored promos with an LLM")
    ap.add_argument("input", help="folder or .zip of scripts")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--show-prompt", action="store_true", help="print the first prompt sent")
    add_llm_args(ap)
    args = ap.parse_args(argv)

    if args.list_models:
        try:
            client = client_from_args(args)
            print("\n".join(client.list_models()) if client else "provider 'none' has no models")
        except LLMError as e:
            print(f"could not list models: {e}")
        return

    say = lambda msg: print(msg, file=sys.stderr, flush=True)  # noqa: E731
    say("Reading scripts...")
    episodes = load_show(args.input)
    say(f"  {len(episodes)} episodes. Generating and scoring candidates (about 10 s)...")
    scorer = Scorer(episodes)
    cands = generate_continuous(episodes) + generate_stitched(
        episodes, hook_rank=scorer.hook_rank, cliff_rank=scorer.cliff_rank)
    for c in cands:
        scorer.score(c)
    kept, _ = deduplicate(cands)
    say(f"  {len(cands)} candidates -> {len(kept)} after de-duplication.")
    try:
        client = client_from_args(args)
    except LLMError as e:
        print(f"LLM unavailable ({e}) - continuing with rules only")
        client = None
    # A new file per run, so two runs at once (or a file still open elsewhere) never collide.
    log_path = Path(tempfile.gettempdir()) / f"promo_llm_raw_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    if args.show_prompt:
        print(SYSTEM_PROMPT, "\n\n=====\n", build_prompt(scorer, kept[0]), "\n=====\n")
    if client:
        say(f"Asking {client.label} about the top {args.top} (first reply can take ~10 s)...")
    stats = judge_candidates(kept, scorer, client, top_n=args.top, log_path=log_path,
                             progress=lambda i, n, c, status: say(f"  {i}/{n} {c.id}: {status}"))
    print(f"Provider: {client.label if client else 'none'} | {stats} | raw log: {log_path}")
    for c in sorted(kept, key=rank_key)[:args.top]:
        llm = c.llm
        print(f"\n{c.id} final={c.scores['final']:.2f} rule={c.scores['rule']:.2f} llm={c.scores['llm']}")
        if llm:
            print(f"  verdict={llm['verdict']} hook={llm['hook']:.0f} cliff={llm['cliffhanger']:.0f} "
                  f"stand={llm['standalone']:.0f} | {llm['why']}")
        if c.llm_adjusted:
            print(f"  cut adjusted: {c.llm_adjusted}")
        for prob in c.llm_problems:
            print(f"  ! {prob}")
        for f in c.flags:
            if f.startswith(("llm failed", "llm skipped")):
                print(f"  ! {f}")


if __name__ == "__main__":
    main()
