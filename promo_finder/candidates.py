"""Candidate generation: list every promo cut that is *structurally* valid.

No judgement about quality happens here - that's scoring (Phase 3) and the
LLM judge (Phase 5). This file only answers: "where is it legal to cut?"

Design choices (short version):
- Cuts snap to beats from the parser. A segment starts at the start of a
  DIALOGUE or ACTION beat and ends at the end of any story beat
  (ACTION/DIALOGUE/SFX). Camera notes and "Cut to black." are never cut points.
  So every timestamp we output exists in the script.
- Never cut mid-sentence: a start or end time that falls inside someone's
  spoken line is rejected, even if it lines up with some other beat
  (e.g. an action that starts while a character is still talking).
- Two shapes:
    continuous : one stretch, 20-60 s
    stitched   : hook segment (10-35 s) + a later cliffhanger segment (5-20 s)
                 from the same or the next episode, 20-60 s in total
- Stitched pairs explode combinatorially (~9M for 50 episodes), so we pair
  only the top-k hook segments with the top-k cliffhanger segments. "Top" is
  decided by a pluggable ranking function: a simple structural one by default,
  and the real rule scores once Phase 3 exists.
- Each candidate records which scene joins it crosses (continuous / cut /
  time_jump) so scoring can penalise confusing jumps. We flag, we don't drop.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable

from .parser import Episode, Event, fmt_time, load_show

# ---------------------------------------------------------------- settings

CONTINUOUS_MIN, CONTINUOUS_MAX = 20.0, 60.0
HOOK_SEG_MIN, HOOK_SEG_MAX = 10.0, 35.0
CLIFF_SEG_MIN, CLIFF_SEG_MAX = 5.0, 20.0
STITCH_MIN_GAP = 3.0      # same episode: cliff must start >= 3 s after the hook ends,
                          # otherwise it's really just a continuous clip with a hole
TOLERANCE = 0.05          # seconds; timestamps are given to 0.1 s

START_KINDS = {"DIALOGUE", "ACTION"}

# ---------------------------------------------------------------- data types


@dataclass
class Segment:
    episode: int
    start: float
    end: float
    start_beat: str              # beat ID the cut opens on
    end_beat: str                # beat ID whose end is the cut-out point
    beat_ids: list[str]          # story beats inside, in order (what the viewer follows)
    crosses: list[str] = field(default_factory=list)  # scene joins inside, e.g. ["continuous"]

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "start": fmt_time(self.start),
            "end": fmt_time(self.end),
            "start_sec": self.start,
            "end_sec": self.end,
            "duration": self.duration,
            "start_beat": self.start_beat,
            "end_beat": self.end_beat,
            "beat_ids": self.beat_ids,
            "crosses": self.crosses,
        }


@dataclass
class Candidate:
    id: str
    shape: str                   # "continuous" | "stitched"
    segments: list[Segment]
    ends_episode: bool = False   # last segment ends on the episode's final beat (an act-out)
    # Filled by scoring (features.Scorer.score):
    scores: dict = field(default_factory=dict)
    reasons: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    hook_text: str = ""
    cliffhanger_text: str = ""
    # Filled by the LLM judge (judge.judge_candidates):
    llm: dict | None = None            # validated judgement
    llm_problems: list = field(default_factory=list)
    llm_adjusted: str = ""             # "01:07.20-01:37.00 -> 01:09.00-01:37.00" if the cut moved
    # Filled by the pipeline:
    rank: int = 0                      # posting order, 1 = post first
    why: str = ""                      # one line on why it works

    @property
    def duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 2)

    @property
    def episodes(self) -> list[int]:
        return sorted({s.episode for s in self.segments})

    @property
    def crosses(self) -> list[str]:
        return [c for s in self.segments for c in s.crosses]

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "id": self.id,
            "shape": self.shape,
            "why": self.why,
            "duration": self.duration,
            "episodes": self.episodes,
            "ends_episode": self.ends_episode,
            "segments": [s.to_dict() for s in self.segments],
            "hook": self.hook_text,
            "cliffhanger": self.cliffhanger_text,
            "scores": self.scores,
            "reasons": self.reasons,
            "flags": self.flags,
            "llm": self.llm,
            "llm_problems": self.llm_problems,
            "llm_adjusted": self.llm_adjusted,
        }


# ---------------------------------------------------------------- cut-point rules


class EpisodeIndex:
    """Pre-computed lookups for one episode, so cut checks are cheap."""

    def __init__(self, ep: Episode):
        self.ep = ep
        self.beats: list[Event] = [e for e in ep.events if e.is_beat]
        self.lines: list[Event] = [e for e in ep.events if e.kind == "DIALOGUE"]
        self.scene_of = {s.number: s for s in ep.scenes}
        self.last_beat_end = max((b.end for b in self.beats), default=0.0)

        # Start points: unique start times of DIALOGUE/ACTION beats, not inside a line.
        starts: dict[float, Event] = {}
        for b in self.beats:
            if b.kind in START_KINDS and not self.inside_line(b.start) and b.start not in starts:
                starts[b.start] = b
        self.starts = sorted(starts.items())

        # End points: unique end times of story beats, not inside a line.
        # If several beats end at the same moment, name the cut after the latest-starting one.
        ends: dict[float, Event] = {}
        for b in self.beats:
            if not self.inside_line(b.end):
                if b.end not in ends or b.start >= ends[b.end].start:
                    ends[b.end] = b
        self.ends = sorted(ends.items())

    def inside_line(self, t: float) -> bool:
        """True if a character is mid-sentence at time t."""
        return any(d.start < t - TOLERANCE and d.end > t + TOLERANCE for d in self.lines)

    def make_segment(self, start: float, start_beat: Event, end: float, end_beat: Event) -> Segment:
        inside = [b.beat_id for b in self.beats if b.start >= start - TOLERANCE and b.end <= end + TOLERANCE]
        scenes = sorted({b.scene for b in self.beats if b.start >= start - TOLERANCE and b.start < end})
        crosses = [self.scene_of[n].transition for n in scenes[1:] if n in self.scene_of]
        return Segment(self.ep.number, start, end, start_beat.beat_id, end_beat.beat_id, inside, crosses)

    def segments(self, min_len: float, max_len: float) -> list[Segment]:
        """Every (start point, end point) pair whose length is within bounds."""
        out = []
        for s_time, s_beat in self.starts:
            for e_time, e_beat in self.ends:
                length = e_time - s_time
                if length < min_len - TOLERANCE:
                    continue
                if length > max_len + TOLERANCE:
                    break  # ends are sorted, later ones only get longer
                if e_beat.start < s_time - TOLERANCE:
                    continue  # the closing beat must belong to the clip
                out.append(self.make_segment(s_time, s_beat, e_time, e_beat))
        return out


# ---------------------------------------------------------------- generators


def generate_continuous(episodes: list[Episode]) -> list[Candidate]:
    out = []
    for ep in episodes:
        idx = EpisodeIndex(ep)
        for n, seg in enumerate(idx.segments(CONTINUOUS_MIN, CONTINUOUS_MAX), start=1):
            out.append(Candidate(
                id=f"C-E{ep.number:02d}-{n:04d}",
                shape="continuous",
                segments=[seg],
                ends_episode=abs(seg.end - idx.last_beat_end) <= TOLERANCE,
            ))
    return out


SegmentRanker = Callable[[Segment, EpisodeIndex], float]


def structural_hook_rank(seg: Segment, idx: EpisodeIndex) -> float:
    """Placeholder until Phase 3: prefer hooks that open a scene or open on a short, sharp line."""
    first = next(b for b in idx.beats if b.beat_id == seg.start_beat)
    score = 0.0
    scene = idx.scene_of.get(first.scene)
    if scene and scene.start is not None and abs(first.start - scene.start) < 3:
        score += 1.0
    if first.kind == "DIALOGUE" and len(first.text.split()) <= 8:
        score += 0.5
    if first.kind == "ACTION":
        score += 0.3
    return score - 0.01 * abs(seg.duration - 20)  # mild preference for ~20 s hooks


def structural_cliff_rank(seg: Segment, idx: EpisodeIndex) -> float:
    """Placeholder until Phase 3: prefer endings at the episode's act-out or a scene's end."""
    last = next(b for b in idx.beats if b.beat_id == seg.end_beat)
    score = 0.0
    if abs(seg.end - idx.last_beat_end) <= TOLERANCE:
        score += 2.0
    scene = idx.scene_of.get(last.scene)
    if scene and scene.end is not None and scene.end - seg.end < 3:
        score += 1.0
    if last.kind == "DIALOGUE" and last.text.rstrip().endswith(("?", "!", "...", "…", "—", "-")):
        score += 0.5
    return score


def generate_stitched(
    episodes: list[Episode],
    hook_rank: SegmentRanker = structural_hook_rank,
    cliff_rank: SegmentRanker = structural_cliff_rank,
    k_hooks: int = 6,
    k_cliffs: int = 6,
) -> list[Candidate]:
    """Pair the top-k hook segments of episode N with the top-k cliffhanger segments
    from later in episode N or from episode N+1."""
    indexes = {ep.number: EpisodeIndex(ep) for ep in episodes}

    def top(idx: EpisodeIndex, lo: float, hi: float, rank: SegmentRanker, k: int) -> list[Segment]:
        segs = idx.segments(lo, hi)
        segs.sort(key=lambda s: (-rank(s, idx), s.start, s.end))
        chosen: list[Segment] = []
        for s in segs:  # skip near-copies (same opening or closing beat) so k means k *different* moments
            if all(s.start_beat != c.start_beat and s.end_beat != c.end_beat for c in chosen):
                chosen.append(s)
            if len(chosen) == k:
                break
        return chosen

    hooks = {n: top(i, HOOK_SEG_MIN, HOOK_SEG_MAX, hook_rank, k_hooks) for n, i in indexes.items()}
    cliffs = {n: top(i, CLIFF_SEG_MIN, CLIFF_SEG_MAX, cliff_rank, k_cliffs) for n, i in indexes.items()}

    out = []
    for n in sorted(indexes):
        count = 0
        for hook in hooks[n]:
            for m in (n, n + 1):
                for cliff in cliffs.get(m, []):
                    if m == n and cliff.start < hook.end + STITCH_MIN_GAP - TOLERANCE:
                        continue
                    total = hook.duration + cliff.duration
                    if not (CONTINUOUS_MIN - TOLERANCE <= total <= CONTINUOUS_MAX + TOLERANCE):
                        continue
                    count += 1
                    out.append(Candidate(
                        id=f"S-E{n:02d}-{count:04d}",
                        shape="stitched",
                        segments=[hook, cliff],
                        ends_episode=abs(cliff.end - indexes[m].last_beat_end) <= TOLERANCE,
                    ))
    return out


def generate_all(episodes: list[Episode], **stitch_kwargs) -> list[Candidate]:
    return generate_continuous(episodes) + generate_stitched(episodes, **stitch_kwargs)


# ---------------------------------------------------------------- de-duplication


def overlap_seconds(a: Candidate, b: Candidate) -> float:
    """Seconds of footage two promos share (segments only overlap within the same episode)."""
    total = 0.0
    for sa in a.segments:
        for sb in b.segments:
            if sa.episode == sb.episode:
                total += max(0.0, min(sa.end, sb.end) - max(sa.start, sb.start))
    return total


def overlap_ratio(a: Candidate, b: Candidate) -> float:
    """Shared seconds / the SHORTER promo's length.

    Dividing by the shorter one means a 25 s clip sitting inside a 50 s clip counts
    as 100% overlap (it is a near-copy), not 50% as intersection-over-union would say.
    """
    shorter = min(a.duration, b.duration)
    return overlap_seconds(a, b) / shorter if shorter > 0 else 0.0


DEFAULT_QUOTA = {"continuous": 4, "stitched": 2}


def deduplicate(
    cands: list[Candidate],
    max_overlap: float = 0.35,
    quota: dict[str, int] | None = None,
    score_key: str = "rule",
) -> tuple[list[Candidate], dict[str, int]]:
    """Greedy pick: best score first; keep a promo only if it is genuinely different.

    A promo is dropped when:
      - it shares > max_overlap of its footage with an already-kept, better promo
      - it reuses the opening beat or the closing (cliffhanger) beat of a kept promo
        (stops one great ending from appearing in five stitched promos)
      - its episode already has its quota of that shape (default 4 continuous + 2 stitched;
        a promo belongs to the episode its first segment comes from)
    Returns (kept promos in score order, count of drops by reason).
    """
    quota = quota or DEFAULT_QUOTA
    ranked = sorted(cands, key=lambda c: (-c.scores.get(score_key, 0.0), c.id))
    kept: list[Candidate] = []
    kept_by_ep: dict[int, list[Candidate]] = {}
    used_open: set[tuple[int, str]] = set()
    used_close: set[tuple[int, str]] = set()
    filled: dict[tuple[int, str], int] = {}
    drops = {"quota full": 0, "reuses hook or cliffhanger beat": 0, "overlaps a better promo": 0}

    for c in ranked:
        home = c.segments[0].episode
        if filled.get((home, c.shape), 0) >= quota.get(c.shape, 0):
            drops["quota full"] += 1
            continue
        first, last = c.segments[0], c.segments[-1]
        open_key, close_key = (first.episode, first.start_beat), (last.episode, last.end_beat)
        if open_key in used_open or close_key in used_close:
            drops["reuses hook or cliffhanger beat"] += 1
            continue
        nearby = [k for ep in c.episodes for k in kept_by_ep.get(ep, [])]
        if any(overlap_ratio(c, k) > max_overlap for k in nearby):
            drops["overlaps a better promo"] += 1
            continue
        kept.append(c)
        for ep in c.episodes:
            kept_by_ep.setdefault(ep, []).append(c)
        used_open.add(open_key)
        used_close.add(close_key)
        filled[(home, c.shape)] = filled.get((home, c.shape), 0) + 1
    return kept, drops


# ---------------------------------------------------------------- stats (CLI)


def print_stats(episodes: list[Episode], out=sys.stdout) -> None:
    import random
    from collections import Counter
    from statistics import mean

    p = lambda *a: print(*a, file=out)  # noqa: E731
    cont = generate_continuous(episodes)
    stit = generate_stitched(episodes)
    beats = {b.beat_id: b for ep in episodes for b in ep.events}

    for label, cands in (("Continuous", cont), ("Stitched", stit)):
        per_ep = Counter(c.segments[0].episode for c in cands)
        counts = [per_ep.get(ep.number, 0) for ep in episodes]
        p(f"{label:<11}: {len(cands):>6} candidates | per episode min {min(counts)}, "
          f"avg {mean(counts):.0f}, max {max(counts)} | episodes with none: "
          f"{[ep.number for ep in episodes if per_ep.get(ep.number, 0) == 0] or 'none'}")
        if cands:
            buckets = Counter(min(int(c.duration // 10) * 10, 50) for c in cands)
            p("             length: " + "  ".join(f"{b}-{b + 10}s: {buckets.get(b, 0)}" for b in range(20, 60, 10)))
    cross = Counter(tuple(sorted(set(c.crosses))) for c in cont)
    p(f"Continuous crossing scene joins: {dict(cross)}")
    p(f"Continuous ending on an episode's final beat: {sum(c.ends_episode for c in cont)}")
    p(f"Stitched across two episodes: {sum(len(c.episodes) == 2 for c in stit)}")

    # Self-check: no cut lands inside a spoken line, all lengths legal.
    bad = 0
    idxs = {ep.number: EpisodeIndex(ep) for ep in episodes}
    for c in cont + stit:
        if not (CONTINUOUS_MIN - TOLERANCE <= c.duration <= CONTINUOUS_MAX + TOLERANCE):
            bad += 1
        for s in c.segments:
            if idxs[s.episode].inside_line(s.start) or idxs[s.episode].inside_line(s.end):
                bad += 1
    p(f"Self-check failures (bad length or mid-line cut): {bad}")

    rng = random.Random(7)
    for c in rng.sample(cont, 2) + rng.sample(stit, 1):
        p(f"\nExample {c.id} ({c.shape}, {c.duration}s, crosses={c.crosses or '-'})")
        for s in c.segments:
            p(f"  Ep {s.episode} {fmt_time(s.start)} -> {fmt_time(s.end)}")
            first, last = beats[s.start_beat], beats[s.end_beat]
            p(f"    opens : [{first.kind}{' ' + first.speaker if first.speaker else ''}] {first.text[:80]}")
            p(f"    closes: [{last.kind}{' ' + last.speaker if last.speaker else ''}] {last.text[:80]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m promo_finder.candidates <folder-or-zip>")
        sys.exit(2)
    print_stats(load_show(sys.argv[1]))
