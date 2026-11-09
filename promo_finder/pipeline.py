"""The whole run, start to finish: script files in -> ranked promos out.

  parse -> candidates -> rule scores -> de-duplicate -> LLM judge (top N) -> posting order

Design choices (short version):
- Each step is a plain function from the earlier files; this file only wires
  them together, times them and collects stats for run_meta.json.
- Posting order (what to post first):
    1. LLM-checked promos before unchecked ones (an unverified score shouldn't
       outrank a verified one); among checked ones, an LLM "skip" goes last.
    2. Within that, by final score + a small early-episode bonus: a promo from
       episode 3 sends a new viewer to the start of the show with fewer spoilers
       than one from episode 47.
    3. The top 10 never repeat an episode, so the first week of posts shows
       different moments of the show.
- The LLM is optional. Without it (or if it fails) everything still runs on
  rule scores, and the outputs say so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .candidates import Candidate, deduplicate, generate_continuous, generate_stitched
from .features import WEIGHTS, Scorer
from .judge import judge_candidates
from .llm import LLMClient
from .parser import Episode, load_show

EARLY_EPISODE_BONUS = 0.05   # episode 1 gets +0.05, the last episode +0.00
TOP_DISTINCT = 10            # no repeated episode within the first N posts
LOW_SCORE = 0.40             # below this, a promo is quota filler, not a recommendation


@dataclass
class RunResult:
    episodes: list[Episode]
    promos: list[Candidate]               # final, in posting order
    n_candidates: int
    drops: dict
    llm_stats: dict
    timings: dict = field(default_factory=dict)
    curves: dict = field(default_factory=dict)    # episode -> tension.TensionCurve ({} with --no-tension)
    weights: dict = field(default_factory=dict)   # rule weights actually used


def posting_order(promos: list[Candidate], n_episodes: int, max_episode: int) -> list[Candidate]:
    """Assign c.rank / c.scores['posting'] and return promos in posting order."""
    span = max(1, max_episode - 1)
    for c in promos:
        home = c.segments[0].episode
        bonus = round(EARLY_EPISODE_BONUS * (1 - (home - 1) / span), 3)
        c.scores["early_bonus"] = bonus
        c.scores["posting"] = round(c.scores.get("final", c.scores["rule"]) + bonus, 3)
    def group(c: Candidate) -> int:
        """0 = LLM-checked and not vetoed, 1 = LLM said 'skip', 2 = not checked by the LLM."""
        if c.scores.get("llm") is None:
            return 2
        return 1 if (c.llm or {}).get("verdict") == "skip" else 0

    ordered = sorted(promos, key=lambda c: (group(c), -c.scores["posting"], c.id))

    top: list[Candidate] = []
    rest: list[Candidate] = []
    used_eps: set[int] = set()
    for c in ordered:
        if len(top) < TOP_DISTINCT and not (set(c.episodes) & used_eps):
            top.append(c)
            used_eps.update(c.episodes)
        else:
            rest.append(c)
    final = top + rest
    for i, c in enumerate(final, 1):
        c.rank = i
        if c.scores.get("final", c.scores["rule"]) < LOW_SCORE and "low score (quota filler)" not in c.flags:
            c.flags.append("low score (quota filler)")
    return final


def why_line(c: Candidate) -> str:
    """One line on why it works: the LLM's sentence if we have it, else built from rule reasons."""
    if c.llm and c.llm.get("why"):
        return c.llm["why"]
    hook = (c.reasons.get("hook") or ["-"])[0]
    cliff = (c.reasons.get("cliffhanger") or ["-"])[0]
    return f"Rules: opens with {hook}; ends with {cliff}."


def run_pipeline(input_path: str, client: LLMClient | None = None, top_n: int = 60,
                 quota: dict | None = None, max_overlap: float = 0.35,
                 llm_log_path: str | None = None, say=print,
                 llm_cache_path: str | None = None, use_tension: bool = True) -> RunResult:
    t: dict[str, float] = {}

    def step(name: str, start: float) -> None:
        t[name] = round(time.monotonic() - start, 2)

    s = time.monotonic()
    say("Reading scripts...")
    episodes = load_show(input_path)
    step("parse", s)

    s = time.monotonic()
    say(f"  {len(episodes)} episodes. Generating and scoring candidates...")
    scorer = Scorer(episodes, use_tension=use_tension)
    cands = generate_continuous(episodes) + generate_stitched(
        episodes, hook_rank=scorer.hook_rank, cliff_rank=scorer.cliff_rank)
    for c in cands:
        scorer.score(c)
    step("candidates_and_rules", s)

    s = time.monotonic()
    kept, drops = deduplicate(cands, max_overlap=max_overlap, quota=quota)
    step("dedupe", s)
    say(f"  {len(cands)} candidates -> {len(kept)} promos after de-duplication.")

    s = time.monotonic()
    if client:
        say(f"Asking {client.label} to judge the top {min(top_n, len(kept))} "
            f"(about {min(top_n, len(kept)) * 5 // 60 + 1} min)...")
    llm_stats = judge_candidates(
        kept, scorer, client, top_n=top_n, log_path=llm_log_path, cache_path=llm_cache_path,
        progress=(lambda i, n, c, status: say(f"  {i}/{n} {c.id}: {status}")) if client else None)
    step("llm_judge", s)

    for c in kept:
        c.why = why_line(c)
    max_ep = max(ep.number for ep in episodes)
    promos = posting_order(kept, len(episodes), max_ep)
    return RunResult(episodes, promos, len(cands), drops, llm_stats, t, scorer.curves, scorer.weights)


__all__ = ["run_pipeline", "posting_order", "RunResult", "WEIGHTS"]
