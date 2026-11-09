"""Find promo clips in a microdrama show. One command, everything in outputs/.

  python run.py <folder-or-zip>                                # rules only
  python run.py <folder-or-zip> --provider gemini              # + LLM judge (needs GEMINI_API_KEY)
  python run.py <folder-or-zip> --provider custom --base-url https://host/v1 --model NAME --api-key-env KEY_VAR

See README.md for all options.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import time
import zipfile
from pathlib import Path

from promo_finder import judge, pipeline
from promo_finder.candidates import DEFAULT_QUOTA
from promo_finder.llm import LLMError, add_llm_args, client_from_args
from promo_finder.parser import fmt_time
from promo_finder.report import write_outputs

VERSION = "1.0"


def input_fingerprint(path: Path) -> dict:
    """Which exact files were processed (names + one combined SHA-256), for run_meta.json."""
    h, names = hashlib.sha256(), []
    if path.is_file():
        with zipfile.ZipFile(path) as zf:
            for n in sorted(x for x in zf.namelist() if x.lower().endswith(".txt")):
                names.append(n.rsplit("/", 1)[-1])
                h.update(zf.read(n))
    else:
        for p in sorted(path.rglob("*.txt")):
            names.append(p.name)
            h.update(p.read_bytes())
    return {"files": len(names), "sha256": h.hexdigest()[:16]}


def run_show(in_path: Path, out_dir: Path, client, top: int = 60, quota: dict | None = None,
             max_overlap: float = 0.35, use_cache: bool = True, say=print, started: str | None = None,
             show_name: str | None = None, use_tension: bool = True):
    """Run the whole pipeline and write every output file. Shared by the CLI (main) and the web app.

    Returns (result, meta, paths, llm_log_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    started = started or time.strftime("%Y-%m-%d %H:%M:%S")
    quota = quota or dict(DEFAULT_QUOTA)
    # Fresh raw log for this run (fall back to a timestamped name if the old one is open elsewhere).
    llm_log = out_dir / "llm_raw.jsonl"
    try:
        llm_log.unlink(missing_ok=True)
    except PermissionError:
        llm_log = out_dir / f"llm_raw_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

    result = pipeline.run_pipeline(str(in_path), client=client, top_n=top, quota=quota,
                                   max_overlap=max_overlap, llm_log_path=str(llm_log) if client else None,
                                   say=say, llm_cache_path=str(out_dir / "llm_cache.jsonl") if use_cache else None,
                                   use_tension=use_tension)

    meta = {
        "tool_version": VERSION,
        "show": show_name or (in_path.stem if in_path.is_file() else in_path.name),
        "input": str(in_path),
        "input_fingerprint": input_fingerprint(in_path),
        "started_at": started,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": len(result.episodes),
        "runtime_minutes": round(sum(e.duration for e in result.episodes) / 60, 1),
        "parse_issues": sum(len(e.issues) for e in result.episodes),
        "candidates": result.n_candidates,
        "promos": len(result.promos),
        "dedupe_drops": result.drops,
        "llm": {
            "label": client.label if client else "none (rules only)",
            "provider": client.cfg.name if client else "none",
            "model": client.cfg.model if client else None,
            "judged_top_n": top if client else 0,
            **{k: v for k, v in result.llm_stats.items()},
            "raw_log": llm_log.name if client else None,
        },
        "settings": {
            "rule_weights": result.weights,
            "tension_curve": use_tension,
            "llm_weights": judge.LLM_WEIGHTS,
            "final_score": f"{judge.RULE_SHARE} * rules + {1 - judge.RULE_SHARE} * llm (rules only if not judged)",
            "quota_per_episode": quota,
            "max_overlap": max_overlap,
            "early_episode_bonus": pipeline.EARLY_EPISODE_BONUS,
            "top_distinct_episodes": pipeline.TOP_DISTINCT,
        },
        "timings_s": result.timings,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    paths = write_outputs(result, meta, out_dir)
    return result, meta, paths, llm_log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Find the best 20-60 s promo clips in microdrama scripts.")
    ap.add_argument("input", help="folder of episode .txt scripts, or a .zip of them")
    ap.add_argument("--out", default="outputs", help="output folder (default: outputs)")
    ap.add_argument("--top", type=int, default=60, help="how many promos the LLM judges (default 60)")
    ap.add_argument("--continuous-per-episode", type=int, default=DEFAULT_QUOTA["continuous"])
    ap.add_argument("--stitched-per-episode", type=int, default=DEFAULT_QUOTA["stitched"])
    ap.add_argument("--max-overlap", type=float, default=0.35, help="max shared footage between two promos")
    ap.add_argument("--no-cache", action="store_true",
                    help="ask the LLM again even for promos it already judged (cache: <out>/llm_cache.jsonl)")
    ap.add_argument("--no-tension", action="store_true",
                    help="turn off the tension-curve score (old five-score ranking, for comparison)")
    add_llm_args(ap)
    args = ap.parse_args(argv)

    say = lambda msg: print(msg, file=sys.stderr, flush=True)  # noqa: E731
    in_path, out_dir = Path(args.input), Path(args.out)
    if not in_path.exists():
        say(f"Input not found: {in_path}")
        return 2
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        client = client_from_args(args)
    except LLMError as e:
        say(f"LLM unavailable ({e}) - running on rules only.")
        client = None
    if args.list_models:
        print("\n".join(client.list_models()) if client else "no LLM provider selected")
        return 0

    quota = {"continuous": args.continuous_per_episode, "stitched": args.stitched_per_episode}
    result, meta, paths, llm_log = run_show(in_path, out_dir, client, top=args.top, quota=quota,
                                            max_overlap=args.max_overlap, use_cache=not args.no_cache,
                                            say=say, started=started, use_tension=not args.no_tension)

    say("")
    st = result.llm_stats
    say(f"Done: {len(result.promos)} promos from {len(result.episodes)} episodes "
        f"({st.get('judged', 0)} judged by {meta['llm']['label']}"
        + (f", {st['cached']} from cache" if st.get("cached") else "")
        + (f"; {st['failed']} failed, {st['skipped']} skipped" if st.get("failed") or st.get("skipped") else "")
        + ").")
    if st.get("last_error"):
        say(f"  last LLM error: {st['last_error'][:160]}")
    for c in result.promos[:5]:
        segs = " + ".join(f"Ep{s.episode} {fmt_time(s.start)}-{fmt_time(s.end)}" for s in c.segments)
        say(f"  {c.rank}. {c.id} [{segs}] final {c.scores.get('final', c.scores['rule']):.2f} "
            f"(+{c.scores['early_bonus']:.2f} early-episode bonus = {c.scores['posting']:.2f})")
        say(f"     hook: {c.hook_text[:90]}")
    say(f"Outputs in {out_dir.resolve()}: " + ", ".join(sorted(p.name for p in paths.values()))
        + (f", {llm_log.name}" if client else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
