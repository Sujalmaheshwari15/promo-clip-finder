"""Tension curve: how dramatically tense each moment of an episode is, 0-1 over time.

Why: a promo should start while tension is climbing and cut at a peak, just
before it drops (the drop is the resolution we want the viewer to pay for).
The other rule scores look at single moments; this one looks at the SHAPE.

Design choices (short version):
- No new lexicons. Each event's raw value comes from features.tag_beat(), the
  same tagger the rule scores use (violence/threat/reveal words, writers'
  emotion tags like "(Furious)", shock reactions, sharp sound effects, "!").
  On top of that, three small extras tag_beat doesn't cover:
    * calm beats ("(Gentle)", "smiles", "embraces") go BELOW the neutral level,
      so a soft ending visibly pulls the curve down;
    * questions and very short shouted lines ("Stop!", "Where is she?") add a bit;
    * music cues are read for mood: ominous/tense music raises tension, tender
      music lowers it. tag_beat ignores music on purpose (for cliffhanger scoring),
      but for "how tense does it FEEL" music matters. The word "cliffhanger" in a
      music note is still ignored - same rule as features.py.
    * quiet menace ("(Steely, Resolute)", "without flinching") scores a medium
      0.5. The loud lexicons miss it, and it's how many episodes end. This list
      lives HERE only, so the old rule scores are untouched.
- Neutral is 0.2, not 0, so "calm" (0.0) and "nothing special" (0.2) differ.
- Events are laid on a 0.5 s grid. Where events overlap (a camera note over an
  action over a line), the grid point takes the MAX, not the sum: tension is
  "the hottest thing on screen right now". Summing would make busy, heavily
  annotated moments look tense just because the writers wrote more blocks.
- Held shots: when only a camera note is running (no line, action or sound),
  the tension of the last story beat carries forward and fades towards
  neutral with a half-life of 3.5 s. A silent close-up after "your husband
  hangs" is still tense; without this, the curve fell to neutral there and
  made episode endings look like drops. A camera note with its own signal
  ("her face drains of colour") can still push the value up.
- Smoothing: centred moving average, ~4 s window. One shouted word shouldn't
  be a "peak"; a 4 s window is about one exchange of lines.
- Normalised to 0-1 per episode by dividing by the peak. So 1.0 means "the most tense moment
  of THIS episode", not of the show. Good for shape, useless for comparing
  episodes - which we never do with it.
- Peaks: local maxima above 0.6, at least 3 s apart (smoothing still leaves
  small ripples; without a minimum gap one bump becomes three "peaks").
  Drops: after a peak, the curve falls by more than 0.25 within 6 s.

Standard library only.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field

from .features import CALM_EMOTIONS, HOT_EMOTIONS, MUSIC_CUE, WARM_EMOTIONS, lex, tag_beat
from .parser import Episode, Event, fmt_time

# ---------------------------------------------------------------- settings (tune here)

STEP = 0.5             # grid resolution, seconds
SMOOTH_WINDOW = 4.0    # moving-average window, seconds
PEAK_MIN = 0.6         # a peak must be at least this high (after normalising)
PEAK_MIN_GAP = 3.0     # two peaks closer than this: keep the higher one
DROP_SIZE = 0.25       # a "drop" is a fall of more than this...
DROP_WITHIN = 6.0      # ...within this many seconds after a peak

NEUTRAL = 0.2          # value of a beat with no tension signal at all
CALM_PENALTY = 0.2     # calm beats sit this far below where they'd otherwise be
QUESTION_BONUS = 0.1
SHORT_LINE_BONUS = 0.05    # extra for a <=4-word line ending in ! or ?
QUIET_VALUE = 0.5          # quiet menace: above neutral (0.2), below loud tags (warm 0.6, hot 0.92)
DECAY_HALF_LIFE = 3.5      # seconds; None = no carry-over during camera-only stretches
NORMALISE = "peak"         # "peak" (divide by max) or "minmax"; see normalise()

# Quiet tension: steel, resolve, menace without shouting. Matched against the writers'
# emotion tags and ACTION text (not dialogue text: "we must determine..." isn't menace).
QUIET_TENSION = lex("steely", "steel", "resolute*", "resolve", "determin*", "menac*", "icy", "ice-cold",
                    "icy calm", "measured threat", "deadly quiet", "deadly", "dangerously quiet",
                    "quietly furious", "cold fury", "unflinching", "without flinching", "unyielding",
                    "implacable", "grim", "grimly", "hardening", "hardened")

# Music mood words. Only what isn't already in the emotion sets (which also cover
# "ominous", "sinister", "chilling", "tender", "gentle", "warm", ...).
TENSE_MUSIC = lex("tense", "tension*", "dramatic*", "sting*", "crescendo", "boom*", "abrupt*", "sudden*",
                  "thunderous", "pounding", "menac*", "dread*", "suspense*", "unresolved", "foreboding")
CALM_MUSIC = lex("triumphant*", "hopeful", "romantic", "fading out", "fades out", "soothing", "lullaby")
WORD_RE = re.compile(r"[a-z]+")


# ---------------------------------------------------------------- per-event value


def music_value(text: str) -> tuple[float, str]:
    """Mood of a music cue -> (value, why). The word 'cliffhanger' is ignored on purpose."""
    words = set(WORD_RE.findall(text.lower().replace("cliffhanger", "")))
    tense = bool(TENSE_MUSIC.search(text) or words & (HOT_EMOTIONS | WARM_EMOTIONS))
    calm = bool(CALM_MUSIC.search(text) or words & CALM_EMOTIONS)
    if tense and not calm:
        return 0.6, "tense music"
    if calm and not tense:
        return 0.0, "calm music"
    return NEUTRAL, "neutral/mixed music"


def quiet_match(ev: Event) -> str:
    """The quiet-menace word this event carries ('' if none): emotion tags, or ACTION text."""
    m = QUIET_TENSION.search(", ".join(ev.emotion))
    if not m and ev.kind == "ACTION":
        m = QUIET_TENSION.search(ev.text)
    return m.group(0) if m else ""


def event_tension(ev: Event, quiet: bool = True) -> tuple[float, str]:
    """Raw tension of one event, 0-1, plus a short reason (for the text printout)."""
    tags = tag_beat(ev)
    if tags.music:
        return music_value(ev.text)
    value = NEUTRAL + (1 - NEUTRAL) * tags.heat
    why = ", ".join(f"{k}:{v}" for k, v in list(tags.cats.items())[:2])
    if ev.kind == "DIALOGUE":
        text = ev.text.rstrip()
        if tags.question:
            value += QUESTION_BONUS
        if len(text.split()) <= 4 and text.endswith(("!", "?")):
            value += SHORT_LINE_BONUS
    if tags.calm:
        value -= CALM_PENALTY
        why = (why + "; " if why else "") + f"calm:{tags.calm_why}"
    word = quiet_match(ev) if quiet else ""
    if word and value < QUIET_VALUE:
        # Steel beats softness: "(Calm, Steely)" is menace, not comfort.
        value = QUIET_VALUE
        why = (why + "; " if why else "") + f"quiet:{word}"
    return round(max(0.0, min(1.0, value)), 3), why


# ---------------------------------------------------------------- the curve


@dataclass
class TensionCurve:
    episode: int
    points: list[tuple[float, float]]                 # (time_sec, 0-1), every STEP seconds
    peaks: list[dict] = field(default_factory=list)   # {"time", "value"}
    drops: list[dict] = field(default_factory=list)   # {"peak_time", "time", "from", "to"}
    flat: bool = False                                # no variation at all (curve is all 0)

    def value_at(self, t: float) -> float:
        """Value at time t (nearest grid point, clamped to the episode)."""
        if not self.points:
            return 0.0
        i = min(len(self.points) - 1, max(0, int(round(t / STEP))))
        return self.points[i][1]

    def window(self, start: float, end: float) -> list[float]:
        """Values for grid points with start <= time <= end."""
        return [v for t, v in self.points if start - 1e-6 <= t <= end + 1e-6]

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "step": STEP,
            "points": [[t, v] for t, v in self.points],
            "peaks": self.peaks,
            "drops": self.drops,
            "flat": self.flat,
        }


def raw_grid(episode: Episode, step: float = STEP, half_life: float | None = DECAY_HALF_LIFE,
             quiet: bool = True) -> list[tuple[float, float]]:
    """Raw (unsmoothed) tension per grid cell [t, t+step).

    - Cell with a story event (line/action/sound): max of everything overlapping it.
    - Camera-only cell: the previous value fading towards NEUTRAL (half-life `half_life`),
      raised by the camera note's own value if that note carries a signal.
      half_life=None turns the carry-over off (camera note's own value only).
    - No event at all: NEUTRAL (a gap in the script isn't calm, it's unknown).
    """
    n = int(episode.duration / step) + 1
    story: list[float | None] = [None] * n
    camera: list[float | None] = [None] * n
    for ev in episode.events:
        value, _ = event_tension(ev, quiet)
        grid = camera if ev.kind == "CAMERA" else story
        end = max(ev.end, ev.start + 0.01)            # zero-length events still cover one cell
        first, last = int(ev.start / step), min(n - 1, int((end - 1e-9) / step))
        for i in range(first, last + 1):
            if grid[i] is None or value > grid[i]:
                grid[i] = value

    fade = 0.5 ** (step / half_life) if half_life else None
    out, last_value = [], NEUTRAL
    for i in range(n):
        s, c = story[i], camera[i]
        if s is not None:
            v = max(s, c) if c is not None else s
        elif c is not None:
            if fade is None:
                v = c
            else:
                carried = NEUTRAL + (last_value - NEUTRAL) * fade
                v = max(carried, c) if c > NEUTRAL else carried
        else:
            v = NEUTRAL
        last_value = v
        out.append((round(i * step, 2), round(v, 4)))
    return out


def smooth(values: list[float], window_sec: float = SMOOTH_WINDOW, step: float = STEP) -> list[float]:
    """Centred moving average. Near the edges it averages over the points that exist."""
    half = max(0, int(round(window_sec / step / 2)))
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def normalise(values: list[float], mode: str | None = None) -> tuple[list[float], bool]:
    """Scale to 0-1 per episode. A perfectly flat episode returns all zeros and flat=True.

    "peak"   : divide by the episode's maximum. The peak is 1.0, but a tense episode's
               quietest moment is NOT forced to 0 (raw values are already on a 0-1 scale).
    "minmax" : (v - min) / (max - min). Stretches the quietest moment to 0 - which turned
               Ep 12's steely ending (raw 0.50 in an episode whose floor is 0.44) into 0.10.
    """
    mode = mode or NORMALISE
    lo, hi = min(values, default=0.0), max(values, default=0.0)
    if hi - lo < 1e-9:
        return [0.0] * len(values), True
    if mode == "minmax":
        return [round((v - lo) / (hi - lo), 3) for v in values], False
    return [round(v / hi, 3) for v in values], False


def find_peaks(points: list[tuple[float, float]], min_value: float = PEAK_MIN,
               min_gap: float = PEAK_MIN_GAP) -> list[dict]:
    """Local maxima above min_value; of two peaks closer than min_gap, keep the higher."""
    vals = [v for _, v in points]
    cands = []
    for i, v in enumerate(vals):
        left = vals[i - 1] if i > 0 else -1.0
        right = vals[i + 1] if i + 1 < len(vals) else -1.0
        # '>' on the left, '>=' on the right: a flat-topped peak counts once (at its first point).
        if v >= min_value and v > left and v >= right:
            cands.append((points[i][0], v))
    kept: list[tuple[float, float]] = []
    for t, v in sorted(cands, key=lambda p: -p[1]):          # highest first
        if all(abs(t - kt) >= min_gap for kt, _ in kept):
            kept.append((t, v))
    return [{"time": t, "value": v} for t, v in sorted(kept)]


def find_drops(points: list[tuple[float, float]], peaks: list[dict], size: float = DROP_SIZE,
               within: float = DROP_WITHIN) -> list[dict]:
    """For each peak: the first point within `within` s after it that is more than `size` lower."""
    drops = []
    for p in peaks:
        for t, v in points:
            if p["time"] < t <= p["time"] + within + 1e-6 and p["value"] - v > size:
                drops.append({"peak_time": p["time"], "time": t, "from": p["value"], "to": v})
                break
    return drops


def analyse(episode: Episode, window_sec: float = SMOOTH_WINDOW, half_life: float | None = DECAY_HALF_LIFE,
            quiet: bool = True, norm: str | None = None) -> TensionCurve:
    """Full curve + peaks + drops for one episode."""
    grid = raw_grid(episode, half_life=half_life, quiet=quiet)
    if not grid:
        return TensionCurve(episode.number, [], flat=True)
    values, flat = normalise(smooth([v for _, v in grid], window_sec), norm)
    points = [(t, v) for (t, _), v in zip(grid, values)]
    peaks = [] if flat else find_peaks(points)
    return TensionCurve(episode.number, points, peaks, find_drops(points, peaks), flat)


def tension_curve(episode: Episode, window_sec: float = SMOOTH_WINDOW) -> list[tuple[float, float]]:
    """Just the points: [(time_sec, value 0-1), ...] every 0.5 s."""
    return analyse(episode, window_sec).points


# ---------------------------------------------------------------- promo shape score

RISE_WINDOW = 5.0      # "the start" = first 5 s of the first segment
RISE_FULL = 0.3        # ending this much higher than the start = full rise marks (this much lower = 0)
PEAK_NEAR = 2.0        # a peak within 2 s of the closing moment = full marks...
PEAK_FAR = 6.0         # ...fading to 0 at 6 s away
RESOLVED_FLAG = "cuts after the peak — tension already resolved"


@dataclass
class ShapeResult:
    value: float | None  # None = not scored (episode-end cut, flat curve)
    reasons: list[str]
    parts: dict          # rise / peak / withheld, each 0-1
    resolved: bool       # True -> RESOLVED_FLAG


def _clamp(x: float) -> float:
    return round(max(0.0, min(1.0, x)), 3)


def shape_score(first: TensionCurve, first_start: float, last: TensionCurve, last_start: float,
                anchor_start: float, cut: float, ends_episode: bool) -> ShapeResult:
    """Score a promo's tension SHAPE, 0-1 = mean of three parts.

    first/first_start : curve + start time of the FIRST segment (rise is judged there)
    last/last_start   : curve + start time of the LAST segment (peak and drop are judged there)
    anchor_start      : start of the closing moment = the last spoken line if it ended within 6 s
                        of the cut (same rule as the cliffhanger score), else the closing beat
    cut               : where the promo cuts out

    rise     : tension at the closing moment vs the mean of the first 5 s.
    peak     : distance from the closing moment [anchor_start, cut] to the nearest peak.
    withheld : does the curve fall right after the cut?
    A cut at the episode's end is not scored (value None, no flag): the cliffhanger score owns it.
    """
    if first.flat or last.flat:
        return ShapeResult(None, ["flat tension curve - not scored"],
                           {"rise": 0.5, "peak": 0.5, "withheld": 0.5}, False)
    start_vals = first.window(first_start, first_start + RISE_WINDOW) or [first.value_at(first_start)]
    start_mean = sum(start_vals) / len(start_vals)
    closing = last.window(anchor_start, cut) or [last.value_at(cut)]
    ref = max(closing)

    rise = _clamp(0.5 + (ref - start_mean) / (2 * RISE_FULL))

    def dist(t: float) -> float:
        return 0.0 if anchor_start <= t <= cut else min(abs(t - anchor_start), abs(t - cut))
    near = min(last.peaks, key=lambda p: dist(p["time"]), default=None)
    d = dist(near["time"]) if near else 99.0
    peak = _clamp(1 - (d - PEAK_NEAR) / (PEAK_FAR - PEAK_NEAR)) if d > PEAK_NEAR else 1.0
    # Resolved: a peak inside this segment, before the closing moment, that has already
    # fallen by more than DROP_SIZE by the time we close - the release is IN the promo.
    earlier = [p for p in last.peaks if last_start <= p["time"] < anchor_start - PEAK_NEAR
               and p["value"] - ref > DROP_SIZE]
    resolved = peak < 0.5 and bool(earlier)
    if resolved:
        peak = 0.0

    v_cut = last.value_at(cut)
    if ends_episode:
        withheld, after_fall = 0.5, None
    else:
        after = last.window(cut + STEP, cut + DROP_WITHIN)
        after_fall = (v_cut - min(after)) if after else 0.0
        withheld = _clamp(after_fall / DROP_SIZE)

    reasons = []
    if peak >= 0.75 and withheld >= 0.75:
        reasons.append("cuts at a tension peak; resolution withheld")
    elif resolved:
        top = max(earlier, key=lambda p: p["value"])
        reasons.append(f"{RESOLVED_FLAG} (peak {top['value']:.2f} at {fmt_time(top['time'])}, "
                       f"{ref:.2f} at the close)")
    elif peak >= 0.75:
        reasons.append("cuts at a tension peak")
    elif near:
        reasons.append(f"nearest tension peak is {d:.1f}s from the closing moment")
    else:
        reasons.append("no tension peak in this episode")
    reasons.append(f"tension {'rises' if ref > start_mean + 0.05 else 'falls' if ref < start_mean - 0.05 else 'is level'}"
                   f" into the cut ({start_mean:.2f} -> {ref:.2f})")
    if after_fall is None:
        reasons.append("episode ends at the cut - drop not judged (0.5)")
    elif withheld >= 0.75 and not reasons[0].endswith("withheld"):
        reasons.append(f"resolution withheld (falls {after_fall:.2f} right after the cut)")
    elif withheld < 0.25:
        reasons.append("no drop after the cut")
    parts = {"rise": rise, "peak": peak, "withheld": withheld}
    if ends_episode:
        # Episode ends at the cut: the cliffhanger score already judges this moment, and the curve
        # misreads quiet act-outs (a steely last line, a two-word shock after a held shot).
        # Measured on the full pack: 15 of 19 such promos dropped, incl. good ones.
        # value None = not scored: the rule score uses the old five weights for this promo,
        # so tension neither helps nor hurts it (a fixed 0.5 still cost endings ~0.03).
        return ShapeResult(None, ["cuts at the episode's end - not scored, left to the cliffhanger score"]
                           + reasons, parts, False)
    value = round((rise + peak + withheld) / 3, 3)
    return ShapeResult(value, reasons, parts, resolved)


def save_curves(curves: dict[int, TensionCurve], path) -> None:
    """outputs/tension_curves.json: {"<episode>": {points, peaks, drops, ...}}."""
    data = {str(n): c.to_dict() for n, c in sorted(curves.items())}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))


# ---------------------------------------------------------------- CLI: print one episode as text


def print_curve(episode: Episode, every: float = 1.0, out=sys.stdout) -> None:
    """One row per `every` seconds: time, value, bar, PEAK/DROP marker, hottest event starting there."""
    curve = analyse(episode)
    peak_t = {p["time"] for p in curve.peaks}
    drop_t = {d["time"]: d for d in curve.drops}
    p = lambda *a: print(*a, file=out)  # noqa: E731
    p(f"Episode {episode.number}: {fmt_time(episode.duration)}, {len(curve.points)} points, "
      f"{len(curve.peaks)} peaks, {len(curve.drops)} drops{' (FLAT)' if curve.flat else ''}")
    p(f"settings: step {STEP}s, smoothing {SMOOTH_WINDOW}s, peak > {PEAK_MIN}, "
      f"drop > {DROP_SIZE} within {DROP_WITHIN}s\n")
    stride = max(1, int(round(every / STEP)))
    for i in range(0, len(curve.points), stride):
        t0, v = curve.points[i]
        span = [pt for pt, _ in curve.points[i:i + stride]]
        marks = [f"PEAK {curve.value_at(t):.2f}@{t:g}" for t in span if t in peak_t]
        marks += [f"DROP (peak {drop_t[t]['peak_time']:g}s {drop_t[t]['from']:.2f} -> {drop_t[t]['to']:.2f})"
                  for t in span if t in drop_t]
        starting = [e for e in episode.events if t0 <= e.start < t0 + every and e.kind != "CAMERA"]
        note = ""
        if starting:
            best = max(starting, key=lambda e: event_tension(e)[0])
            raw, why = event_tension(best)
            who = best.speaker.title() if best.kind == "DIALOGUE" else best.kind.lower()
            note = f"raw {raw:.2f} {who}: {best.text[:55]}" + (f"  [{why}]" if why else "")
        p(f"{fmt_time(t0)}  {v:4.2f} {'#' * int(round(v * 20)):<20} {' '.join(marks):<34} {note}")


if __name__ == "__main__":
    import argparse

    from .parser import load_show

    ap = argparse.ArgumentParser(description="Print an episode's tension curve as text.")
    ap.add_argument("input", help="folder or .zip of scripts")
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--every", type=float, default=1.0, help="seconds per printed row (default 1)")
    a = ap.parse_args()
    eps = {e.number: e for e in load_show(a.input)}
    if a.episode not in eps:
        sys.exit(f"episode {a.episode} not found")
    print_curve(eps[a.episode], a.every)
