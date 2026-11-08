"""Parse microdrama screenplay .txt files into a flat list of timed events.

Design choices (short version):
- Plain code, no AI. Parsing is deterministic and must never invent timestamps,
  so every later step (candidates, LLM judge) only refers to events built here.
- Forgiving: header keys are matched case-insensitively ("Scene start:" and
  "SCENE START:" both work), and anything odd is recorded as an *issue* instead
  of crashing, so a new show with slightly different formatting still runs.
- Each event gets a stable beat ID like "E07-B031". Later the LLM refers to
  beats by ID only, never by timestamp.
- Input can be a folder or a .zip; zip members are read in memory (nothing is
  extracted to disk).
"""

from __future__ import annotations

import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------- data types


@dataclass
class Event:
    """One timed block from the script: a camera note, action, sound or line."""

    episode: int
    scene: int
    heading: str          # e.g. "INT. CATHEDRAL – DAY"
    kind: str             # CAMERA | ACTION | SFX | DIALOGUE
    speaker: str          # "" unless DIALOGUE
    emotion: list[str]    # e.g. ["Cold", "Commanding"]; [] if none
    delivery: str         # "O.S." / "V.O." / "" (how the line is heard)
    start: float          # seconds from episode start
    end: float
    text: str
    beat_id: str = ""     # filled after sorting, e.g. "E01-B004"
    is_beat: bool = False        # ACTION/DIALOGUE/SFX: what a viewer actually follows
    is_end_marker: bool = False  # "Cut to black." etc. - never end a promo on it

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 2)
        return d


@dataclass
class Scene:
    number: int
    heading: str
    start: float | None
    end: float | None
    # How this scene joins the previous one (see classify_transition):
    # "episode_start" | "continuous" | "cut" | "time_jump"
    transition: str = "cut"


@dataclass
class Episode:
    number: int
    source: str                       # file name it came from
    logline: str = ""
    continuity: str = ""
    scenes: list[Scene] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Episode length = last scene end, or last event end as a fallback."""
        ends = [s.end for s in self.scenes if s.end is not None]
        ends += [e.end for e in self.events]
        return max(ends) if ends else 0.0


# ---------------------------------------------------------------- regexes

# "Start: (00:11.30) | End: (00:13.10)" — parentheses optional, any case.
TIMING_RE = re.compile(
    r"^\s*start\s*:\s*\(?\s*([\d:.]+)\s*\)?\s*\|\s*end\s*:\s*\(?\s*([\d:.]+)\s*\)?\s*$",
    re.IGNORECASE,
)
# "**CAMERA:**", "**LADY ANNA (Confused, Breathless):**", colon in or out of the bold.
BLOCK_RE = re.compile(r"^\s*\*\*\s*(.+?)\s*:?\s*\*\*\s*:?\s*$")
SCENE_RE = re.compile(r"^\s*scene\s+(\d+)\s*$", re.IGNORECASE)
HEADER_RE = re.compile(
    r"^\s*(episode number|episode logline|continuity notes|scene heading|scene start|scene end)\s*:\s*(.*)$",
    re.IGNORECASE,
)
PAREN_RE = re.compile(r"\(([^)]*)\)")
SKIP_RE = re.compile(r"^\s*(---.*---|actions and dialogue\s*:)\s*$", re.IGNORECASE)

DELIVERY_TAGS = {"O.S.": "O.S.", "OS": "O.S.", "O.C.": "O.S.", "V.O.": "V.O.", "VO": "V.O."}
NON_DIALOGUE_KINDS = {
    "CAMERA": "CAMERA",
    "ACTION": "ACTION",
    "SOUND FX": "SFX",
    "SOUND": "SFX",
    "SFX": "SFX",
    "MUSIC": "SFX",
}
KIND_ORDER = {"CAMERA": 0, "ACTION": 1, "SFX": 2, "DIALOGUE": 3}
BEAT_KINDS = {"ACTION", "DIALOGUE", "SFX"}

# Lines that close an episode visually; a promo should end on the beat before them.
END_MARKER_RE = re.compile(
    r"\b(cut to black|smash cut to black|fade (out|to black)|black screen|end credits)\b", re.IGNORECASE
)
# Heading words meaning "same moment, new place" -> joining across is seamless.
CONTINUOUS_RE = re.compile(r"\b(CONTINUOUS|SAME TIME|MOMENTS LATER|SIMULTANEOUS)\b", re.IGNORECASE)
# Heading words meaning time has passed or we left the present -> a new viewer may get lost.
TIME_JUMP_RE = re.compile(
    r"\b(FLASHBACK|FLASH FORWARD|DREAM|MEMORY|MONTAGE|PAST|YEARS?|MONTHS?|WEEKS?|DAYS? LATER|"
    r"NEXT (DAY|MORNING|NIGHT|EVENING|WEEK)|(?<!MOMENTS )LATER)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------- helpers


def parse_time(value: str) -> float:
    """'01:05.60' -> 65.6 ; also accepts 'H:MM:SS.ss' and plain seconds."""
    parts = value.strip().split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return round(seconds, 3)


def fmt_time(seconds: float) -> str:
    """65.6 -> '01:05.60' (same style as the scripts)."""
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes):02d}:{sec:05.2f}"


def parse_block_label(label: str) -> tuple[str, str, list[str], str]:
    """Split a block label into (kind, speaker, emotions, delivery).

    'THE BEARDED STRANGER (O.S.) (Deep, Resonant)' ->
        ('DIALOGUE', 'THE BEARDED STRANGER', ['Deep', 'Resonant'], 'O.S.')
    """
    upper = label.strip().upper()
    if upper in NON_DIALOGUE_KINDS:
        return NON_DIALOGUE_KINDS[upper], "", [], ""

    speaker = PAREN_RE.sub("", label).strip()
    emotions: list[str] = []
    delivery = ""
    for group in PAREN_RE.findall(label):
        tag = group.strip().upper()
        if tag in DELIVERY_TAGS:
            delivery = DELIVERY_TAGS[tag]
        elif tag in {"CONT'D", "CONTD", "CONTINUED"}:
            continue
        else:
            emotions += [e.strip() for e in group.split(",") if e.strip()]
    return "DIALOGUE", speaker, emotions, delivery


def classify_transition(heading: str, is_first: bool) -> str:
    """How a scene joins the one before it, judged from its heading.

    'INT. CORRIDOR – CONTINUOUS' -> 'continuous'  (safe to run a promo across it)
    'INT. HALL – NIGHT'          -> 'cut'         (new scene; usually fine)
    'EXT. FIELD – FLASHBACK'     -> 'time_jump'   (confusing for a new viewer)
    Time-jump words win over 'continuous' ones because they are the bigger risk.
    """
    if is_first:
        return "episode_start"
    if TIME_JUMP_RE.search(heading):
        return "time_jump"
    if CONTINUOUS_RE.search(heading):
        return "continuous"
    return "cut"


def episode_number_from_name(name: str) -> int | None:
    """Fallback when the header is missing: 'episode-007-...' -> 7."""
    m = re.search(r"episode[-_ ]?0*(\d+)", name, re.IGNORECASE)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- main parser


def parse_episode(text: str, source: str = "<text>") -> Episode:
    """Parse one script's text into an Episode. Never raises on bad content."""
    ep = Episode(number=0, source=source)
    scene: Scene | None = None
    block: dict | None = None  # the block currently being read

    def close_block(line_no: int) -> None:
        nonlocal block
        if block is None:
            return
        b, block = block, None
        if b["start"] is None:
            ep.issues.append(f"line {b['line']}: block '{b['label']}' has no Start/End line - skipped")
            return
        text_ = " ".join(b["text"]).strip()
        if b["kind"] == "DIALOGUE":
            text_ = text_.strip().strip('"“”').strip()
        if not text_:
            ep.issues.append(f"line {b['line']}: block '{b['label']}' has no text")
        if b["end"] < b["start"]:
            ep.issues.append(f"line {b['line']}: end before start ({fmt_time(b['start'])} > {fmt_time(b['end'])})")
            b["end"] = b["start"]
        ep.events.append(
            Event(
                episode=ep.number,
                scene=scene.number if scene else 0,
                heading=scene.heading if scene else "",
                kind=b["kind"],
                speaker=b["speaker"],
                emotion=b["emotion"],
                delivery=b["delivery"],
                start=b["start"],
                end=b["end"],
                text=text_,
            )
        )

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()

        if not line:
            # A blank line ends a block only once it has text (keeps it tolerant
            # of a stray blank line between the timing line and the text).
            if block is not None and block["text"]:
                close_block(line_no)
            continue
        if SKIP_RE.match(line):
            close_block(line_no)
            continue

        m = HEADER_RE.match(line)
        if m:
            close_block(line_no)
            key, value = m.group(1).lower(), m.group(2).strip()
            if key == "episode number":
                digits = re.search(r"\d+", value)
                ep.number = int(digits.group()) if digits else 0
            elif key == "episode logline":
                ep.logline = value
            elif key == "continuity notes":
                ep.continuity = value
            elif scene is None:
                ep.issues.append(f"line {line_no}: '{key}' outside any SCENE")
            elif key == "scene heading":
                scene.heading = value
            elif key == "scene start":
                scene.start = _safe_time(value, ep, line_no)
            elif key == "scene end":
                scene.end = _safe_time(value, ep, line_no)
            continue

        m = SCENE_RE.match(line)
        if m:
            close_block(line_no)
            scene = Scene(number=int(m.group(1)), heading="", start=None, end=None)
            ep.scenes.append(scene)
            continue

        m = BLOCK_RE.match(line)
        if m:
            close_block(line_no)
            label = m.group(1)
            kind, speaker, emotion, delivery = parse_block_label(label)
            block = dict(label=label, line=line_no, kind=kind, speaker=speaker,
                         emotion=emotion, delivery=delivery, start=None, end=None, text=[])
            continue

        m = TIMING_RE.match(line)
        if m:
            if block is None:
                ep.issues.append(f"line {line_no}: timing line with no block above it")
            elif block["start"] is not None:
                ep.issues.append(f"line {line_no}: second timing line in one block - ignored")
            else:
                try:
                    block["start"], block["end"] = parse_time(m.group(1)), parse_time(m.group(2))
                except ValueError:
                    ep.issues.append(f"line {line_no}: bad timestamp '{line}'")
            continue

        if block is not None:
            block["text"].append(line)
        else:
            ep.issues.append(f"line {line_no}: stray text outside a block: {line[:60]!r}")

    close_block(-1)
    _finish(ep)
    return ep


def _safe_time(value: str, ep: Episode, line_no: int) -> float | None:
    try:
        return parse_time(value)
    except ValueError:
        ep.issues.append(f"line {line_no}: bad scene timestamp {value!r}")
        return None


def _finish(ep: Episode) -> None:
    """Fill the episode number, sort events in time order, assign beat IDs, sanity-check."""
    if ep.number == 0:
        guess = episode_number_from_name(ep.source)
        ep.issues.append(f"no EPISODE NUMBER header - using {guess} from file name")
        ep.number = guess or 0
    if not ep.events:
        ep.issues.append("no timed events found")

    # Camera and action blocks overlap in time, so sort by start, then a fixed kind order.
    ep.events.sort(key=lambda e: (e.start, KIND_ORDER.get(e.kind, 9), e.end))
    for i, ev in enumerate(ep.events, start=1):
        ev.episode = ep.number
        ev.beat_id = f"E{ep.number:02d}-B{i:03d}"
        # Only camera notes count: an ACTION like "...her gaze, into fade out." is still story.
        ev.is_end_marker = ev.kind == "CAMERA" and bool(END_MARKER_RE.search(ev.text))
        ev.is_beat = ev.kind in BEAT_KINDS and not ev.is_end_marker

    for i, s in enumerate(ep.scenes):
        s.transition = classify_transition(s.heading, is_first=(i == 0))

    # Events should sit inside their scene (0.5 s tolerance for rounding).
    by_num = {s.number: s for s in ep.scenes}
    for ev in ep.events:
        s = by_num.get(ev.scene)
        if s and s.start is not None and s.end is not None:
            if ev.start < s.start - 0.5 or ev.end > s.end + 0.5:
                ep.issues.append(
                    f"{ev.beat_id} ({fmt_time(ev.start)}-{fmt_time(ev.end)}) lies outside "
                    f"scene {s.number} ({fmt_time(s.start)}-{fmt_time(s.end)})"
                )


# ---------------------------------------------------------------- loading a show


def _read_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_show(path: str | Path) -> list[Episode]:
    """Load every .txt script from a folder (recursively) or a .zip file."""
    path = Path(path)
    files: list[tuple[str, bytes]] = []
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                base = name.rsplit("/", 1)[-1]
                if name.lower().endswith(".txt") and not base.startswith(("._", ".")):
                    files.append((base, zf.read(name)))
    elif path.is_dir():
        for p in sorted(path.rglob("*.txt")):
            files.append((p.name, p.read_bytes()))
    else:
        raise FileNotFoundError(f"Not a folder or .zip: {path}")

    if not files:
        raise ValueError(f"No .txt scripts found in {path}")

    episodes = [parse_episode(_read_text(data), source=name) for name, data in files]
    episodes.sort(key=lambda e: e.number)

    numbers = Counter(e.number for e in episodes)
    for ep in episodes:
        if numbers[ep.number] > 1:
            ep.issues.append(f"episode number {ep.number} appears in {numbers[ep.number]} files")
    return episodes


def all_events(episodes: list[Episode]) -> list[Event]:
    return [ev for ep in episodes for ev in ep.events]


# ---------------------------------------------------------------- stats (CLI)


def print_stats(episodes: list[Episode], out=sys.stdout) -> None:
    events = all_events(episodes)
    kinds = Counter(e.kind for e in events)
    total_sec = sum(ep.duration for ep in episodes)
    speakers = Counter(e.speaker for e in events if e.kind == "DIALOGUE")
    deliveries = Counter(e.delivery for e in events if e.delivery)
    durations = [ep.duration for ep in episodes]
    numbers = [ep.number for ep in episodes]
    missing = sorted(set(range(min(numbers), max(numbers) + 1)) - set(numbers)) if numbers else []

    p = lambda *a: print(*a, file=out)  # noqa: E731
    p(f"Episodes parsed : {len(episodes)}  (numbers {min(numbers)}-{max(numbers)}"
      f"{', missing ' + str(missing) if missing else ', none missing'})")
    p(f"Scenes          : {sum(len(ep.scenes) for ep in episodes)}")
    p(f"Total runtime   : {total_sec / 60:.1f} min  "
      f"(episode length min {fmt_time(min(durations))}, max {fmt_time(max(durations))}, "
      f"avg {fmt_time(total_sec / len(episodes))})")
    p(f"Events          : {len(events)}")
    for kind in ("CAMERA", "ACTION", "DIALOGUE", "SFX"):
        p(f"  {kind:<9}: {kinds.get(kind, 0)}")
    p(f"Story beats     : {sum(e.is_beat for e in events)} (ACTION+DIALOGUE+SFX, excluding end markers)")
    p(f"End markers     : {sum(e.is_end_marker for e in events)} (e.g. 'Cut to black.')")
    transitions = Counter(s.transition for ep in episodes for s in ep.scenes)
    p(f"Scene joins     : {dict(transitions)}")
    for ep in episodes:
        for s in ep.scenes:
            if s.transition == "time_jump":
                p(f"  time jump: Ep {ep.number} scene {s.number}: {s.heading}")
    p(f"Dialogue lines heard off-screen: {dict(deliveries)}")
    p(f"Distinct speakers: {len(speakers)}; top 8: "
      + ", ".join(f"{s} ({n})" for s, n in speakers.most_common(8)))

    issues = [(ep.number, msg) for ep in episodes for msg in ep.issues]
    p(f"Parse issues    : {len(issues)}")
    for num, msg in issues[:40]:
        p(f"  Ep {num:>3}: {msg}")
    if len(issues) > 40:
        p(f"  ... and {len(issues) - 40} more")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m promo_finder.parser <folder-or-zip>")
        sys.exit(2)
    print_stats(load_show(sys.argv[1]))
