"""Rule scoring: cheap, transparent scores for every candidate.

Design choices (short version):
- Two levels. First every beat is tagged once (keyword lexicons + the emotion
  notes the script writers already give, like "(Gasping, Devastated)").
  Then each candidate is scored from the tags of the beats it contains.
- Five scores, each 0-1, each with human-readable reasons:
    hook        - only events starting in the first 3 s (that's all a scroller sees)
    cliffhanger - the last ~3 s; open questions/threats/cut-off lines score,
                  calm/romantic/resolved endings are penalised
    standalone  - starts at 1.0 and loses points for warning signs
                  (opens on an answer, mid-conversation, time jump, one voice...)
    intensity   - how heated the whole clip is
    length      - 28-45 s is best
  rule_score = weighted sum (WEIGHTS below, one place to tune).
- The scripts sometimes say "cliffhanger" in a music cue laid over a tender
  beat. That word is ignored: music cues (swell/score/drone/strings) never add
  cliffhanger points; only what happens on screen counts.
- Lexicons are deliberately generic (drama, not this one show) so a new show
  still scores sensibly. Known gap: subtle lines with no "loud" words
  ("He's not dead.") score low - that's what the LLM judge is for.
"""

from __future__ import annotations

import re
import string
import sys
from dataclasses import dataclass, field

from .candidates import Candidate, EpisodeIndex, Segment
from .parser import Episode, Event, fmt_time

# ---------------------------------------------------------------- weights (tune here)

WEIGHTS = {"hook": 0.30, "cliffhanger": 0.30, "standalone": 0.20, "intensity": 0.12, "length": 0.08}
HOOK_WINDOW = 3.0        # seconds a scrolling viewer gives you
CLIFF_WINDOW = 3.0       # the final moment we judge the ending on
LAST_LINE_WINDOW = 6.0   # the last spoken line still "is" the ending if it ended this close to the cut
LENGTH_BEST = (28.0, 45.0)


# ---------------------------------------------------------------- lexicons


def lex(*words: str) -> re.Pattern:
    """Build a word-boundary regex. 'slap*' = any word starting with slap; spaces match any whitespace."""
    parts = []
    for w in words:
        stem = w.endswith("*")
        body = re.escape(w.rstrip("*")).replace(r"\ ", r"\s+")
        parts.append(body + (r"\w*" if stem else ""))
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.IGNORECASE)


# (name, weight, pattern). Weight = how strongly it grabs attention.
TEXT_LEXICONS = [
    ("violence", 1.0, lex("slap*", "punch*", "strike*", "struck", "hit", "hits", "chok*", "strangl*",
                          "stab*", "knife", "dagger", "sword", "blade", "pistol", "gun*", "whip*", "lash*",
                          "kick*", "attack*", "beaten", "bleed*", "blood*", "wound*", "slash*",
                          "riding crop", "riding-crop")),
    # Physical but not necessarily violent ("grabs her arm to keep her moving") - weaker signal.
    ("physical", 0.6, lex("grab*", "drag*", "shove*", "hurl*", "wrestl*", "seiz*", "pins her", "pins him",
                          "violently", "fiercely", "forcibly", "furious*", "rage*", "roar*", "snarl*",
                          "lunge*", "charges at", "charges toward", "charging", "burst*", "storm* out",
                          "storm* in")),
    ("reveal", 0.9, lex("truth", "secret*", "reveal*", "unmask*", "identity", "who you really are",
                        "you are the", "heir", "pregnant", "alive", "murder*", "killed", "lied", "liar",
                        "forged", "forger*", "proof", "evidence", "witness*", "confess*", "real name",
                        "all along", "imposter", "impostor", "it was you", "recogni*")),
    # "ruin" only as a threat ("I will ruin you"), not as an adjective describing a place or event.
    ("threat", 0.9, lex("you will pay", "you'll pay", "regret", "or else", "destroy*", "ruin you", "ruin her",
                        "ruin him", "ruin them", "ruin your", "ruin everything", "you are ruined",
                        "you're ruined", "will be ruined", "kill you",
                        "kill her", "kill him", "mark my words", "warn*", "how dare", "get out", "stay away",
                        "never forgive", "watch your", "make you suffer", "you will suffer", "hang*",
                        "execut*", "arrest*", "seize her", "seize him", "take her", "take him")),
    ("betrayal", 0.8, lex("divorce*", "annul*", "abandon*", "betray*", "humiliat*", "disgrace*", "shame*",
                          "cast out", "disown*", "jilt*", "cheat*", "affair", "mistress", "worthless",
                          "pathetic", "beggar", "marrying me", "changed my mind", "never loved")),
    ("danger", 0.7, lex("poison*", "arsenic", "fire", "flame*", "inferno", "burning", "collaps*", "faint*",
                        "chains", "shackle*", "prison", "dungeon", "trap*", "dying", "dead", "death",
                        "corpse", "scream*", "shriek*")),
]
# Visible shock reactions (ACTION/SFX/CAMERA): someone just learned something.
REACTION = lex("gasp*", "freez*", "stunned", "horrified", "jaw drops", "jaw agape", "eyes widen*",
               "drops the", "staggers", "recoil*", "goes pale", "drained of", "terror", "shock*")
SHARP_SFX = lex("slam*", "crash*", "shatter*", "smash*", "bang*", "crack*", "thud*", "gunshot", "clang*",
                "thunder*", "slap*", "scream*", "gasp*")
MUSIC_CUE = lex("music", "score", "swell*", "drone", "strings", "orchestral", "sting", "melody", "cliffhanger")
CALM_TEXT = lex("thank you", "i love you", "forgive you", "it's over", "it is over", "you're safe",
                "you are safe", "rest now", "smil*", "laugh*", "embrac*", "kiss*", "hug*", "caress*",
                "peaceful*", "gently", "tender*", "warmly", "good night", "sleep*", "content", "relie*",
                # Triumphant exits are resolutions too (found via LLM disagreement on an episode
                # that ends with the couple leaving together, not on a cliffhanger).
                "hand-in-hand", "hand in hand", "into the sunlight", "triumphant*", "walk away together")
# Camera/edit directions that contain reaction words but aren't a character reacting.
NOT_A_REACTION = re.compile(r"\b(scene|frame|image|picture|shot) freez\w*|freeze[- ]frame", re.IGNORECASE)
# A request still waiting for an answer is an open question even without a "?":
# e.g. a proposal that ends the episode before she answers.
PENDING_REQUEST = lex("accept my hand", "marry me", "take my hand", "say yes", "choose", "come with me",
                      "run away with me", "will you", "would you", "promise me", "tell me", "answer me",
                      "swear it", "decide")
ANSWER_OPENERS = lex("yes", "no", "because", "that's why", "exactly", "of course", "indeed", "fine",
                     "very well", "and", "but", "so", "then", "well")
PRONOUN_OPENERS = lex("he", "she", "they", "it", "that", "this", "him", "her", "them", "those")

HOT_EMOTIONS = {  # weight 1.0
    "screaming", "shouting", "hysterical", "terrified", "horrified", "shocked", "stunned", "gasping",
    "furious", "outraged", "enraged", "seething", "panicked", "frantic", "threatening", "menacing",
    "betrayed", "crying out", "dangerous", "livid",
}
WARM_EMOTIONS = {  # weight 0.6 - conflict, cruelty, distress
    "desperate", "sobbing", "weeping", "heartbroken", "venomous", "cruel", "sinister", "chilling",
    "malicious", "vindictive", "gloating", "mocking", "taunting", "spiteful", "derisive", "contemptuous",
    "scornful", "accusatory", "accusing", "alarmed", "incredulous", "defiant", "fierce", "piercing",
    "demanding", "urgent", "pleading", "trembling", "choked", "broken", "despairing", "smug", "arrogant",
    "condescending", "callous", "harsh", "bitter", "ominous", "warning", "indignant", "devastated",
    "tearful", "shaken", "anguished", "sharp", "cold", "scheming",
}
CALM_EMOTIONS = {
    "tender", "gentle", "soft", "warm", "loving", "peaceful", "reassuring", "affectionate", "relieved",
    "content", "serene", "amused", "playful", "happy", "joyful", "smiling", "safe", "grateful",
    "reflective", "sincere", "calm", "reverent", "deeply moved", "soothing", "fond", "teasing",
}


# ---------------------------------------------------------------- beat tagging


@dataclass
class BeatTags:
    heat: float = 0.0                         # 0-1, how attention-grabbing this beat is
    cats: dict[str, str] = field(default_factory=dict)  # category -> matched word
    calm: bool = False
    calm_why: str = ""
    question: bool = False
    exclaim: bool = False
    cut_off: bool = False
    reaction: bool = False
    music: bool = False


def tag_beat(ev: Event) -> BeatTags:
    t = BeatTags()
    text = ev.text
    if ev.kind == "SFX" and MUSIC_CUE.search(text):
        # A music cue tells us about mood, not about story. Ignore its words entirely
        # (this is where scripts write things like "swells to a cliffhanger peak").
        t.music = True
        return t

    weight_scale = 0.5 if ev.kind == "CAMERA" else 1.0  # camera notes describe visuals, count half
    best = 0.0
    for name, w, pat in TEXT_LEXICONS:
        m = pat.search(text)
        if m:
            t.cats[name] = m.group(0)
            best = max(best, w * weight_scale)
    reaction_text = NOT_A_REACTION.sub("", text)  # "the scene freezes" is an edit, not a gasp
    if ev.kind in ("ACTION", "SFX", "CAMERA") and REACTION.search(reaction_text):
        t.reaction = True
        t.cats.setdefault("reaction", REACTION.search(reaction_text).group(0))
        best = max(best, 0.7 * weight_scale)
    if ev.kind == "SFX" and SHARP_SFX.search(text):
        t.cats.setdefault("sharp_sfx", SHARP_SFX.search(text).group(0))
        best = max(best, 0.6)

    emo = {e.lower() for e in ev.emotion}
    if emo & HOT_EMOTIONS:
        best = max(best, 0.9)
        t.cats.setdefault("emotion", sorted(emo & HOT_EMOTIONS)[0])
    elif emo & WARM_EMOTIONS:
        best = max(best, 0.5)
        t.cats.setdefault("emotion", sorted(emo & WARM_EMOTIONS)[0])

    if ev.kind == "DIALOGUE":
        stripped = text.rstrip()
        t.question = stripped.endswith("?") or stripped.endswith("?!")
        t.exclaim = "!" in stripped[-3:]
        t.cut_off = stripped.endswith(("...", "…", "—", "--", "-"))
        if t.exclaim:
            best += 0.1
    extra = max(0, len(t.cats) - 1)
    t.heat = round(min(1.0, best + 0.1 * extra), 3)

    calm_emo = emo & CALM_EMOTIONS
    calm_txt = CALM_TEXT.search(text) if ev.kind != "CAMERA" else None
    if (calm_emo or calm_txt) and t.heat < 0.6 and not (emo & HOT_EMOTIONS):
        t.calm = True
        t.calm_why = sorted(calm_emo)[0] if calm_emo else calm_txt.group(0)
    return t


# ---------------------------------------------------------------- candidate scoring


@dataclass
class ScoreResult:
    value: float
    reasons: list[str]


def continues_sentence(text: str) -> bool:
    """'...a secret medicine in her hand.' or 'and then he left' = the tail of an earlier line."""
    t = text.strip().lstrip('"“')
    return t.startswith(("...", "…", "—", "-")) or (t[:1].islower())


def clamp(x: float) -> float:
    return round(max(0.0, min(1.0, x)), 3)


def norm_speaker(name: str) -> str:
    """'LADY ANNA' and 'ANNA' are the same person for 'how many voices' checks."""
    parts = name.split()
    if len(parts) > 1 and parts[0] in {"LADY", "LORD", "SIR", "THE", "MR.", "MRS.", "MISS", "MR", "MRS"}:
        return " ".join(parts[1:])
    return name


class Scorer:
    """Holds tags for every event and scores segments / candidates."""

    def __init__(self, episodes: list[Episode]):
        self.indexes = {ep.number: EpisodeIndex(ep) for ep in episodes}
        self.events = {ev.beat_id: ev for ep in episodes for ev in ep.events}
        self.tags = {bid: tag_beat(ev) for bid, ev in self.events.items()}

    # ---- helpers
    def _events_between(self, ep_no: int, start: float, end: float, kinds=None) -> list[Event]:
        out = []
        for ev in self.indexes[ep_no].ep.events:
            if ev.start >= start - 0.05 and ev.start < end and ev.end <= end + 0.05:
                if kinds is None or ev.kind in kinds:
                    out.append(ev)
        return out

    def _first_line(self, seg: Segment) -> Event | None:
        lines = self._events_between(seg.episode, seg.start, seg.end, {"DIALOGUE"})
        return lines[0] if lines else None

    # ---- hook
    def hook(self, seg: Segment) -> ScoreResult:
        window = self._events_between(seg.episode, seg.start, seg.start + HOOK_WINDOW)
        window = [e for e in window if e.start < seg.start + HOOK_WINDOW]
        reasons: list[str] = []
        if not window:
            return ScoreResult(0.0, ["nothing starts in the first 3 s"])
        best_ev = max(window, key=lambda e: self.tags[e.beat_id].heat)
        bt = self.tags[best_ev.beat_id]
        score = bt.heat
        if bt.cats:
            what = ", ".join(f"{k}:'{v}'" for k, v in list(bt.cats.items())[:2])
            reasons.append(f"{best_ev.kind} at +{best_ev.start - seg.start:.1f}s ({what})")
        lines = [e for e in window if e.kind == "DIALOGUE"]
        if lines:
            first = lines[0]
            ft = self.tags[first.beat_id]
            words = len(first.text.split())
            if continues_sentence(first.text):
                score -= 0.2
                reasons.append(f"opens on the second half of a sentence: \"{first.text[:40]}\"")
            elif words <= 6 and (ft.exclaim or ft.question):
                score += 0.15
                reasons.append(f"short punchy opener: \"{first.text}\"")
            elif ft.question:
                score += 0.1
                reasons.append("opens on a question")
        else:
            score -= 0.05
            reasons.append("no spoken line in first 3 s")
        if any(self.tags[e.beat_id].calm for e in window) and bt.heat < 0.5:
            score -= 0.2
            reasons.append("opens calm")
        if not reasons:
            reasons.append("no hook signal in first 3 s")
        return ScoreResult(clamp(score), reasons)

    # ---- cliffhanger
    def cliffhanger(self, seg: Segment, ends_episode: bool) -> ScoreResult:
        idx = self.indexes[seg.episode]
        final = [e for e in self._events_between(seg.episode, seg.end - CLIFF_WINDOW, seg.end)
                 if e.end >= seg.end - CLIFF_WINDOW]
        closing = self.events[seg.end_beat]
        if closing not in final:
            final.append(closing)
        story = [e for e in final if e.is_beat]
        reasons: list[str] = []
        score = 0.0

        # Cliffhangers often hold on a silent reaction shot after the last line, so the last
        # spoken line counts if it ended within LAST_LINE_WINDOW of the cut.
        seg_lines = self._events_between(seg.episode, seg.start, seg.end, {"DIALOGUE"})
        last_line = seg_lines[-1] if seg_lines and seg.end - seg_lines[-1].end <= LAST_LINE_WINDOW else None
        if last_line:
            lt = self.tags[last_line.beat_id]
            if lt.question:
                score += 0.35
                reasons.append(f"ends on a question: \"{last_line.text[:60]}\"")
            elif PENDING_REQUEST.search(last_line.text):
                score += 0.3
                reasons.append(f"ends on an unanswered request: \"{last_line.text[:60]}\"")
            if lt.cut_off:
                score += 0.3
                reasons.append("last line is cut off")
        peak = max(story, key=lambda e: self.tags[e.beat_id].heat, default=None)
        if peak is not None:
            pt = self.tags[peak.beat_id]
            hot_cats = {k: v for k, v in pt.cats.items()
                        if k in ("threat", "reveal", "betrayal", "violence", "physical", "danger")}
            if hot_cats:
                score += 0.45 * pt.heat
                reasons.append("final moment has " + ", ".join(f"{k}:'{v}'" for k, v in hot_cats.items()))
            elif pt.heat >= 0.5:
                score += 0.25 * pt.heat
                reasons.append(f"heated final moment ({', '.join(pt.cats.values())})")
        if any(self.tags[e.beat_id].reaction for e in final):
            score += 0.2
            reasons.append("shock reaction on screen")
        calm_at_end = [e for e in story if self.tags[e.beat_id].calm]
        if ends_episode and not calm_at_end:
            score += 0.3
            reasons.append("cuts where the episode itself ends (writers' act-out)")
        elif ends_episode:
            # Episodes don't always end on a cliffhanger (Ep 4 ends on a triumphant exit).
            reasons.append("episode ends here, but on a resolved note - no act-out bonus")
        scene = idx.scene_of.get(closing.scene)
        if scene and scene.end is not None and scene.end - seg.end < 1.0 and not ends_episode:
            score += 0.1
            reasons.append("cuts at a scene end")

        calm = [e for e in story if self.tags[e.beat_id].calm]
        if calm:
            score -= 0.35
            reasons.append(f"calm/resolved ending ({self.tags[calm[-1].beat_id].calm_why}) -0.35")
        # Writers sometimes put the word "cliffhanger" in a music/camera note. It never adds points;
        # we only say so, and flag it when the beat underneath is actually calm.
        labelled = [e for e in final if re.search("cliffhanger", e.text, re.I)]
        if labelled and calm:
            reasons.append("script labels this moment 'cliffhanger' but the beat is calm - label ignored")
        elif labelled:
            reasons.append("(script labels this moment 'cliffhanger' - label ignored, scored on content)")
        if last_line and re.match(r"^\s*(yes|thank you|i will|i do|very well|good)\b", last_line.text, re.I) \
                and not self.tags[last_line.beat_id].question:
            score -= 0.15
            reasons.append(f"ends on an answer/agreement: \"{last_line.text[:40]}\"")
        if not reasons:
            reasons.append("no unresolved signal at the cut")
        return ScoreResult(clamp(score), reasons)

    # ---- standalone
    def standalone(self, cand: Candidate) -> ScoreResult:
        score, reasons = 1.0, []
        seg = cand.segments[0]
        idx = self.indexes[seg.episode]
        first = self._first_line(seg)
        if first is not None and first.start - seg.start < HOOK_WINDOW:
            if continues_sentence(first.text):
                score -= 0.3
                reasons.append(f"opens mid-sentence: \"{first.text[:40]}\"")
            elif ANSWER_OPENERS.match(first.text.strip(' "“')):
                score -= 0.25
                reasons.append(f"opens on an answer/continuation: \"{first.text[:40]}\"")
            elif PRONOUN_OPENERS.match(first.text.strip(' "“')):
                score -= 0.1
                reasons.append(f"opens with a pronoun that needs context: \"{first.text[:40]}\"")
        # 65% of lines follow the previous one within 1 s, so "a line just ended" is normal.
        # The real problem is opening on the *reply* to a question the viewer never heard.
        prev = [d for d in idx.lines if d.end <= seg.start + 0.05 and d.scene == self.events[seg.start_beat].scene]
        if prev and seg.start - prev[-1].end < 1.5 and self.tags[prev[-1].beat_id].question:
            score -= 0.2
            reasons.append(f"opens mid-conversation: skips the question \"{prev[-1].text[:40]}\"")
        for c in cand.crosses:
            if c == "time_jump":
                score -= 0.35
                reasons.append("crosses a time jump")
            elif c == "cut":
                score -= 0.1
                reasons.append("crosses a hard scene cut")
        if cand.shape == "stitched":
            score -= 0.05
            reasons.append("stitched: viewer must follow a jump")
            if len(cand.episodes) == 2:
                score -= 0.05
                reasons.append("stitched across two episodes")

        lines = [e for s in cand.segments for e in self._events_between(s.episode, s.start, s.end, {"DIALOGUE"})]
        speakers = {norm_speaker(e.speaker) for e in lines}
        if not lines:
            score -= 0.25
            reasons.append("no dialogue")
        elif len(speakers) == 1:
            score -= 0.15
            reasons.append(f"only one voice ({next(iter(speakers))})")
        offscreen = sum(1 for e in lines if e.delivery)
        if lines and offscreen / len(lines) > 0.5:
            score -= 0.15
            reasons.append("mostly off-screen / voice-over lines")
        if not reasons:
            reasons.append("no context warning signs")
        return ScoreResult(clamp(score), reasons)

    # ---- intensity and length
    def intensity(self, cand: Candidate) -> ScoreResult:
        beats = [self.events[b] for s in cand.segments for b in s.beat_ids]
        if not beats:
            return ScoreResult(0.0, ["no story beats"])
        heats = [self.tags[b.beat_id].heat for b in beats]
        mean = sum(heats) / len(heats)
        hot = sum(h >= 0.5 for h in heats)
        return ScoreResult(clamp(mean * 1.5), [f"{hot}/{len(beats)} heated beats, mean heat {mean:.2f}"])

    @staticmethod
    def length(cand: Candidate) -> ScoreResult:
        d = cand.duration
        lo, hi = LENGTH_BEST
        if lo <= d <= hi:
            v = 1.0
        elif d < lo:
            v = 0.5 + 0.5 * (d - 20) / (lo - 20)
        else:
            v = 1.0 - 0.5 * (d - hi) / (60 - hi)
        return ScoreResult(clamp(v), [f"{d:.1f}s"])

    # ---- everything
    def score(self, cand: Candidate) -> Candidate:
        hook_seg, cliff_seg = cand.segments[0], cand.segments[-1]
        parts = {
            "hook": self.hook(hook_seg),
            "cliffhanger": self.cliffhanger(cliff_seg, cand.ends_episode),
            "standalone": self.standalone(cand),
            "intensity": self.intensity(cand),
            "length": self.length(cand),
        }
        cand.scores = {k: r.value for k, r in parts.items()}
        cand.scores["rule"] = round(sum(WEIGHTS[k] * parts[k].value for k in WEIGHTS), 3)
        cand.reasons = {k: r.reasons for k, r in parts.items()}
        cand.flags = self.flags(cand)
        cand.hook_text = self.describe_hook(hook_seg)
        cand.cliffhanger_text = self.describe_cliff(cliff_seg)
        return cand

    def flags(self, cand: Candidate) -> list[str]:
        s, r = cand.scores, cand.reasons
        out = []
        if s["hook"] < 0.3:
            out.append("weak hook")
        if s["cliffhanger"] < 0.3:
            out.append("weak cliffhanger")
        if any("calm/resolved" in x for x in r["cliffhanger"]):
            out.append("calm ending")
        if any("but the beat is calm" in x for x in r["cliffhanger"]):
            out.append("script 'cliffhanger' label over calm beat")
        for key, flag in (("time jump", "crosses time jump"), ("mid-conversation", "opens mid-conversation"),
                          ("mid-sentence", "opens mid-sentence"),
                          ("answer/continuation", "opens on an answer"), ("only one voice", "single voice"),
                          ("off-screen", "mostly off-screen voices")):
            if any(key in x for x in r["standalone"]):
                out.append(flag)
        return out

    def describe_hook(self, seg: Segment) -> str:
        window = [e for e in self._events_between(seg.episode, seg.start, seg.start + HOOK_WINDOW)
                  if e.start < seg.start + HOOK_WINDOW and e.kind != "CAMERA"]
        if not window:
            return ""
        lines = [e for e in window if e.kind == "DIALOGUE"]
        best = max(window, key=lambda e: self.tags[e.beat_id].heat)
        pick = best if (self.tags[best.beat_id].heat > 0.3 or not lines) else lines[0]
        return _say(pick)

    def describe_cliff(self, seg: Segment) -> str:
        lines = self._events_between(seg.episode, seg.start, seg.end, {"DIALOGUE"})
        if lines and seg.end - lines[-1].end <= LAST_LINE_WINDOW:
            return _say(lines[-1])
        # Describe the last thing that *happens*, not a music cue.
        story = [self.events[b] for b in seg.beat_ids if not self.tags[b].music]
        return _say(story[-1] if story else self.events[seg.end_beat])

    # ---- rankers for stitched pairing (plug into candidates.generate_stitched)
    def hook_rank(self, seg: Segment, idx: EpisodeIndex) -> float:
        return self.hook(seg).value

    def cliff_rank(self, seg: Segment, idx: EpisodeIndex) -> float:
        return self.cliffhanger(seg, abs(seg.end - idx.last_beat_end) <= 0.05).value


def _say(ev: Event) -> str:
    if ev.kind == "DIALOGUE":
        # capwords, not .title(): "BEN'S MOTHER" -> "Ben's Mother" (not "Ben'S")
        return f'{string.capwords(norm_speaker(ev.speaker))}: "{ev.text}"'
    return f"[{ev.kind.lower()}] {ev.text}"


# ---------------------------------------------------------------- CLI


def print_report(episodes: list[Episode], out=sys.stdout) -> None:
    from collections import Counter
    from statistics import mean

    from .candidates import generate_continuous, generate_stitched

    p = lambda *a: print(*a, file=out)  # noqa: E731
    scorer = Scorer(episodes)
    cands = generate_continuous(episodes) + generate_stitched(
        episodes, hook_rank=scorer.hook_rank, cliff_rank=scorer.cliff_rank)
    for c in cands:
        scorer.score(c)

    p(f"Scored {len(cands)} candidates")
    for k in list(WEIGHTS) + ["rule"]:
        vals = sorted(c.scores[k] for c in cands)
        p(f"  {k:<12} mean {mean(vals):.2f}  p10 {vals[len(vals) // 10]:.2f}  "
          f"median {vals[len(vals) // 2]:.2f}  p90 {vals[9 * len(vals) // 10]:.2f}  max {vals[-1]:.2f}")
    flags = Counter(f for c in cands for f in c.flags)
    p("Flags: " + ", ".join(f"{k} {v}" for k, v in flags.most_common()))

    best_per_ep = {}
    for c in cands:
        e = c.segments[0].episode
        if e not in best_per_ep or c.scores["rule"] > best_per_ep[e].scores["rule"]:
            best_per_ep[e] = c
    vals = [c.scores["rule"] for c in best_per_ep.values()]
    weakest = sorted(best_per_ep.items(), key=lambda kv: kv[1].scores["rule"])[:5]
    p(f"Best per episode: min {min(vals):.2f}, mean {mean(vals):.2f}; weakest episodes: "
      + ", ".join(f"Ep{e} ({c.scores['rule']:.2f})" for e, c in weakest))

    def show(c: Candidate) -> None:
        segs = " + ".join(f"Ep{s.episode} {fmt_time(s.start)}-{fmt_time(s.end)}" for s in c.segments)
        sc = " ".join(f"{k[:5]}={v:.2f}" for k, v in c.scores.items())
        p(f"\n{c.id} [{c.shape}, {c.duration}s] {segs}\n  {sc}")
        p(f"  HOOK : {c.hook_text[:110]}\n         why: {'; '.join(c.reasons['hook'])[:150]}")
        p(f"  CLIFF: {c.cliffhanger_text[:110]}\n         why: {'; '.join(c.reasons['cliffhanger'])[:150]}")
        if c.flags:
            p(f"  FLAGS: {', '.join(c.flags)}")

    # ---- de-duplication (Phase 4)
    from .candidates import deduplicate, overlap_ratio

    kept, drops = deduplicate(cands)
    shapes = Counter(c.shape for c in kept)
    p(f"\nDe-duplicated: {len(cands)} -> {len(kept)} promos "
      f"({shapes['continuous']} continuous, {shapes['stitched']} stitched)")
    p("  dropped: " + ", ".join(f"{k} {v}" for k, v in drops.items()))
    per_ep = Counter(c.segments[0].episode for c in kept)
    short = sorted(e.number for e in episodes if per_ep.get(e.number, 0) < 6)
    p(f"  episodes with fewer than 6 promos: {short or 'none'}")
    worst = max((overlap_ratio(a, b) for i, a in enumerate(kept) for b in kept[i + 1:]), default=0.0)
    closes = Counter((c.segments[-1].episode, c.segments[-1].end_beat) for c in kept)
    p(f"  self-check: max overlap between kept promos {worst:.0%} (limit 35%), "
      f"cliffhanger beats used twice: {sum(v > 1 for v in closes.values())}")
    p(f"  rule score of kept promos: best {kept[0].scores['rule']:.2f}, "
      f"median {kept[len(kept) // 2].scores['rule']:.2f}, worst {kept[-1].scores['rule']:.2f}")

    p("\n===== TOP 8 after de-duplication =====")
    for c in kept[:8]:
        show(c)


if __name__ == "__main__":
    from .parser import load_show

    if len(sys.argv) != 2:
        print("usage: python -m promo_finder.features <folder-or-zip>")
        sys.exit(2)
    print_report(load_show(sys.argv[1]))
