"""Candidate generation tests on synthetic scripts. Run: python tests/test_candidates.py"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promo_finder.candidates import (  # noqa: E402
    CLIFF_SEG_MAX, CLIFF_SEG_MIN, HOOK_SEG_MAX, HOOK_SEG_MIN, STITCH_MIN_GAP,
    EpisodeIndex, generate_continuous, generate_stitched,
)
from promo_finder.parser import fmt_time, parse_episode  # noqa: E402


def make_script(number: int, length: int = 90, second_scene_at: int | None = None,
                second_heading: str = "INT. HALL – CONTINUOUS") -> str:
    """A fake episode: a camera shot + a 3 s line every 5 s, plus one long line
    (40.0-48.0) that an ACTION beat starts in the middle of (44.0)."""
    out = [f"EPISODE NUMBER: Episode {number}", "", "SCENE 01", "Scene heading: INT. ROOM – DAY",
           "Scene start: 00:00.00", f"Scene end: {fmt_time(second_scene_at or length)}", ""]

    def block(label, start, end, text):
        out.extend([f"**{label}:**", f"Start: ({fmt_time(start)}) | End: ({fmt_time(end)})", text, ""])

    for t in range(0, length, 5):
        if second_scene_at is not None and t == second_scene_at:
            out.extend(["SCENE 02", f"Scene heading: {second_heading}",
                        f"Scene start: {fmt_time(t)}", f"Scene end: {fmt_time(length)}", ""])
        block("CAMERA", t, t + 5, "A shot.")
        if t == 40:
            block("ALEX (Angry)", 40, 48, '"This is one very long line that keeps going."')
            block("ACTION", 44, 46, "Someone slams a door.")
        elif t != 45:
            block("ALEX (Calm)", t + 0.5, t + 3.5, f'"Line at {t}?"')
    block("CAMERA", length - 1, length, "Cut to black.")
    return "\n".join(out)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.ep1 = parse_episode(make_script(1), "e1.txt")
        self.ep2 = parse_episode(make_script(2), "e2.txt")
        self.assertEqual(self.ep1.issues, [])

    def test_continuous_lengths_and_cut_points(self):
        cands = generate_continuous([self.ep1])
        self.assertTrue(cands)
        idx = EpisodeIndex(self.ep1)
        beat_starts = {b.start for b in idx.beats if b.kind in ("DIALOGUE", "ACTION")}
        beat_ends = {b.end for b in idx.beats}
        for c in cands:
            self.assertEqual(len(c.segments), 1)
            seg = c.segments[0]
            self.assertGreaterEqual(c.duration, 20 - 0.05)
            self.assertLessEqual(c.duration, 60 + 0.05)
            self.assertIn(seg.start, beat_starts)       # opens on a real beat
            self.assertIn(seg.end, beat_ends)           # closes on a real beat's end
            self.assertEqual(seg.beat_ids[0], seg.start_beat)

    def test_never_cuts_inside_a_spoken_line(self):
        # The ACTION at 44.0 starts while ALEX is talking (40-48): it must not be a start point,
        # and the ACTION's end (46.0) must not be an end point.
        cands = generate_continuous([self.ep1])
        self.assertFalse(any(c.segments[0].start == 44.0 for c in cands))
        self.assertFalse(any(c.segments[0].end == 46.0 for c in cands))
        self.assertTrue(any(c.segments[0].end == 48.0 for c in cands))  # end of the long line is fine

    def test_never_ends_on_cut_to_black(self):
        cands = generate_continuous([self.ep1])
        self.assertFalse(any(c.segments[0].end == 90.0 for c in cands))
        last_line_end = 85 + 3.5
        self.assertTrue(all(c.segments[0].end <= last_line_end for c in cands))
        self.assertTrue(any(c.ends_episode for c in cands))

    def test_scene_crossing_is_recorded(self):
        ep = parse_episode(make_script(3, second_scene_at=50, second_heading="EXT. FIELD – LATER"), "e3.txt")
        self.assertEqual(ep.issues, [])
        cands = generate_continuous([ep])
        crossing = [c for c in cands if c.segments[0].start < 50 < c.segments[0].end]
        self.assertTrue(crossing)
        self.assertTrue(all(c.crosses == ["time_jump"] for c in crossing))
        self.assertTrue(all(c.crosses == [] for c in cands if c.segments[0].end <= 50))

    def test_stitched_rules(self):
        cands = generate_stitched([self.ep1, self.ep2], k_hooks=4, k_cliffs=4)
        self.assertTrue(cands)
        self.assertTrue(any(len(c.episodes) == 2 for c in cands))  # cross-episode pairs exist
        for c in cands:
            hook, cliff = c.segments
            self.assertTrue(HOOK_SEG_MIN - 0.05 <= hook.duration <= HOOK_SEG_MAX + 0.05)
            self.assertTrue(CLIFF_SEG_MIN - 0.05 <= cliff.duration <= CLIFF_SEG_MAX + 0.05)
            self.assertTrue(20 - 0.05 <= c.duration <= 60 + 0.05)
            self.assertIn(cliff.episode, (hook.episode, hook.episode + 1))
            if cliff.episode == hook.episode:
                self.assertGreaterEqual(cliff.start, hook.end + STITCH_MIN_GAP - 0.05)

    def test_stitched_never_pairs_with_non_adjacent_episode(self):
        ep5 = parse_episode(make_script(5), "e5.txt")
        cands = generate_stitched([self.ep1, ep5])
        self.assertTrue(all(len(c.episodes) == 1 for c in cands))

    def test_ids_are_unique_and_stable(self):
        a = [c.id for c in generate_continuous([self.ep1, self.ep2]) + generate_stitched([self.ep1, self.ep2])]
        b = [c.id for c in generate_continuous([self.ep1, self.ep2]) + generate_stitched([self.ep1, self.ep2])]
        self.assertEqual(len(a), len(set(a)))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
