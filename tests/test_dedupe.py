"""De-duplication tests with hand-built candidates. Run: python tests/test_dedupe.py"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promo_finder.candidates import Candidate, Segment, deduplicate, overlap_ratio  # noqa: E402


def seg(ep, start, end, open_b=None, close_b=None):
    return Segment(ep, start, end, open_b or f"E{ep}-o{start}", close_b or f"E{ep}-c{end}", [])


def cand(cid, score, *segments, shape=None):
    c = Candidate(cid, shape or ("stitched" if len(segments) > 1 else "continuous"), list(segments))
    c.scores = {"rule": score}
    return c


class OverlapTests(unittest.TestCase):
    def test_ratio_uses_shorter_clip(self):
        long_ = cand("a", 1, seg(1, 0, 50))
        inner = cand("b", 1, seg(1, 10, 35))
        self.assertAlmostEqual(overlap_ratio(long_, inner), 1.0)  # inner clip fully inside = near-copy

    def test_no_overlap_across_episodes(self):
        self.assertEqual(overlap_ratio(cand("a", 1, seg(1, 0, 30)), cand("b", 1, seg(2, 0, 30))), 0.0)

    def test_stitched_overlap_counts_each_segment(self):
        st = cand("s", 1, seg(1, 0, 15), seg(2, 10, 25))
        c2 = cand("c", 1, seg(2, 0, 30))
        self.assertAlmostEqual(overlap_ratio(st, c2), 15 / 30)


class DedupeTests(unittest.TestCase):
    def test_keeps_better_of_two_overlapping(self):
        a = cand("a", 0.9, seg(1, 0, 30))
        b = cand("b", 0.8, seg(1, 5, 35))            # 25/30 = 83% overlap -> dropped
        c = cand("c", 0.7, seg(1, 60, 90))           # no overlap -> kept
        kept, drops = deduplicate([b, c, a])
        self.assertEqual([k.id for k in kept], ["a", "c"])
        self.assertEqual(drops["overlaps a better promo"], 1)

    def test_small_overlap_allowed(self):
        a = cand("a", 0.9, seg(1, 0, 30))
        b = cand("b", 0.8, seg(1, 20, 50))           # 10/30 = 33% -> allowed
        kept, _ = deduplicate([a, b])
        self.assertEqual(len(kept), 2)

    def test_same_cliffhanger_beat_not_reused(self):
        a = cand("a", 0.9, seg(1, 0, 12), seg(2, 40, 55, close_b="E2-B050"))
        b = cand("b", 0.8, seg(1, 70, 82), seg(2, 40, 55, close_b="E2-B050"))
        kept, drops = deduplicate([a, b])
        self.assertEqual([k.id for k in kept], ["a"])
        self.assertEqual(drops["reuses hook or cliffhanger beat"], 1)

    def test_quota_per_episode_and_shape(self):
        conts = [cand(f"c{i}", 0.9 - i * 0.01, seg(1, i * 100, i * 100 + 30)) for i in range(6)]
        stits = [cand(f"s{i}", 0.5 - i * 0.01, seg(1, 1000 + i * 100, 1000 + i * 100 + 12),
                      seg(1, 1050 + i * 100, 1050 + i * 100 + 15)) for i in range(4)]
        kept, drops = deduplicate(conts + stits)
        self.assertEqual(sum(k.shape == "continuous" for k in kept), 4)
        self.assertEqual(sum(k.shape == "stitched" for k in kept), 2)
        self.assertEqual(drops["quota full"], 4)
        kept2, _ = deduplicate(conts, quota={"continuous": 2, "stitched": 0})
        self.assertEqual([k.id for k in kept2], ["c0", "c1"])

    def test_output_sorted_by_score_and_deterministic(self):
        cs = [cand(f"x{i}", s, seg(1, i * 100, i * 100 + 30)) for i, s in enumerate([0.3, 0.9, 0.6])]
        kept, _ = deduplicate(cs)
        self.assertEqual([k.id for k in kept], ["x1", "x2", "x0"])
        self.assertEqual([k.id for k in deduplicate(list(reversed(cs)))[0]], ["x1", "x2", "x0"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
