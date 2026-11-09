"""Tension-curve tests on tiny synthetic scripts. Run: python tests/test_tension.py"""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promo_finder import tension  # noqa: E402
from promo_finder.parser import Event, fmt_time, load_show, parse_episode  # noqa: E402

DEMO = Path(__file__).resolve().parents[1] / "demo" / "the_contract_bride"


def script(blocks, number=1):
    """blocks: (label, start, end, text) -> parsed one-scene Episode."""
    end = max(b[2] for b in blocks)
    out = [f"EPISODE NUMBER: Episode {number}", "", "SCENE 01", "Scene heading: INT. HALL – DAY",
           "Scene start: 00:00.00", f"Scene end: {fmt_time(end)}", ""]
    for label, s, e, text in blocks:
        out += [f"**{label}:**", f"Start: ({fmt_time(s)}) | End: ({fmt_time(e)})", text, ""]
    return parse_episode("\n".join(out), "t.txt")


def build_up_then_calm():
    """0-20 s small talk, 20-40 s escalating fight (hottest at 36-40 s), 40-60 s tender calm."""
    blocks = []
    for i, t in enumerate(range(0, 20, 4)):
        blocks.append((["ANNA (Neutral)", "BEN (Neutral)"][i % 2], t, t + 3, '"The weather is mild today."'))
    blocks += [
        ("BEN (Stern)", 20, 23, '"Where were you last night?"'),
        ("ANNA (Defiant)", 24, 27, '"That is none of your concern."'),
        ("BEN (Accusing, Bitter)", 28, 31, '"You lied to me. I saw the letter."'),
        ("ANNA (Desperate)", 32, 35, '"Please, it is not what you think!"'),
        ("ACTION", 36, 37.5, "Ben slaps the table and grabs her arm violently."),
        ("BEN (Furious, Threatening)", 37.5, 40, '"You will pay for this betrayal!"'),
    ]
    for i, t in enumerate(range(40, 60, 4)):
        blocks.append((["ANNA (Gentle, Tender)", "BEN (Soft, Relieved)"][i % 2], t, t + 3,
                       '"I forgive you. Rest now, you are safe."'))
    return script(blocks)


class EventValueTests(unittest.TestCase):
    def ev(self, kind, text, emotion=()):
        return Event(1, 1, "", kind, "X" if kind == "DIALOGUE" else "", list(emotion), "", 0.0, 1.0, text)

    def test_hot_above_neutral_above_calm(self):
        hot, _ = tension.event_tension(self.ev("DIALOGUE", "I will kill you!", ["Furious"]))
        neutral, _ = tension.event_tension(self.ev("DIALOGUE", "The weather is mild today."))
        calm, _ = tension.event_tension(self.ev("DIALOGUE", "I love you.", ["Gentle"]))
        self.assertGreater(hot, neutral)
        self.assertGreater(neutral, calm)
        self.assertEqual(neutral, tension.NEUTRAL)

    def test_question_adds_a_little(self):
        q, _ = tension.event_tension(self.ev("DIALOGUE", "Where were you last night?"))
        s, _ = tension.event_tension(self.ev("DIALOGUE", "I was at home last night."))
        self.assertGreater(q, s)

    def test_music_mood(self):
        self.assertEqual(tension.music_value("An ominous cello drone builds.")[0], 0.6)
        self.assertEqual(tension.music_value("Tender strings swell softly.")[0], 0.0)
        # The word "cliffhanger" alone must not make music tense (same rule as features.py).
        self.assertEqual(tension.music_value("Strings swell to a cliffhanger.")[0], tension.NEUTRAL)


class CurveTests(unittest.TestCase):
    def test_values_in_0_1_on_grid(self):
        ep = build_up_then_calm()
        pts = tension.tension_curve(ep)
        self.assertTrue(all(0.0 <= v <= 1.0 for _, v in pts))
        self.assertEqual(pts[1][0] - pts[0][0], tension.STEP)
        self.assertAlmostEqual(max(v for _, v in pts), 1.0)   # normalised per episode
        self.assertLess(min(v for _, v in pts), 0.1)            # calm end is near 0 (not forced to 0)

    def test_peak_normalisation_keeps_a_tense_floor(self):
        # Tense all the way through, louder in the middle: the quiet-but-tense parts must stay
        # well above 0 (min-max would stretch them to 0 - the Ep 12 ending problem).
        ep = script([("ANNA (Steely)", t, t + 3, '"Tell me where it is."') for t in range(0, 40, 4)]
                    + [("BEN (Furious)", 40, 43, '"I will kill you!"')]
                    + [("ANNA (Steely)", t, t + 3, '"Tell me where it is."') for t in range(44, 80, 4)])
        self.assertGreater(tension.analyse(ep).value_at(70), 0.4)
        self.assertLess(tension.analyse(ep, norm="minmax").value_at(70), 0.2)

    def test_values_in_0_1_on_demo_show(self):
        for ep in load_show(DEMO):
            vals = [v for _, v in tension.tension_curve(ep)]
            self.assertTrue(vals and all(0.0 <= v <= 1.0 for v in vals), ep.number)

    def test_build_up_peaks_at_the_fight_then_drops(self):
        curve = tension.analyse(build_up_then_calm())
        top = max(curve.peaks, key=lambda p: p["value"])
        self.assertTrue(33 <= top["time"] <= 41, top)            # the slap + threat, 36-40 s
        self.assertTrue(any(d["peak_time"] == top["time"] for d in curve.drops), curve.drops)
        self.assertLess(curve.value_at(55), 0.2)                  # calm ending is low
        self.assertLess(curve.value_at(8), curve.value_at(top["time"]) - 0.4)   # small talk is lower

    def test_flat_episode(self):
        ep = script([("ANNA (Neutral)", t, t + 3, '"The weather is mild today."') for t in range(0, 30, 4)])
        curve = tension.analyse(ep)
        self.assertTrue(curve.flat)
        self.assertEqual(curve.peaks, [])
        self.assertTrue(all(v == 0.0 for _, v in curve.points))

    def test_smoothing_window_is_configurable(self):
        ep = build_up_then_calm()
        sharp = [v for _, v in tension.tension_curve(ep, window_sec=1.0)]
        soft = [v for _, v in tension.tension_curve(ep, window_sec=10.0)]
        roughness = lambda vs: sum(abs(a - b) for a, b in zip(vs, vs[1:]))  # noqa: E731
        self.assertLess(roughness(soft), roughness(sharp))

    def test_peaks_respect_min_gap_and_threshold(self):
        pts = [(i * 0.5, v) for i, v in enumerate([0, 0.7, 0, 0.9, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.8, 0])]
        peaks = tension.find_peaks(pts)
        self.assertEqual([p["time"] for p in peaks], [1.5, 7.0])  # 0.7 merged into 0.9; 0.5 too low
        drops = tension.find_drops(pts, peaks)
        self.assertEqual([d["peak_time"] for d in drops], [1.5, 7.0])

    def test_save_curves_roundtrip(self):
        import json
        import tempfile
        curve = tension.analyse(build_up_then_calm())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tension_curves.json"
            tension.save_curves({1: curve}, path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(data["1"]), {"episode", "step", "points", "peaks", "drops", "flat"})
        self.assertEqual(len(data["1"]["points"]), len(curve.points))


class QuietAndDecayTests(unittest.TestCase):
    def test_quiet_menace_is_medium_and_old_tags_unchanged(self):
        from promo_finder.features import tag_beat
        e = Event(1, 1, "", "DIALOGUE", "JULIANA", ["Steely", "Resolute"], "", 0, 2, "Tell me where the garden is.")
        self.assertEqual(tension.event_tension(e)[0], tension.QUIET_VALUE)
        self.assertEqual(tension.event_tension(e, quiet=False)[0], tension.NEUTRAL)
        self.assertEqual(tag_beat(e).heat, 0.0)        # the rule-score lexicons don't see it
        loud = Event(1, 1, "", "DIALOGUE", "X", ["Furious"], "", 0, 2, "Get out!")
        self.assertGreater(tension.event_tension(loud)[0], tension.QUIET_VALUE)

    def test_held_camera_shot_carries_tension_with_half_life(self):
        ep = script([("BEN (Furious, Threatening)", 0, 3, '"You will pay for this!"'),
                     ("CAMERA", 3, 13, "Close-up of Anna, holding on her face.")])
        raw = dict(tension.raw_grid(ep))
        hot = raw[2.5]
        expected = tension.NEUTRAL + (hot - tension.NEUTRAL) * 0.5   # one half-life later
        self.assertAlmostEqual(raw[2.5 + tension.DECAY_HALF_LIFE], expected, places=2)
        self.assertGreater(raw[8.0], tension.NEUTRAL)
        off = dict(tension.raw_grid(ep, half_life=None))
        self.assertEqual(off[8.0], tension.NEUTRAL)                   # old behaviour


class ShapeScoreTests(unittest.TestCase):
    def setUp(self):
        from promo_finder.features import Scorer
        from test_features import whole_clip
        self.ep = build_up_then_calm()
        self.scorer = Scorer([self.ep])
        self.clip = lambda s, e: self.scorer.score(whole_clip(self.ep, s, e))

    def test_cut_at_peak_beats_cut_after_drop(self):
        at_peak = self.clip(12, 40)       # ends on "You will pay for this betrayal!"
        after = self.clip(24, 51)         # runs on into "I forgive you. Rest now..."
        self.assertGreater(at_peak.scores["tension_shape"], after.scores["tension_shape"] + 0.3)
        self.assertIn("cuts at a tension peak; resolution withheld", at_peak.reasons["tension_shape"])
        self.assertIn(tension.RESOLVED_FLAG, after.flags)
        self.assertNotIn(tension.RESOLVED_FLAG, at_peak.flags)

    def test_episode_end_drop_is_neutral(self):
        c = self.clip(20, 59)
        self.assertTrue(c.ends_episode)
        self.assertIsNone(c.scores["tension_shape"])
        from promo_finder.features import WEIGHTS
        self.assertEqual(c.scores["rule"], round(sum(WEIGHTS[k] * c.scores[k] for k in WEIGHTS), 3))
        self.assertTrue(c.reasons["tension_shape"][0].startswith("cuts at the episode's end"))
        self.assertNotIn(tension.RESOLVED_FLAG, c.flags)

    def test_episode_end_cut_has_no_resolved_reason(self):
        # 24-60 s: the fight peaks at ~38 s and has fallen into the calm by the episode's end.
        curve = tension.analyse(self.ep)
        args = (curve, 24.0, curve, 24.0, 56.0, 59.0)
        mid = tension.shape_score(*args, ends_episode=False)       # same cut, scored normally
        self.assertTrue(mid.resolved)
        self.assertTrue(any(tension.RESOLVED_FLAG in r for r in mid.reasons))
        end = tension.shape_score(*args, ends_episode=True)        # at the episode's end: not scored
        self.assertIsNone(end.value)
        self.assertFalse(end.resolved)
        self.assertEqual(end.reasons, ["cuts at the episode's end - not scored, left to the cliffhanger score"])

    def test_no_tension_is_the_old_formula_on_demo_show(self):
        from promo_finder.candidates import generate_continuous, generate_stitched
        from promo_finder.features import WEIGHTS, Scorer, rule_weights
        eps = load_show(DEMO)
        s = Scorer(eps, use_tension=False)
        cands = generate_continuous(eps) + generate_stitched(eps, hook_rank=s.hook_rank, cliff_rank=s.cliff_rank)
        for c in cands:
            s.score(c)
            self.assertNotIn("tension_shape", c.scores)
            self.assertEqual(c.scores["rule"], round(sum(WEIGHTS[k] * c.scores[k] for k in WEIGHTS), 3))
        self.assertAlmostEqual(sum(rule_weights(True).values()), 1.0)
        self.assertEqual(rule_weights(False), WEIGHTS)


class OutputTests(unittest.TestCase):
    def test_run_writes_curves_and_sparklines(self):
        import json
        import tempfile
        import xml.etree.ElementTree as ET
        from run import run_show
        with tempfile.TemporaryDirectory() as d:
            for use_tension in (True, False):
                out = Path(d) / str(use_tension)
                result, meta, paths, _ = run_show(DEMO, out, None, say=lambda *a: None, use_tension=use_tension)
                curves = json.loads((out / "tension_curves.json").read_text(encoding="utf-8"))
                self.assertEqual(set(curves), {str(e.number) for e in result.episodes})
                page = (out / "report.html").read_text(encoding="utf-8")
                raw = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1)
                records = json.loads(raw.replace("<\\/", "</"))
                sparks = [r["spark"] for r in records if r.get("spark")]
                self.assertEqual(len(sparks), min(15, len(records)))
                for svg in re.findall(r"<svg.*?</svg>", sparks[0], re.S):
                    ET.fromstring(svg)                                   # well-formed SVG
                self.assertEqual(meta["settings"]["tension_curve"], use_tension)
                self.assertEqual("tension_shape" in meta["settings"]["rule_weights"], use_tension)


if __name__ == "__main__":
    unittest.main()
