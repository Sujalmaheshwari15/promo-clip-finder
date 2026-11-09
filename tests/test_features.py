"""Rule-scoring tests on tiny synthetic scripts. Run: python tests/test_features.py"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promo_finder.candidates import Candidate, EpisodeIndex  # noqa: E402
from promo_finder.features import Scorer, continues_sentence, tag_beat  # noqa: E402
from promo_finder.parser import Event, fmt_time, parse_episode  # noqa: E402


def ev(kind, text, emotion=(), speaker=""):
    return Event(1, 1, "", kind, speaker, list(emotion), "", 0.0, 1.0, text)


def script(blocks, number=1, heading2=None, scene2_at=None):
    """blocks: (label, start, end, text). Optional second scene starting at scene2_at."""
    end = max(b[2] for b in blocks) + 1
    out = [f"EPISODE NUMBER: Episode {number}", "", "SCENE 01", "Scene heading: INT. HALL – DAY",
           "Scene start: 00:00.00", f"Scene end: {fmt_time(scene2_at or end)}", ""]
    for label, s, e, text in blocks:
        if scene2_at is not None and s >= scene2_at and "SCENE 02" not in out:
            out += ["SCENE 02", f"Scene heading: {heading2}", f"Scene start: {fmt_time(scene2_at)}",
                    f"Scene end: {fmt_time(end)}", ""]
        out += [f"**{label}:**", f"Start: ({fmt_time(s)}) | End: ({fmt_time(e)})", text, ""]
    return parse_episode("\n".join(out), "t.txt")


def filler(start, end, step=4.0):
    """Neutral back-and-forth so clips are long enough."""
    out, t, who = [], start, 0
    while t + 3 <= end:
        out.append((["ANNA (Neutral)", "BEN (Neutral)"][who], t, t + 3, '"The weather is mild today."'))
        t += step
        who ^= 1
    return out


def whole_clip(ep, start, end):
    idx = EpisodeIndex(ep)
    s_beat = next(b for b in idx.beats if abs(b.start - start) < 0.01)
    # Snap to the last beat ending at or before `end`, like the real generator does.
    e_beat = max((b for b in idx.beats if b.end <= end + 0.01), key=lambda b: (b.end, b.start))
    end = e_beat.end
    seg = idx.make_segment(start, s_beat, end, e_beat)
    return Candidate("T-1", "continuous", [seg], ends_episode=abs(end - idx.last_beat_end) < 0.05)


class TagTests(unittest.TestCase):
    def test_lexicons(self):
        self.assertIn("violence", tag_beat(ev("ACTION", "She slaps him hard.")).cats)
        self.assertNotIn("physical", tag_beat(ev("DIALOGUE", "Witchcraft is a capital charge.")).cats)
        self.assertEqual(tag_beat(ev("ACTION", "He grabs her arm.")).cats.get("physical"), "grabs")
        self.assertGreaterEqual(tag_beat(ev("DIALOGUE", "Leave.", ["Furious"])).heat, 0.9)

    def test_music_cue_with_cliffhanger_word_scores_nothing(self):
        t = tag_beat(ev("SFX", "Strings swell to an ominous cliffhanger peak."))
        self.assertTrue(t.music)
        self.assertEqual(t.heat, 0.0)

    def test_calm(self):
        self.assertTrue(tag_beat(ev("DIALOGUE", "Rest now.", ["Tender"])).calm)
        self.assertFalse(tag_beat(ev("DIALOGUE", "You will pay!", ["Tender"])).calm)  # hot beats aren't calm

    def test_fixes_found_via_llm_disagreement(self):
        # Ep 4: "ruined" as an adjective is not a threat; the triumphant exit is calm.
        exit_ = tag_beat(ev("ACTION", "They walk hand-in-hand into the sunlight, leaving the ruined banquet behind."))
        self.assertNotIn("threat", exit_.cats)
        self.assertTrue(exit_.calm)
        self.assertIn("threat", tag_beat(ev("DIALOGUE", "I will ruin you!")).cats)
        # Ep 12: "the scene freezes" is an edit, not a character reacting; a person freezing still is.
        self.assertFalse(tag_beat(ev("ACTION", "She stares as the scene freezes on her defiance.")).reaction)
        self.assertTrue(tag_beat(ev("ACTION", "Anna freezes in the doorway.")).reaction)

    def test_continues_sentence(self):
        self.assertTrue(continues_sentence("...a secret medicine in her hand."))
        self.assertTrue(continues_sentence("and then he left."))
        self.assertFalse(continues_sentence("Poison."))


class ScoreTests(unittest.TestCase):
    def test_hook_only_counts_first_three_seconds(self):
        slap_first = script([("ACTION", 0, 2, "She slaps him across the face.")] + filler(2.5, 30))
        slap_late = script(filler(0, 10) + [("ACTION", 10, 12, "She slaps him across the face.")] + filler(12.5, 30))
        a = Scorer([slap_first]).hook(whole_clip(slap_first, 0, 29.5).segments[0])
        b = Scorer([slap_late]).hook(whole_clip(slap_late, 0, 29.5).segments[0])
        self.assertGreaterEqual(a.value, 0.9)
        self.assertLess(b.value, 0.3)

    def test_cliffhanger_question_beats_calm_ending(self):
        q = script(filler(0, 25) + [("ANNA (Shocked)", 25, 27, '"Who are you really?"')])
        calm = script(filler(0, 25) + [("ANNA (Tender, Smiling)", 25, 27, '"Thank you. Rest now."')])
        sq, sc = Scorer([q]), Scorer([calm])
        cq = sq.score(whole_clip(q, 0, 27))
        cc = sc.score(whole_clip(calm, 0, 27))
        self.assertGreater(cq.scores["cliffhanger"], 0.6)
        self.assertLess(cc.scores["cliffhanger"], 0.2)
        self.assertIn("calm ending", cc.flags)
        self.assertIn("Who are you really?", cq.cliffhanger_text)

    def test_calm_episode_ending_gets_no_act_out_bonus(self):
        ep = script(filler(0, 25) + [("ACTION", 25, 28, "They walk hand-in-hand into the sunlight.")])
        c = Scorer([ep]).score(whole_clip(ep, 0, 28))
        self.assertTrue(c.ends_episode)
        self.assertFalse(any("writers' act-out" in r for r in c.reasons["cliffhanger"]))
        self.assertIn("calm ending", c.flags)

    def test_unanswered_request_counts_even_after_silent_hold(self):
        ep = script(filler(0, 22) + [("BEN (Solemn)", 22, 24, '"Lady Anna... accept my hand."'),
                                     ("ACTION", 24.5, 28, "She stares down at him, tears in her eyes.")])
        c = Scorer([ep]).score(whole_clip(ep, 0, 28))
        self.assertTrue(any("unanswered request" in r for r in c.reasons["cliffhanger"]))

    def test_cliffhanger_label_adds_no_points(self):
        base = filler(0, 25) + [("ANNA (Shocked)", 25, 27, '"Who are you really?"')]
        plain = script(base + [("SOUND FX", 27, 28, "A low drone.")])
        labelled = script(base + [("SOUND FX", 27, 28, "A low drone swells to a cliffhanger peak.")])
        a = Scorer([plain]).score(whole_clip(plain, 0, 28)).scores["cliffhanger"]
        b = Scorer([labelled]).score(whole_clip(labelled, 0, 28)).scores["cliffhanger"]
        self.assertEqual(a, b)

    def test_standalone_penalties(self):
        # Opening right after a question = opening on the answer the viewer never heard the question to.
        ep = script([("BEN (Neutral)", 0, 2, '"Where were you?"'), ("ANNA (Neutral)", 2.5, 4, '"Yes, at the mill."')]
                    + filler(4.5, 30))
        c = Scorer([ep]).score(whole_clip(ep, 2.5, 29.5))
        self.assertIn("opens mid-conversation", c.flags)
        self.assertIn("opens on an answer", c.flags)
        self.assertLessEqual(c.scores["standalone"], 0.6)

    def test_time_jump_penalty(self):
        blocks = filler(0, 40)
        ep = script(blocks, heading2="EXT. FIELD – YEARS LATER", scene2_at=20)
        c = Scorer([ep]).score(whole_clip(ep, 0, 39))
        self.assertIn("crosses time jump", c.flags)

    def test_length_preference(self):
        ep = script(filler(0, 62))
        s = Scorer([ep])
        self.assertEqual(s.length(whole_clip(ep, 0, 35)).value, 1.0)
        self.assertLess(s.length(whole_clip(ep, 0, 23)).value, 1.0)
        self.assertLess(s.length(whole_clip(ep, 0, 59)).value, 1.0)

    def test_rule_score_is_weighted_sum_in_range(self):
        ep = script([("ACTION", 0, 2, "She slaps him.")] + filler(2.5, 30))
        five = {"hook", "cliffhanger", "standalone", "intensity", "length"}
        c = Scorer([ep], use_tension=False).score(whole_clip(ep, 0, 29.5))
        self.assertTrue(0 <= c.scores["rule"] <= 1)
        self.assertEqual(set(c.reasons), five)
        c = Scorer([ep]).score(whole_clip(ep, 0, 29.5))       # tension curve on: a sixth score
        self.assertTrue(0 <= c.scores["rule"] <= 1)
        self.assertEqual(set(c.reasons), five | {"tension_shape"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
