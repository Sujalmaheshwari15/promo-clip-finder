"""Parser tests on small synthetic scripts (no confidential data). Run: python tests/test_parser.py"""

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promo_finder.parser import (  # noqa: E402
    classify_transition, fmt_time, load_show, parse_block_label, parse_episode, parse_time,
)

SAMPLE = """EPISODE NUMBER: Episode 3
EPISODE LOGLINE: A test logline.
CONTINUITY NOTES: Test notes.

--- SCRIPT START ---

SCENE 01
SCENE HEADING: INT. HALL – NIGHT
SCENE START: 00:00.00
SCENE END: 00:20.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:02.00)
Wide shot of the hall.

**ACTION:**
Start: (00:00.50) | End: (00:02.00)
She slaps him.

**THE STRANGER (O.S.) (Deep, Resonant):**
Start: (00:02.10) | End: (00:04.00)
"Who are you?"

**SOUND FX:**
Start: (00:04.00) | End: (00:05.00)
A door slams.

SCENE 02
Scene heading: EXT. ROAD – DAY
Scene start: 00:20.00
Scene end: 01:05.60

**ANNA (Cold):**
Start: (01:00.00) | End: (01:05.00)
"Never again."

**CAMERA:**
Start: (01:05.00) | End: (01:05.60)
Cut to black.

--- SCRIPT END ---
"""

BROKEN = """EPISODE NUMBER: Episode 9

SCENE 01
Scene heading: INT. ROOM – DAY
Scene start: 00:00.00
Scene end: 00:10.00

**CAMERA:**
A block with no timing line.

**ACTION:**
Start: (00:05.00) | End: (00:03.00)
End before start.

Stray text with no block.

**ALEX:**
Start: (00:08.00) | End: (00:30.00)
"Runs past the scene end."
"""


class ParserTests(unittest.TestCase):
    def test_time_round_trip(self):
        self.assertEqual(parse_time("01:05.60"), 65.6)
        self.assertEqual(parse_time("1:01:05.5"), 3665.5)
        self.assertEqual(fmt_time(65.6), "01:05.60")

    def test_block_label(self):
        self.assertEqual(parse_block_label("CAMERA"), ("CAMERA", "", [], ""))
        self.assertEqual(parse_block_label("SOUND FX"), ("SFX", "", [], ""))
        self.assertEqual(
            parse_block_label("THE BEARDED STRANGER (O.S.) (Deep, Resonant)"),
            ("DIALOGUE", "THE BEARDED STRANGER", ["Deep", "Resonant"], "O.S."),
        )
        self.assertEqual(parse_block_label("ANNA (V.O.)"), ("DIALOGUE", "ANNA", [], "V.O."))

    def test_sample_episode(self):
        ep = parse_episode(SAMPLE, "sample.txt")
        self.assertEqual(ep.issues, [])
        self.assertEqual(ep.number, 3)
        self.assertEqual(len(ep.scenes), 2)
        self.assertEqual(ep.scenes[0].heading, "INT. HALL – NIGHT")  # uppercase keys work
        self.assertEqual(ep.scenes[1].end, 65.6)                     # mixed-case keys work
        self.assertEqual([e.kind for e in ep.events], ["CAMERA", "ACTION", "DIALOGUE", "SFX", "DIALOGUE", "CAMERA"])
        self.assertEqual([e.is_beat for e in ep.events], [False, True, True, True, True, False])
        self.assertEqual([e.is_end_marker for e in ep.events], [False] * 5 + [True])
        self.assertEqual([s.transition for s in ep.scenes], ["episode_start", "cut"])
        line = ep.events[2]
        self.assertEqual((line.speaker, line.delivery, line.emotion), ("THE STRANGER", "O.S.", ["Deep", "Resonant"]))
        self.assertEqual(line.text, "Who are you?")                  # quotes stripped
        self.assertEqual(line.beat_id, "E03-B003")
        self.assertEqual(ep.events[-1].scene, 2)

    def test_classify_transition(self):
        cases = {
            "INT. MANOR CORRIDOR – CONTINUOUS": "continuous",
            "INT. HALL – MOMENTS LATER": "continuous",
            "INT. CATHEDRAL – DAY": "cut",
            "EXT. FIELD – NIGHT (FLASHBACK)": "time_jump",
            "INT. STUDY – LATER": "time_jump",
            "EXT. VILLAGE – THREE YEARS AGO": "time_jump",
            "INT. BEDROOM – NEXT MORNING": "time_jump",
            "INT. HALL – CONTINUOUS (DREAM)": "time_jump",  # time jump wins
        }
        for heading, expected in cases.items():
            self.assertEqual(classify_transition(heading, is_first=False), expected, heading)
        self.assertEqual(classify_transition("INT. HALL – CONTINUOUS", is_first=True), "episode_start")

    def test_only_camera_lines_are_end_markers(self):
        text = (SAMPLE.replace('"Never again."', '"Then cut to black, they said."')
                      .replace("She slaps him.", "She slaps him as the scene fades out."))
        ep = parse_episode(text, "x.txt")
        for ev in ep.events:
            if ev.kind in ("DIALOGUE", "ACTION"):
                self.assertTrue(ev.is_beat, ev.text)
                self.assertFalse(ev.is_end_marker, ev.text)
        self.assertAlmostEqual(ep.duration, 65.6)

    def test_broken_episode_reports_issues_without_crashing(self):
        ep = parse_episode(BROKEN, "broken.txt")
        joined = "\n".join(ep.issues)
        self.assertIn("no Start/End line", joined)
        self.assertIn("end before start", joined)
        self.assertIn("stray text", joined)
        self.assertIn("outside scene 1", joined)

    def test_missing_episode_number_falls_back_to_filename(self):
        ep = parse_episode(SAMPLE.replace("EPISODE NUMBER: Episode 3\n", ""), "episode-012-x.txt")
        self.assertEqual(ep.number, 12)
        self.assertEqual(ep.events[0].beat_id, "E12-B001")

    def test_folder_and_zip_give_same_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "show"
            folder.mkdir()
            (folder / "episode-003.txt").write_text(SAMPLE, encoding="utf-8")
            (folder / "episode-009.txt").write_text(BROKEN, encoding="utf-8")
            zpath = Path(tmp) / "show.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                for p in folder.iterdir():
                    zf.write(p, f"show/{p.name}")
            from_dir = load_show(folder)
            from_zip = load_show(zpath)
            self.assertEqual([e.number for e in from_dir], [3, 9])
            self.assertEqual(
                [ev.to_dict() for ep in from_dir for ev in ep.events],
                [ev.to_dict() for ep in from_zip for ev in ep.events],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
