"""Pipeline + report tests on a small synthetic show (no confidential data). Run: python tests/test_pipeline.py"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run  # noqa: E402
from promo_finder.candidates import Candidate, Segment  # noqa: E402
from promo_finder.parser import fmt_time  # noqa: E402
from promo_finder.pipeline import posting_order  # noqa: E402

LINES = ['"You lied to me!"', '"Who are you really?"', '"Get out of my house!"', '"I will ruin you."',
         '"The weather is mild."', '"She slaps him."', '"You will pay for this!"', '"Is it true?"']


def write_show(folder: Path, episodes: int = 3) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for n in range(1, episodes + 1):
        out = [f"EPISODE NUMBER: Episode {n}", "", "SCENE 01", "Scene heading: INT. HALL – NIGHT",
               "Scene start: 00:00.00", "Scene end: 01:30.00", ""]
        for i, t in enumerate(range(0, 88, 4)):
            who = ["ANNA (Furious)", "BEN (Shocked)"][i % 2]
            out += ["**CAMERA:**", f"Start: ({fmt_time(t)}) | End: ({fmt_time(t + 4)})", "Close-up.", "",
                    f"**{who}:**", f"Start: ({fmt_time(t + 0.5)}) | End: ({fmt_time(t + 3.5)})",
                    LINES[(i + n) % len(LINES)], ""]
            if i % 5 == 0:
                out += ["**ACTION:**", f"Start: ({fmt_time(t + 0.5)}) | End: ({fmt_time(t + 2)})",
                        "Anna slaps Ben hard.", ""]
        (folder / f"episode-{n:03d}.txt").write_text("\n".join(out), encoding="utf-8")


def promo(cid, ep, final, llm=None, verdict=None):
    c = Candidate(cid, "continuous", [Segment(ep, 0, 30, f"E{ep}-a", f"E{ep}-b", [])])
    c.scores = {"rule": final, "final": final, "llm": llm}
    if llm is not None:
        c.llm = {"verdict": verdict or "post"}
    return c


class PostingOrderTests(unittest.TestCase):
    def test_checked_first_skip_last_and_early_bonus(self):
        a = promo("unchecked-high", 1, 0.99)
        b = promo("checked", 40, 0.70, llm=0.7)
        c = promo("checked-skip", 3, 0.95, llm=0.9, verdict="skip")  # own episode: tests grouping only
        d = promo("checked-early", 2, 0.69, llm=0.7)   # loses 0.01 on score, wins on early bonus
        order = [x.id for x in posting_order([a, b, c, d], 50, 50)]
        self.assertEqual(order, ["checked-early", "checked", "checked-skip", "unchecked-high"])
        self.assertEqual([x.rank for x in posting_order([a, b, c, d], 50, 50)], [1, 2, 3, 4])

    def test_top_ten_never_repeats_an_episode(self):
        ps = [promo(f"p{i}", 1 + i % 3, 0.9 - i * 0.01, llm=0.8) for i in range(12)]
        order = posting_order(ps, 3, 3)
        firsts = [p.segments[0].episode for p in order[:3]]
        self.assertEqual(sorted(firsts), [1, 2, 3])

    def test_low_score_flag(self):
        p = posting_order([promo("weak", 1, 0.2)], 1, 1)[0]
        self.assertIn("low score (quota filler)", p.flags)


class EndToEndTests(unittest.TestCase):
    def test_run_writes_all_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            show, out = Path(tmp) / "Test Show", Path(tmp) / "out"
            write_show(show)
            code = run.main([str(show), "--out", str(out), "--provider", "mock", "--top", "5"])
            self.assertEqual(code, 0)
            for name in ("promos.json", "promos.csv", "report.md", "report.html", "run_meta.json", "llm_raw.jsonl"):
                self.assertTrue((out / name).exists(), name)

            data = json.loads((out / "promos.json").read_text(encoding="utf-8"))
            promos = data["promos"]
            self.assertTrue(promos)
            self.assertEqual([p["rank"] for p in promos], list(range(1, len(promos) + 1)))
            for p in promos:
                self.assertTrue(20 - 0.05 <= p["duration"] <= 60 + 0.05)
                self.assertTrue(p["hook"] and p["cliffhanger"] and p["why"])
                for s in p["segments"]:
                    self.assertRegex(s["start"], r"^\d\d:\d\d\.\d\d$")

            rows = list(csv.DictReader((out / "promos.csv").open(encoding="utf-8-sig")))
            self.assertEqual(len(rows), len(promos))

            meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["episodes"], meta["show"]), (3, "Test Show"))
            self.assertEqual(meta["llm"]["judged"], 5)

            page = (out / "report.html").read_text(encoding="utf-8")
            self.assertIn('id="data"', page)
            self.assertNotIn("__DATA__", page)
            md = (out / "report.md").read_text(encoding="utf-8")
            self.assertIn("## Coverage by episode", md)
            self.assertIn("## Where it picks badly", md)
            self.assertEqual(len((out / "llm_raw.jsonl").read_text(encoding="utf-8").splitlines()), 5)

    def test_rules_only_and_zip_input(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            show = Path(tmp) / "show"
            write_show(show, episodes=2)
            zpath = Path(tmp) / "show.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                for p in show.iterdir():
                    zf.write(p, f"show/{p.name}")
            out = Path(tmp) / "out"
            self.assertEqual(run.main([str(zpath), "--out", str(out)]), 0)
            meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["llm"]["provider"], "none")
            self.assertFalse((out / "llm_raw.jsonl").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
