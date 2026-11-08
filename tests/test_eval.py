"""eval.py must catch broken promos, not just pass good ones. Run: python tests/test_eval.py"""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval as evaluator  # noqa: E402
import run  # noqa: E402
from promo_finder.parser import load_show  # noqa: E402
from test_pipeline import write_show  # noqa: E402


class EvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.show, cls.out = root / "Show", root / "out"
        write_show(cls.show)
        assert run.main([str(cls.show), "--out", str(cls.out), "--provider", "mock", "--top", "5"]) == 0
        cls.data = json.loads((cls.out / "promos.json").read_text(encoding="utf-8"))
        cls.script = evaluator.Script(load_show(cls.show))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, promos):
        return evaluator.hard_checks(promos, self.script, 0.35)

    def test_good_run_passes(self):
        fails = self.check(self.data["promos"])
        self.assertEqual({k: v for k, v in fails.items() if v}, {})
        self.assertEqual(evaluator.main(["--out", str(self.out)]), 0)
        self.assertTrue((self.out / "eval.md").exists())

    def test_catches_mid_line_cut_and_fake_cut_point(self):
        promos = copy.deepcopy(self.data["promos"])
        seg = promos[0]["segments"][0]
        seg["start_sec"] += 1.0           # lands inside the first spoken line, on no beat start
        promos[0]["duration"] = round(promos[0]["duration"] - 1.0, 2)
        fails = self.check(promos)
        self.assertTrue(fails["mid-line"])
        self.assertTrue(fails["real cuts"])

    def test_catches_bad_length(self):
        promos = copy.deepcopy(self.data["promos"])
        promos[0]["duration"] = 75.0
        self.assertTrue(self.check(promos)["length"])

    def test_catches_invented_quotes(self):
        promos = copy.deepcopy(self.data["promos"])
        promos[0]["hook"] = 'Anna: "This line was never written."'
        promos[0]["llm"] = {"hook_quote": "nor was this one", "cliffhanger_quote": ""}
        fails = self.check(promos)
        self.assertEqual(len(fails["quotes"]), 2)

    def test_catches_duplicates_and_bad_ranking(self):
        promos = copy.deepcopy(self.data["promos"])
        dup = copy.deepcopy(promos[0])
        dup["id"] = "DUPLICATE"
        promos.insert(1, dup)
        fails = self.check(promos)
        self.assertTrue(fails["overlap"])
        self.assertTrue(fails["ranking"])   # ranks no longer 1..N and top 10 repeats an episode

    def test_catches_backwards_stitch(self):
        promos = copy.deepcopy(self.data["promos"])
        st = next((p for p in promos if p["shape"] == "stitched"), None)
        if st is None:
            self.skipTest("no stitched promo in this tiny show")
        st["segments"].reverse()
        self.assertTrue(self.check(promos)["order"])

    def test_gold_precision_recall(self):
        top = self.data["promos"][0]["segments"][0]
        gold = Path(self.tmp.name) / "gold.csv"
        gold.write_text("episode,start,end,note\n"
                        f"{top['episode']},{top['start']},{top['end']},matches rank 1\n"
                        "99,00:10.00,00:40.00,a moment from an episode we never saw\n", encoding="utf-8")
        lines = evaluator.gold_check(self.data["promos"], gold, k=5)
        text = "\n".join(lines)
        self.assertIn("Recall@5: 1/2", text)
        self.assertIn("missed: Ep99", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
