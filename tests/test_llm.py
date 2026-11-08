"""LLM client + judge tests. A tiny local HTTP server stands in for the provider,
so these run offline. Run: python tests/test_llm.py"""

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promo_finder.candidates import Candidate, EpisodeIndex  # noqa: E402
from promo_finder.features import Scorer  # noqa: E402
from promo_finder.judge import (  # noqa: E402
    apply_cut_suggestion, build_prompt, extract_json, judge_candidates, rank_key, validate_reply,
)
from promo_finder.llm import LLMError, make_client, mock_reply  # noqa: E402
from promo_finder.parser import fmt_time, parse_episode  # noqa: E402


# ---------------------------------------------------------------- fake provider server

class FakeProvider(BaseHTTPRequestHandler):
    script: list = []      # list of (status, body_dict, headers) to return in order
    seen: list = []        # requests received

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        headers = {k.lower(): v for k, v in self.headers.items()}  # HTTP header names are case-insensitive
        FakeProvider.seen.append({"path": self.path, "headers": headers, "body": body})
        status, payload, headers = FakeProvider.script.pop(0)
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def do_GET(self):  # /models listing
        FakeProvider.seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"data": [{"id": "models/fake-b"}, {"id": "fake-a"}]}).encode())

    def log_message(self, *a):
        pass


def openai_ok(text):
    return (200, {"model": "fake-1", "choices": [{"message": {"content": text}, "finish_reason": "stop"}]}, None)


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeProvider)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeProvider.script, FakeProvider.seen = [], []
        os.environ["FAKE_KEY"] = "sk-test"

    def client(self, **kw):
        c = make_client("custom", model="fake-1", base_url=self.url, api_key_env="FAKE_KEY", **kw)
        c._backoff = lambda attempt: 0.01  # keep tests fast
        return c

    def test_openai_style_request_and_reply(self):
        FakeProvider.script = [openai_ok('{"hook": 7}')]
        reply = self.client().complete("sys", "user")
        self.assertEqual(reply.text, '{"hook": 7}')
        req = FakeProvider.seen[0]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["headers"]["authorization"], "Bearer sk-test")
        self.assertEqual([m["role"] for m in req["body"]["messages"]], ["system", "user"])
        # Cloudflare (in front of Groq) blocks urllib's default UA with "403 error code: 1010".
        self.assertEqual(req["headers"]["user-agent"], "promo-finder/1.0")

    def test_retries_429_then_succeeds(self):
        FakeProvider.script = [(429, {"error": "slow down"}, {"Retry-After": "0"}),
                               (503, {"error": "busy"}, None), openai_ok("ok")]
        reply = self.client(max_retries=3).complete("s", "u")
        self.assertEqual((reply.text, reply.attempts), ("ok", 3))

    def test_gives_up_after_max_retries(self):
        FakeProvider.script = [(500, {"e": 1}, None)] * 3
        with self.assertRaises(LLMError) as ctx:
            self.client(max_retries=2).complete("s", "u")
        self.assertIn("gave up after 3 attempts", str(ctx.exception))

    def test_non_retryable_error_fails_fast(self):
        FakeProvider.script = [(401, {"error": "bad key"}, None), openai_ok("never reached")]
        with self.assertRaises(LLMError):
            self.client().complete("s", "u")
        self.assertEqual(len(FakeProvider.seen), 1)

    def test_unknown_model_is_fatal_with_hint(self):
        FakeProvider.script = [(404, {"error": {"message": "model retired"}}, None)]
        with self.assertRaises(LLMError) as ctx:
            self.client().complete("s", "u")
        self.assertTrue(ctx.exception.fatal)
        self.assertIn("--list-models", str(ctx.exception))
        self.assertNotIn("\n", str(ctx.exception))

    def test_gemini_preset_sends_low_reasoning_effort(self):
        os.environ["GEMINI_TEST_KEY"] = "k"
        c = make_client("gemini", base_url=self.url, api_key_env="GEMINI_TEST_KEY")
        FakeProvider.script = [openai_ok("{}")]
        c.complete("s", "u")
        self.assertEqual(FakeProvider.seen[0]["body"]["reasoning_effort"], "low")
        # Groq's default (gpt-oss-120b) is a reasoning model too; OpenRouter's Llama is not.
        self.assertEqual(make_client("groq", api_key_env="GEMINI_TEST_KEY").cfg.extra_body["reasoning_effort"], "low")
        self.assertNotIn("reasoning_effort", make_client("openrouter", api_key_env="GEMINI_TEST_KEY").cfg.extra_body)

    def test_gemini_bad_key_400_is_fatal(self):
        FakeProvider.script = [(400, {"error": {"message": "Please pass a valid API key"}}, None)]
        with self.assertRaises(LLMError) as ctx:
            self.client().complete("s", "u")
        self.assertTrue(ctx.exception.fatal)

    def test_daily_quota_429_is_fatal_without_retries(self):
        body = {"error": {"code": 429, "message": "You exceeded your current quota", "details": [
            {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}
        FakeProvider.script = [(429, body, None), openai_ok("never reached")]
        with self.assertRaises(LLMError) as ctx:
            self.client(max_retries=3).complete("s", "u")
        self.assertTrue(ctx.exception.fatal)
        self.assertIn("daily free-tier quota", str(ctx.exception))
        self.assertEqual(len(FakeProvider.seen), 1)

    def test_retry_delay_read_from_body(self):
        from promo_finder.llm import _retry_delay_in_body
        self.assertEqual(_retry_delay_in_body('{"details":[{"retryDelay": "34s"}]}'), 35.0)
        self.assertIsNone(_retry_delay_in_body("{}"))

    def test_list_models(self):
        self.assertEqual(self.client().list_models(), ["fake-a", "fake-b"])
        self.assertEqual(FakeProvider.seen[0]["path"], "/v1/models")
        self.assertEqual(FakeProvider.seen[0]["headers"]["authorization"], "Bearer sk-test")

    def test_missing_key_is_a_clear_error(self):
        os.environ.pop("GROQ_API_KEY", None)
        with self.assertRaises(LLMError) as ctx:
            make_client("groq")
        self.assertIn("GROQ_API_KEY", str(ctx.exception))

    def test_key_reading_forgives_cmd_slips(self):
        from promo_finder.llm import read_key
        os.environ["SLIP_A "] = " gsk_abc"            # set SLIP_A = gsk_abc
        os.environ["SLIP_B"] = '"gsk_def"'            # set SLIP_B="gsk_def"
        try:
            self.assertEqual(read_key("SLIP_A"), "gsk_abc")
            self.assertEqual(read_key("SLIP_B"), "gsk_def")
            self.assertEqual(read_key("SLIP_MISSING"), "")
        finally:
            os.environ.pop("SLIP_A ", None)
            os.environ.pop("SLIP_B", None)

    def test_anthropic_native_messages_api(self):
        os.environ["ANTHROPIC_TEST_KEY"] = "sk-ant-test"
        c = make_client("anthropic", base_url=self.url, api_key_env="ANTHROPIC_TEST_KEY")
        FakeProvider.script = [(200, {"model": "claude-opus-5-5", "stop_reason": "end_turn", "content": [
            {"type": "thinking", "thinking": ""}, {"type": "text", "text": '{"hook": 9}'}]}, None)]
        self.assertEqual(c.complete("sys", "user").text, '{"hook": 9}')
        req = FakeProvider.seen[0]
        self.assertEqual(req["path"], "/v1/messages")
        self.assertEqual(req["headers"]["x-api-key"], "sk-ant-test")
        self.assertEqual(req["body"]["system"], "sys")
        self.assertNotIn("temperature", req["body"])

    def test_anthropic_refusal_is_an_llm_error(self):
        os.environ["ANTHROPIC_TEST_KEY"] = "sk-ant-test"
        c = make_client("anthropic", base_url=self.url, api_key_env="ANTHROPIC_TEST_KEY")
        FakeProvider.script = [(200, {"stop_reason": "refusal", "stop_details": {"category": "x"}, "content": []}, None)]
        with self.assertRaises(LLMError):
            c.complete("s", "u")

    def test_none_and_mock(self):
        self.assertIsNone(make_client("none"))
        m = make_client("mock")
        self.assertEqual(m.complete("s", "u [E01-B001]").text, m.complete("s", "u [E01-B001]").text)


# ---------------------------------------------------------------- judge

def make_episode():
    lines = []
    for i, t in enumerate(range(0, 60, 5)):
        lines += [f"**CAMERA:**", f"Start: ({fmt_time(t)}) | End: ({fmt_time(t + 5)})", "A shot.", "",
                  f"**{'ANNA' if i % 2 else 'BEN'} (Angry):**", f"Start: ({fmt_time(t + 0.5)}) | End: ({fmt_time(t + 3.5)})",
                  f'"Line number {i}, you liar?"', ""]
    text = ("EPISODE NUMBER: Episode 1\n\nSCENE 01\nScene heading: INT. HALL – DAY\nScene start: 00:00.00\n"
            "Scene end: 01:00.00\n\n" + "\n".join(lines))
    return parse_episode(text, "e1.txt")


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.ep = make_episode()
        self.scorer = Scorer([self.ep])
        idx = EpisodeIndex(self.ep)
        start = dict((b.beat_id, (t, b)) for t, b in idx.starts)
        end = dict((b.beat_id, (t, b)) for t, b in idx.ends)
        s_id, e_id = "E01-B002", "E01-B014"   # BEN line at 0.5 s ... line at 30.5-33.5
        seg = idx.make_segment(start[s_id][0], start[s_id][1], end[e_id][0], end[e_id][1])
        self.cand = self.scorer.score(Candidate("C-T-1", "continuous", [seg]))

    def test_prompt_has_beat_ids_and_no_timestamps(self):
        p = build_prompt(self.scorer, self.cand)
        self.assertIn("[E01-B002]", p)
        self.assertNotRegex(p, r"\d{2}:\d{2}\.\d{2}")

    def test_extract_json_handles_fences_and_chatter(self):
        self.assertEqual(extract_json('Sure!\n```json\n{"hook": 5}\n```'), {"hook": 5})
        with self.assertRaises(ValueError):
            extract_json("no json here")

    def test_validation_clamps_drops_fake_quotes_and_timestamps(self):
        obj = {"hook": 14, "cliffhanger": "x", "standalone": 6, "verdict": "YES",
               "why": "Great slap at 00:03.20 here.", "hook_quote": "line number 0, you liar",
               "cliffhanger_quote": "a line that was never written"}
        clean, problems = validate_reply(obj, self.scorer, self.cand)
        self.assertEqual((clean["hook"], clean["cliffhanger"], clean["verdict"]), (10.0, 0.0, "maybe"))
        self.assertEqual(clean["hook_quote"], "line number 0, you liar")   # real quote kept (case-insensitive)
        self.assertEqual(clean["cliffhanger_quote"], "")                 # fake quote dropped
        self.assertNotRegex(clean["why"], r"\d:\d\d")
        self.assertEqual(clean["why"], "Great slap here.")
        self.assertEqual(len(problems), 5)

    def test_valid_cut_suggestion_is_applied(self):
        clean = {"new_start_beat": "E01-B004", "new_end_beat": None}     # line at 5.5 s
        changed, problems = apply_cut_suggestion(clean, self.scorer, self.cand)
        self.assertTrue(changed, problems)
        self.assertEqual(self.cand.segments[0].start, 5.5)
        self.assertIn("->", self.cand.llm_adjusted)

    def test_invalid_cut_suggestions_are_ignored(self):
        for bad in ({"new_start_beat": "E01-B003"},          # a CAMERA beat - not a legal cut
                    {"new_start_beat": "E01-B999"},          # doesn't exist
                    {"new_start_beat": "E01-B012"}):         # legal beat, but leaves < 20 s
            clean = {"new_start_beat": None, "new_end_beat": None, **bad}
            changed, problems = apply_cut_suggestion(clean, self.scorer, self.cand)
            self.assertFalse(changed, bad)
            self.assertTrue(problems, bad)
        self.assertEqual(self.cand.segments[0].start, 0.5)

    def test_failed_llm_falls_back_to_rules(self):
        class Broken:
            label = "broken:x"

            def complete(self, s, u):
                raise LLMError("HTTP 503 forever")
        stats = judge_candidates([self.cand], self.scorer, Broken(), top_n=5)
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(self.cand.scores["final"], self.cand.scores["rule"])
        self.assertTrue(any(f.startswith("llm failed") for f in self.cand.flags))

    def test_stops_calling_after_three_failures_in_a_row(self):
        class Broken:
            label = "broken:x"
            calls = 0

            def complete(self, s, u):
                Broken.calls += 1
                raise LLMError("SSL error")
        cands = []
        for i in range(6):
            c = Candidate(f"C-T-{i}", "continuous", list(self.cand.segments))
            c.scores = {"rule": 0.9 - i * 0.01}
            cands.append(c)
        stats = judge_candidates(cands, self.scorer, Broken(), top_n=6)
        self.assertEqual(Broken.calls, 3)
        self.assertEqual((stats["failed"], stats["skipped"]), (3, 3))
        self.assertTrue(all(c.scores["final"] == c.scores["rule"] for c in cands))
        self.assertIn("SSL error", stats["last_error"])

    def test_fatal_error_stops_after_first_call(self):
        class WrongModel:
            label = "x:retired"
            calls = 0

            def complete(self, s, u):
                WrongModel.calls += 1
                raise LLMError("HTTP 404: model retired", fatal=True)
        cands = []
        for i in range(4):
            c = Candidate(f"C-F-{i}", "continuous", list(self.cand.segments))
            c.scores = {"rule": 0.9 - i * 0.01}
            cands.append(c)
        stats = judge_candidates(cands, self.scorer, WrongModel(), top_n=4)
        self.assertEqual(WrongModel.calls, 1)
        self.assertEqual((stats["failed"], stats["skipped"]), (1, 3))

    def test_ssl_context_uses_explicit_bundle(self):
        from promo_finder.llm import KNOWN_CA_BUNDLES, build_ssl_context
        existing = [p for p in KNOWN_CA_BUNDLES if os.path.isfile(p)]
        if not existing:
            self.skipTest("no CA bundle file on this machine")
        ctx, src = build_ssl_context(existing[0])
        self.assertEqual(src, existing[0])
        self.assertGreater(len(ctx.get_ca_certs()), 0)

    def test_garbage_reply_falls_back_to_rules(self):
        class Garbage:
            label = "garbage:x"

            def complete(self, s, u):
                from promo_finder.llm import LLMReply
                return LLMReply("I think it's great!", "x", 0.0, 1)
        judge_candidates([self.cand], self.scorer, Garbage(), top_n=5)
        self.assertIsNone(self.cand.scores["llm"])

    def test_mock_run_end_to_end_and_ranking(self):
        other = Candidate("C-T-2", "continuous", list(self.cand.segments))
        other.scores = {"rule": 0.99}  # unjudged but higher rule score
        stats = judge_candidates([self.cand, other], self.scorer, make_client("mock"), top_n=1)
        self.assertEqual(stats["judged"], 1)
        judged = other if other.scores["llm"] is not None else self.cand
        unjudged = self.cand if judged is other else other
        self.assertLess(rank_key(judged), rank_key(unjudged))  # checked promos rank first

    def test_cache_avoids_asking_twice(self):
        import tempfile
        from promo_finder.llm import LLMReply

        class Counting:
            label = "counting:x"
            calls = 0

            def complete(self, s, u):
                Counting.calls += 1
                return LLMReply('{"hook": 8, "cliffhanger": 7, "standalone": 9, "verdict": "post", "why": "ok"}',
                                "x", 0.1, 1)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.jsonl"
            s1 = judge_candidates([self.cand], self.scorer, Counting(), top_n=1, cache_path=cache)
            first_llm = self.cand.scores["llm"]
            s2 = judge_candidates([self.cand], self.scorer, Counting(), top_n=1, cache_path=cache)
            self.assertEqual(Counting.calls, 1)                 # second run asked nobody
            self.assertEqual((s1["cached"], s2["cached"]), (0, 1))
            self.assertEqual(self.cand.scores["llm"], first_llm)

    def test_mock_reply_is_json(self):
        self.assertIn("hook", extract_json(mock_reply('[E01-B001] A: "hello there"')))


if __name__ == "__main__":
    unittest.main(verbosity=2)
