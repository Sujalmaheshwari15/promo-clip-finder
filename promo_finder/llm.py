"""One small, swappable LLM client over plain HTTP (urllib - no SDKs).

Design choices (short version):
- Every provider that speaks the OpenAI-style POST /chat/completions API is
  just a preset: base URL + model + the env var holding the key. Swapping
  models is a command-line flag, not a code change (--provider groq,
  --model ..., or --provider custom --base-url ... for anything new).
- Anthropic is called through its native Messages API (POST /v1/messages),
  also over plain HTTP, via a tiny adapter; the rest of the code can't tell.
- "mock" is an offline fake (deterministic, no network) for tests and demos.
  "none" means rules only.
- Free tiers rate-limit hard, so: --min-interval spaces calls out, and
  429 / 5xx / network errors are retried with exponential backoff (honouring
  Retry-After). Anything else raises LLMError - the judge catches it and
  falls back to the rule score, so a run never crashes because of the LLM.
- Keys are read from environment variables only and never logged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class LLMError(Exception):
    """Any LLM failure the caller should handle by falling back to rules.

    fatal=True means every later call will fail the same way (bad key, unknown model,
    no access), so the caller should stop calling instead of trying the next promo.
    """

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


FATAL_STATUS = {401, 403, 404}


@dataclass
class ProviderConfig:
    name: str
    style: str                    # "openai" | "anthropic" | "mock"
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""         # "" = no key needed (ollama, mock)
    temperature: float | None = 0.0   # None = don't send (some models reject it)
    json_mode: bool = True        # ask for response_format=json_object (OpenAI style)
    extra_headers: dict = field(default_factory=dict)
    extra_body: dict = field(default_factory=dict)   # provider-specific request fields


# Model names change often - every one of these can be overridden with --model.
PRESETS: dict[str, ProviderConfig] = {
    "gemini": ProviderConfig("gemini", "openai", "https://generativelanguage.googleapis.com/v1beta/openai",
                             # Nov 2026: 2.5-flash retired for new keys; 3.8-flash often 503 "high demand";
                             # 3.5-flash answered in ~4 s. Use --list-models to see current names.
                             "gemini-3.5-flash", "GEMINI_API_KEY",
                             # Thinking models can think for >60 s on a long prompt; judging one clip
                             # doesn't need that. Override with --reasoning-effort if needed.
                             extra_body={"reasoning_effort": "low"}),
    # Nov 2026: llama-3.3-70b-versatile retired on Groq; gpt-oss-120b is the largest chat model listed.
    "groq": ProviderConfig("groq", "openai", "https://api.groq.com/openai/v1",
                           "openai/gpt-oss-120b", "GROQ_API_KEY",
                           extra_body={"reasoning_effort": "low"}),  # reasoning model: keep replies quick
    "openrouter": ProviderConfig("openrouter", "openai", "https://openrouter.ai/api/v1",
                                 "meta-llama/llama-3.3-70b-instruct:free", "OPENROUTER_API_KEY",
                                 json_mode=False),  # not every free model supports it; we parse JSON anyway
    "ollama": ProviderConfig("ollama", "openai", "http://localhost:11434/v1", "llama3.1:8b", ""),
    "openai": ProviderConfig("openai", "openai", "https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY",
                             temperature=None),
    "anthropic": ProviderConfig("anthropic", "anthropic", "https://api.anthropic.com/v1", "claude-opus-5-5",
                                "ANTHROPIC_API_KEY", temperature=None, json_mode=False),
    "mock": ProviderConfig("mock", "mock", model="mock-judge-v1"),
}
PROVIDER_CHOICES = sorted(PRESETS) + ["custom", "none"]

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}

# urllib's default "Python-urllib/3.x" User-Agent is blocked by Cloudflare-fronted APIs
# (Groq returned "HTTP 403: error code: 1010"), so every request names this tool instead.
USER_AGENT = "promo-finder/1.0"

# Where trusted root certificates usually live, for Python builds that ship without any
# (e.g. MSYS2's Python on Windows loads 0 roots and can't verify any HTTPS site).
KNOWN_CA_BUNDLES = [
    os.path.join(sys.prefix, "etc", "ssl", "certs", "ca-bundle.crt"),
    os.path.join(sys.prefix, "..", "usr", "ssl", "certs", "ca-bundle.crt"),
    "C:/msys64/usr/ssl/certs/ca-bundle.crt",
    "/etc/ssl/certs/ca-certificates.crt",      # Debian/Ubuntu
    "/etc/pki/tls/certs/ca-bundle.crt",        # Fedora/RHEL
    "/etc/ssl/cert.pem",                       # macOS/Alpine
]


def build_ssl_context(ca_bundle: str | None = None) -> tuple[ssl.SSLContext, str]:
    """An SSL context that can actually verify servers. Returns (context, where the roots came from).

    Order: --ca-bundle / SSL_CERT_FILE, then Python's defaults (incl. the Windows store),
    then certifi if installed, then well-known bundle files. Verification is never turned off.
    """
    explicit = ca_bundle or os.environ.get("SSL_CERT_FILE")
    if explicit:
        return ssl.create_default_context(cafile=explicit), explicit
    ctx = ssl.create_default_context()
    if ctx.get_ca_certs():
        return ctx, "python default"
    try:
        import certifi  # optional; not required
        ctx.load_verify_locations(certifi.where())
        return ctx, f"certifi ({certifi.where()})"
    except Exception:
        pass
    for path in KNOWN_CA_BUNDLES:
        if os.path.isfile(path):
            ctx.load_verify_locations(path)
            return ctx, os.path.normpath(path)
    return ctx, "none found"


SSL_HINT = ("this Python has no trusted root certificates. Fix: run with another Python "
            "(e.g. Anaconda / python.org), or pass --ca-bundle <file.pem>, or set SSL_CERT_FILE")


def read_key(name: str) -> str:
    """Read an API key from the environment, forgiving two common Command Prompt slips:

    - `set NAME = key`  creates a variable called "NAME " (trailing space) holding " key"
    - `set NAME="key"`  stores the quotes as part of the value
    """
    value = os.environ.get(name)
    if value is None:
        for env_name, env_value in os.environ.items():
            if env_name.strip().upper() == name.upper():
                print(f"    (note: found {name} with extra spaces in its name - using it)", file=sys.stderr)
                value = env_value
                break
    return (value or "").strip().strip('"').strip("'").strip()


@dataclass
class LLMReply:
    text: str
    model: str
    latency_s: float
    attempts: int


class LLMClient:
    def __init__(self, cfg: ProviderConfig, min_interval: float = 0.0, max_retries: int = 4,
                 timeout: float = 90.0, max_tokens: int = 4000, ca_bundle: str | None = None):
        self.cfg = cfg
        self.ssl_context, self.ca_source = build_ssl_context(ca_bundle)
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._last_call = 0.0
        self.api_key = ""
        if cfg.api_key_env:
            self.api_key = read_key(cfg.api_key_env)
            if not self.api_key:
                raise LLMError(
                    f"environment variable {cfg.api_key_env} is not set (needed for {cfg.name}). "
                    f"In Command Prompt type exactly:  set {cfg.api_key_env}=YOUR_KEY   "
                    f"(no spaces around =, no quotes), in the SAME window you run this from")

    @property
    def label(self) -> str:
        return f"{self.cfg.name}:{self.cfg.model}"

    # ---- public
    def complete(self, system: str, user: str) -> LLMReply:
        if self.cfg.style == "mock":
            return LLMReply(mock_reply(user), self.cfg.model, 0.0, 1)
        url, headers, body = (self._anthropic_request if self.cfg.style == "anthropic"
                              else self._openai_request)(system, user)
        started = time.monotonic()
        data, attempts = self._post_with_retries(url, headers, body)
        text = (self._anthropic_text if self.cfg.style == "anthropic" else self._openai_text)(data)
        return LLMReply(text, data.get("model", self.cfg.model), round(time.monotonic() - started, 2), attempts)

    def list_models(self) -> list[str]:
        """Model IDs this key can use (GET {base}/models - same shape on OpenAI-style APIs and Anthropic)."""
        if self.cfg.style == "mock":
            return [self.cfg.model]
        headers = {"User-Agent": USER_AGENT, **self.cfg.extra_headers}
        if self.cfg.style == "anthropic":
            headers.update({"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        elif self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.cfg.base_url.rstrip("/") + "/models", headers=headers)
        try:
            ctx = self.ssl_context if req.full_url.startswith("https") else None
            with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMError(f"HTTP {e.code}: {' '.join(e.read().decode('utf-8', 'replace').split())[:300]}")
        except urllib.error.URLError as e:
            raise LLMError(f"network error: {e}")
        items = data.get("data") or data.get("models") or []
        return sorted(str(m.get("id") or m.get("name", "")).removeprefix("models/") for m in items)

    # ---- OpenAI-compatible /chat/completions
    def _openai_request(self, system: str, user: str):
        body = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
        }
        if self.cfg.temperature is not None:
            body["temperature"] = self.cfg.temperature
        if self.cfg.json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(self.cfg.extra_body)
        headers = {"Content-Type": "application/json", **self.cfg.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return self.cfg.base_url.rstrip("/") + "/chat/completions", headers, body

    @staticmethod
    def _openai_text(data: dict) -> str:
        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected response shape: {str(data)[:200]}")
        if not text.strip():
            raise LLMError(f"empty reply (finish_reason={choice.get('finish_reason')})")
        return text

    # ---- Anthropic native /v1/messages
    def _anthropic_request(self, system: str, user: str):
        body = {
            "model": self.cfg.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            # Short judging task: thinking can't be turned off on current Claude models,
            # so keep it cheap and quick with low effort.
            "output_config": {"effort": "low"},
            # If a safety classifier declines, let Anthropic route the request to a fallback
            # model instead of failing (server-side fallback, routed by refusal category).
            "fallbacks": "default",
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "server-side-fallback-2026-07-01",
            **self.cfg.extra_headers,
        }
        return self.cfg.base_url.rstrip("/") + "/messages", headers, body

    @staticmethod
    def _anthropic_text(data: dict) -> str:
        if data.get("stop_reason") == "refusal":
            raise LLMError(f"model refused: {data.get('stop_details')}")
        # Thinking blocks come back too (empty by default); keep only text blocks.
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        if not text.strip():
            raise LLMError(f"empty reply (stop_reason={data.get('stop_reason')})")
        return text

    # ---- transport
    def _post_with_retries(self, url: str, headers: dict, body: dict) -> tuple[dict, int]:
        payload = json.dumps(body).encode("utf-8")
        last_error = ""
        for attempt in range(1, self.max_retries + 2):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            req = urllib.request.Request(url, data=payload, headers={"User-Agent": USER_AGENT, **headers},
                                         method="POST")
            try:
                ctx = self.ssl_context if url.startswith("https") else None
                with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as resp:
                    return json.loads(resp.read().decode("utf-8")), attempt
            except urllib.error.HTTPError as e:
                body_text = e.read().decode("utf-8", errors="replace")
                detail = " ".join(body_text.split())[:400]
                last_error = f"HTTP {e.code}: {detail}"
                if e.code == 429 and DAILY_QUOTA_RE.search(body_text):
                    # A per-day free-tier cap (Gemini: 20 requests/model/day) won't reset by retrying.
                    raise LLMError("daily free-tier quota used up for this model - try tomorrow, another "
                                   "--model, or another --provider. " + last_error[:200], fatal=True)
                if e.code not in RETRY_STATUS:
                    hint = " (check --model; try --list-models)" if e.code == 404 else ""
                    if e.code == 403 and "1010" in detail:
                        hint = " (Cloudflare blocked this client - a firewall/VPN or proxy may be interfering)"
                    # Gemini reports a bad key as HTTP 400, not 401 - still fatal.
                    bad_key = e.code == 400 and "api key" in detail.lower()
                    raise LLMError(last_error + hint, fatal=e.code in FATAL_STATUS or bad_key)
                delay = (_retry_after(e.headers.get("Retry-After")) or _retry_delay_in_body(body_text)
                         or self._backoff(attempt))
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
                reason = getattr(e, "reason", e)
                if isinstance(reason, ssl.SSLError) and "unable to get local issuer" in str(reason):
                    # No trusted roots: retrying can't fix it - fail fast with a useful hint.
                    raise LLMError(f"SSL error ({reason}); roots: {self.ca_source}; {SSL_HINT}", fatal=True)
                # Other SSL hiccups (seen in practice: a Google edge server briefly presenting the
                # wrong certificate -> "Hostname mismatch") are retried. Verification stays ON.
                last_error = f"{'SSL' if isinstance(reason, ssl.SSLError) else 'network'} error: {reason}"
                delay = self._backoff(attempt)
            except json.JSONDecodeError as e:
                raise LLMError(f"response was not JSON: {e}")
            if attempt <= self.max_retries:
                # Say why we're waiting - a silent multi-minute pause looks like a hang.
                print(f"    ({last_error[:90]} - waiting {delay:.0f}s, retry {attempt}/{self.max_retries})",
                      file=sys.stderr, flush=True)
                time.sleep(delay)
        raise LLMError(f"gave up after {self.max_retries + 1} attempts; last error: {last_error}")

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(60.0, 2.0 ** attempt) + random.uniform(0, 0.5)


DAILY_QUOTA_RE = re.compile(r"PerDay|per[ _-]day|daily", re.IGNORECASE)
RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')


def _retry_delay_in_body(body: str) -> float | None:
    """Gemini puts the wait in the JSON body ("retryDelay": "34s") instead of a Retry-After header."""
    m = RETRY_DELAY_RE.search(body)
    return min(120.0, float(m.group(1)) + 1.0) if m else None


def _retry_after(value: str | None) -> float | None:
    try:
        return min(120.0, float(value)) if value else None
    except ValueError:
        return None


# ---------------------------------------------------------------- mock provider

BEAT_ID_RE = re.compile(r"\[(E\d+-B\d+)\]")
QUOTE_RE = re.compile(r'"([^"\n]{3,80})"')


def mock_reply(prompt: str) -> str:
    """A deterministic fake judge: same prompt -> same reply, no network.

    It deliberately misbehaves sometimes so the validation code gets exercised:
    ~1 in 5 replies suggests a beat ID that doesn't exist, ~1 in 5 'quotes' words
    that are not in the script, and ~1 in 10 sneaks a timestamp into its reason.
    """
    h = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
    ids = BEAT_ID_RE.findall(prompt)
    quotes = QUOTE_RE.findall(prompt)
    pick = lambda n: (h >> n) % 11  # noqa: E731
    reply = {
        "hook": pick(3), "cliffhanger": pick(9), "standalone": max(3, pick(15)),
        "verdict": ["skip", "maybe", "post"][(h >> 21) % 3],
        "why": "Mock judge: opens on conflict and cuts before the answer.",
        "hook_quote": quotes[0] if quotes else "",
        "cliffhanger_quote": quotes[-1] if quotes else "",
        "new_start_beat": None,
        "new_end_beat": None,
    }
    if (h >> 30) % 5 == 0 and ids:
        reply["new_end_beat"] = "E99-B999"                      # invalid ID
    elif (h >> 30) % 5 == 1 and len(ids) > 4:
        reply["new_start_beat"] = ids[1]                         # plausible tighten
    if (h >> 34) % 5 == 0:
        reply["cliffhanger_quote"] = "This line was never in the script"
    if (h >> 38) % 10 == 0:
        reply["why"] = "Strong slap at 00:03.20 grabs attention."  # timestamp the model shouldn't write
    return "```json\n" + json.dumps(reply) + "\n```"


# ---------------------------------------------------------------- construction / CLI


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("LLM judge")
    g.add_argument("--provider", default="none", choices=PROVIDER_CHOICES,
                   help="LLM preset; 'none' = rules only, 'mock' = offline fake")
    g.add_argument("--model", help="override the preset's model name")
    g.add_argument("--base-url", help="for --provider custom: OpenAI-compatible base URL (…/v1)")
    g.add_argument("--api-key-env", help="env var holding the API key (custom, or override a preset)")
    g.add_argument("--min-interval", type=float, default=0.0,
                   help="minimum seconds between LLM calls (free tiers: e.g. 4 for 15 req/min)")
    g.add_argument("--max-retries", type=int, default=4)
    g.add_argument("--timeout", type=float, default=90.0)
    g.add_argument("--ca-bundle", help="PEM file of trusted root certificates (if HTTPS verification fails)")
    g.add_argument("--list-models", action="store_true", help="print the models your key can use, then exit")
    g.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "off"],
                   help="for thinking models on OpenAI-style APIs; 'off' = don't send the field")


def make_client(provider: str, model: str | None = None, base_url: str | None = None,
                api_key_env: str | None = None, min_interval: float = 0.0, max_retries: int = 4,
                timeout: float = 90.0, ca_bundle: str | None = None) -> LLMClient | None:
    """Return a client, or None for 'none'. Raises LLMError for bad config (e.g. missing key)."""
    if provider == "none":
        return None
    if provider == "custom":
        if not base_url or not model:
            raise LLMError("--provider custom needs --base-url and --model")
        cfg = ProviderConfig("custom", "openai", base_url, model, api_key_env or "", json_mode=False)
    else:
        if provider not in PRESETS:
            raise LLMError(f"unknown provider {provider!r}; choose from {PROVIDER_CHOICES}")
        p = PRESETS[provider]
        cfg = ProviderConfig(p.name, p.style, base_url or p.base_url, model or p.model,
                             p.api_key_env if api_key_env is None else api_key_env,
                             p.temperature, p.json_mode, dict(p.extra_headers), dict(p.extra_body))
    return LLMClient(cfg, min_interval=min_interval, max_retries=max_retries, timeout=timeout,
                     ca_bundle=ca_bundle)


def client_from_args(args: argparse.Namespace) -> LLMClient | None:
    client = make_client(args.provider, args.model, args.base_url, args.api_key_env,
                         args.min_interval, args.max_retries, args.timeout, args.ca_bundle)
    effort = getattr(args, "reasoning_effort", None)
    if client and effort and client.cfg.style == "openai":
        if effort == "off":
            client.cfg.extra_body.pop("reasoning_effort", None)
        else:
            client.cfg.extra_body["reasoning_effort"] = effort
    return client
