"""Web demo for Promo Finder (Streamlit).

  streamlit run app.py

- "Demo show" runs on a short ORIGINAL show bundled in demo/ (the real input scripts are
  confidential and are never part of this app or repo).
- "Upload" runs on your own scripts (.zip or .txt files). Files are processed in a temporary
  folder that is deleted after the run; nothing is stored.
- The LLM step is optional. A pasted API key is used only for your run: it is handed straight
  to your own client object (never put in an environment variable shared with other visitors).
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

import eval as evaluator
from promo_finder.llm import PRESETS, LLMError, make_client
from promo_finder.parser import load_show
from run import run_show

ROOT = Path(__file__).parent
DEMO_SHOW = ROOT / "demo" / "the_contract_bride"
REPO_URL = "https://github.com/Sujalmaheshwari15/promo-clip-finder"

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")

# ---------------------------------------------------------------- header

st.title("🎬 Promo Finder")
st.markdown(
    "Finds the best **20–60 second promo clips** in microdrama screenplays: it opens on a **hook** in the first "
    "3 seconds, ends on a **cliffhanger**, and should make sense to a new viewer. Every timestamp comes from the "
    "script; the optional AI judge may only point at beat IDs, never write times. "
    f"[Source code & README]({REPO_URL})"
)
st.caption("On the real 50-episode show from the assignment (confidential, not included here) it found "
           "**214 distinct promos from 17,612 candidate cuts** and passed all 8 quality checks. "
           "The demo below is a short, made-up 3-episode show, so it yields only a handful.")

# ---------------------------------------------------------------- sidebar: input + settings

with st.sidebar:
    st.header("1 · Input")
    source = st.radio("Scripts", ["Demo show (made-up, 3 episodes)", "Upload my own scripts"], index=0)
    uploads = None
    if source.startswith("Upload"):
        uploads = st.file_uploader("A .zip of episode scripts, or several .txt files",
                                   type=["zip", "txt"], accept_multiple_files=True)
        st.caption("Processed in a temporary folder and deleted after the run. Nothing is stored.")

    st.header("2 · AI judge (optional)")
    provider = st.selectbox(
        "Provider",
        ["none (rules only - instant)", "mock (offline demo of the AI step)", "groq", "gemini", "openrouter",
         "openai", "anthropic", "custom (OpenAI-compatible URL)"],
        index=0,
        help="Rules alone already rank every promo. The AI judge re-scores the top N as a new viewer would.")
    provider_key = provider.split(" ")[0]
    api_key, model, base_url = "", "", ""
    if provider_key in ("groq", "gemini", "openrouter", "openai", "anthropic", "custom"):
        api_key = st.text_input("Your API key", type="password",
                                help="Used only for this run; not stored or logged.")
        default_model = PRESETS[provider_key].model if provider_key in PRESETS else ""
        model = st.text_input("Model", value=default_model)
        if provider_key == "custom":
            base_url = st.text_input("Base URL (…/v1)", placeholder="https://host/v1")
    top_n, min_interval = 10, 3.0
    if provider_key != "none":   # these only matter when an AI judge is used
        top_n = st.slider("How many of the top promos the AI re-scores", 3, 60, 10,
                          help="This does NOT create promos - it picks how many of the promos found get an AI "
                               "second opinion (one call each). Free tiers rate-limit; 10 takes about a minute.")
        if provider_key != "mock":
            min_interval = st.number_input("Seconds between AI calls", 0.0, 30.0, 3.0, 1.0,
                                           help="Spaces out calls to stay under free-tier rate limits.")

    st.header("3 · How many promos")
    with st.expander("Advanced"):
        max_overlap = st.slider("Max shared footage between two promos", 0.20, 0.80, 0.35, 0.05,
                                help="The main limit on how many promos you get. Higher = more promos, "
                                     "but more of them are near-copies of each other.")
        per_cont = st.number_input("Continuous promos per episode (max)", 1, 12, 4)
        per_stitch = st.number_input("Stitched promos per episode (max)", 0, 8, 2)
        st.caption("Defaults (35%, 4 + 2) keep every promo genuinely different. Short episodes fit only a few "
                   "distinct 20-60 s clips.")

    run_clicked = st.button("▶ Find promos", type="primary", width="stretch")

# ---------------------------------------------------------------- run


def prepare_input(tmp: Path) -> tuple[Path | None, str]:
    if source.startswith("Demo"):
        return DEMO_SHOW, "The Contract Bride (demo)"
    if not uploads:
        st.warning("Upload a .zip or some .txt scripts first - or pick the demo show.")
        return None, ""
    zips = [u for u in uploads if u.name.lower().endswith(".zip")]
    if zips:
        path = tmp / "input.zip"
        path.write_bytes(zips[0].getvalue())
        return path, Path(zips[0].name).stem
    folder = tmp / "input"
    folder.mkdir()
    for u in uploads:
        (folder / Path(u.name).name).write_bytes(u.getvalue())
    return folder, "Uploaded show"


def make_llm_client():
    if provider_key == "none":
        return None
    if provider_key == "mock":
        return make_client("mock", min_interval=0)
    if not api_key:
        st.info("No API key entered - running on rules only.")
        return None
    try:
        if provider_key == "custom":
            client = make_client("custom", model=model, base_url=base_url, api_key_env="",
                                 min_interval=min_interval, max_retries=3, timeout=90)
        else:
            client = make_client(provider_key, model=model or None, api_key_env="",
                                 min_interval=min_interval, max_retries=3, timeout=90)
        client.api_key = api_key.strip().strip('"')   # per-visitor key, never via os.environ
        return client
    except LLMError as e:
        st.error(f"AI judge unavailable ({e}) - running on rules only.")
        return None


if run_clicked:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        in_path, show_name = prepare_input(tmp)
        if in_path:
            client = make_llm_client()
            out_dir = tmp / "out"
            log_box = st.status("Running…", expanded=True)
            lines: list[str] = []

            def say(msg: str) -> None:
                lines.append(msg)
                log_box.write(msg)

            try:
                result, meta, paths, llm_log = run_show(
                    in_path, out_dir, client, top=top_n, use_cache=False, say=say, show_name=show_name,
                    quota={"continuous": int(per_cont), "stitched": int(per_stitch)}, max_overlap=max_overlap)
                with contextlib.redirect_stdout(io.StringIO()):
                    evaluator.main(["--out", str(out_dir), "--input", str(in_path)])
                files = {p.name: p.read_bytes() for p in sorted(out_dir.iterdir()) if p.is_file()}
                episodes = {e.number: e for e in load_show(in_path)}
                st.session_state["run"] = {
                    "meta": meta,
                    "promos": json.loads(files["promos.json"])["promos"],
                    "files": files,
                    "episodes_events": {n: [(ev.kind, ev.speaker, ev.text, ev.start, ev.is_end_marker)
                                            for ev in e.events] for n, e in episodes.items()},
                }
                log_box.update(label=f"Done - {len(result.promos)} promos from {meta['episodes']} episodes",
                               state="complete", expanded=False)
            except Exception as e:  # show the problem instead of a blank page
                log_box.update(label="Run failed", state="error")
                st.exception(e)

# ---------------------------------------------------------------- results

run = st.session_state.get("run")
if not run:
    st.info("Pick the **demo show** (or upload scripts) in the sidebar and press **▶ Find promos**. "
            "Rules-only runs take a few seconds.")
    st.stop()

meta, promos, files = run["meta"], run["promos"], run["files"]
eval_md = files.get("eval.md", b"").decode("utf-8")
passed = "FAIL" not in eval_md.split("## Soft checks")[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Episodes", meta["episodes"])
c2.metric("Candidate cuts", f"{meta['candidates']:,}")
c3.metric("Distinct promos", meta["promos"])
c4.metric("AI-judged", meta["llm"].get("judged", 0))
c5.metric("Checks passed", "8/8" if passed else "✗")
st.caption(f"Show: **{meta['show']}** · AI judge: {meta['llm']['label']} · "
           f"{meta['runtime_minutes']} min of show")
if meta["llm"].get("provider") == "mock":
    st.warning("**Mock AI judge:** it returns *random* scores and verdicts. It only shows how the AI step "
               "plugs in (validation, blending, ranking). For real judgements pick a provider and paste a key, "
               "or choose *none* to see the rule-based ranking alone.")


def clip_text(p: dict) -> list[str]:
    out = []
    for i, seg in enumerate(p["segments"]):
        if i:
            out.append("*— cut to a later moment —*")
        for kind, speaker, text, start, end_marker in run["episodes_events"][seg["episode"]]:
            if seg["start_sec"] - 0.05 <= start < seg["end_sec"] and not end_marker:
                if kind == "DIALOGUE":
                    out.append(f"**{speaker.title()}:** “{text}”")
                elif kind in ("ACTION", "SFX"):
                    out.append(f"*[{kind.lower()}] {text}*")
    return out


curves = json.loads(files["tension_curves.json"]) if "tension_curves.json" in files else {}


def tension_chart_spec(p: dict, ep: int, curves: dict) -> dict | None:
    """Vega-Lite spec: one episode's tension curve, the promo's segment(s) shaded, the cut as a rule."""
    curve = curves.get(str(ep))
    if not curve:
        return None
    segs = [s for s in p["segments"] if s["episode"] == ep]
    last = p["segments"][-1]
    layers = [
        {"data": {"values": [{"s": s["start_sec"], "e": s["end_sec"]} for s in segs]},
         "mark": {"type": "rect", "color": "#e07a5a", "opacity": 0.25},
         "encoding": {"x": {"field": "s", "type": "quantitative"}, "x2": {"field": "e"}}},
        {"data": {"values": [{"t": t, "v": v} for t, v in curve["points"]]},
         "mark": {"type": "line", "strokeWidth": 2},
         "encoding": {"x": {"field": "t", "type": "quantitative", "title": f"Episode {ep}, seconds"},
                      "y": {"field": "v", "type": "quantitative", "title": "tension",
                            "scale": {"domain": [0, 1]}},
                      "tooltip": [{"field": "t", "title": "sec"}, {"field": "v", "title": "tension"}]}},
    ]
    if last["episode"] == ep:
        layers.append({"data": {"values": [{"t": last["end_sec"]}]},
                       "mark": {"type": "rule", "color": "#e07a5a", "strokeWidth": 2},
                       "encoding": {"x": {"field": "t", "type": "quantitative"}}})
    return {"height": 140, "width": "container", "layer": layers}


def show_chart(spec: dict) -> None:
    try:
        st.vega_lite_chart(spec, width="stretch")          # newer Streamlit
    except TypeError:
        st.vega_lite_chart(spec, use_container_width=True)  # older Streamlit


tab_top, tab_all, tab_quality, tab_files = st.tabs(["🏆 Top promos", "📋 All promos", "✅ Quality checks",
                                                    "⬇️ Download outputs"])

with tab_top:
    st.subheader("Post in this order")
    for p in promos[:10]:
        s = p["scores"]
        segs = " + ".join(f"Ep{x['episode']} {x['start']}–{x['end']}" for x in p["segments"])
        verdict = f" · AI: **{p['verdict']}**" if p.get("verdict") else ""
        with st.expander(f"#{p['rank']} · {segs} · {p['duration']:.0f}s · {p['shape']} · "
                         f"score {s.get('final') or s['rule']:.2f}", expanded=p["rank"] <= 2):
            left, right = st.columns([3, 2])
            with left:
                st.markdown(f"**Hook (first 3 s):** {p['hook']}")
                st.markdown(f"**Cliffhanger:** {p['cliffhanger']}")
                st.markdown(f"**Why it works:** {p['why']}{verdict}")
                if p["flags"]:
                    st.markdown("**Flags:** " + "; ".join(f"⚠ {f}" for f in p["flags"]))
            with right:
                st.markdown(f"Hook **{s['hook']:.2f}** · Cliffhanger **{s['cliffhanger']:.2f}** · "
                            f"Standalone **{s['standalone']:.2f}**")
                llm = f"{s['llm']:.2f}" if s.get("llm") is not None else "—"
                st.markdown(f"Rules **{s['rule']:.2f}** · AI **{llm}** · Final **{s.get('final') or s['rule']:.2f}**")
            if curves:
                ts = s.get("tension_shape")
                why = (p["reasons"].get("tension_shape") or [""])[0]
                label = (f"**{ts:.2f}** · {why}" if ts is not None
                         else "not scored (episode end - the cliffhanger score decides)" if "tension_shape" in s
                         else "not scored (run with tension off)")
                st.markdown(f"**Tension curve** (clip shaded, cut = vertical line) · tension shape {label}")
                for ep in p["episodes"]:
                    spec = tension_chart_spec(p, ep, curves)
                    if spec:
                        show_chart(spec)
            st.markdown("**The clip, as the viewer sees it:**")
            st.markdown("  \n".join(clip_text(p)))

with tab_all:
    rows = list(csv.DictReader(io.StringIO(files["promos.csv"].decode("utf-8-sig"))))
    st.dataframe(
        [{"#": int(r["rank"]), "promo": r["id"], "type": r["shape"], "sec": float(r["duration"]),
          "segments": r["segments"], "hook": r["hook"], "cliffhanger": r["cliffhanger"],
          "final": float(r["final"] or r["rule"]), "rules": float(r["rule"]),
          "AI": float(r["llm"]) if r["llm"] else None, "flags": r["flags"]} for r in rows],
        width="stretch", hide_index=True, height=520)

with tab_quality:
    st.markdown(eval_md.replace("# Evaluation", "### Evaluation", 1))
    if "report.md" in files:
        with st.expander("Full report (coverage by episode, where it picks badly)"):
            st.markdown(files["report.md"].decode("utf-8"))

with tab_files:
    st.markdown("Same files the command-line tool writes to `outputs/`.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    st.download_button("⬇️ All outputs (.zip)", buf.getvalue(), "promo_outputs.zip", "application/zip",
                       type="primary")
    for name in ("report.html", "report.md", "promos.csv", "promos.json", "eval.md", "run_meta.json",
                 "llm_raw.jsonl"):
        if name in files:
            st.download_button(name, files[name], name, key=f"dl-{name}")
