"""ZentroMart VoiceOps agent - one entry point.

  python run.py doctor                      check EVERYTHING at once (keys, dataset, packages, LLM, Gnani)
  python run.py chat                        talk to the agent by typing
  python run.py voice                       talk to the agent with your microphone (Gnani STT + TTS)
  python run.py batch --audio_dir DIR       the submission command: folder of audio in -> replies out
  python run.py eval [--limit N]            Step 1 vs Step 2 vs Step 3 scores on the practice files
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):                       # Windows consoles + Hindi text
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac"}


# ------------------------------------------------------------------------------------ doctor
def cmd_doctor(_):
    from src import config
    ok = True

    def line(good, msg, fix=""):
        nonlocal ok
        ok &= bool(good)
        print(("  OK   " if good else "  FAIL ") + msg + (("\n         -> " + fix) if (fix and not good) else ""))

    print("Python", sys.version.split()[0])
    config.load_env()
    env_file = config.ROOT / ".env"
    line(env_file.exists(), f".env file at {env_file}", "copy .env.example to .env and fill in your keys")
    for w in config.warnings:
        print("  NOTE ", w)

    provider = config.env("LLM_PROVIDER", "anthropic").lower()
    key_name = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    line(config.env(key_name), f"{key_name} is set (provider={provider}, model={config.env('LLM_MODEL')})",
         f"put {key_name}=... in .env")
    line(config.env("GNANI_API_KEY"), "GNANI_API_KEY is set", "put GNANI_API_KEY=... in .env")

    try:
        ds = config.dataset_dir()
        line(True, f"dataset folder found: {ds}")
    except Exception as e:
        line(False, "dataset folder", str(e))
        ds = None

    for mod, pip in (("anthropic" if provider == "anthropic" else "openai", None), ("gnani", "gnani-vachana")):
        try:
            __import__(mod)
            line(True, f"python package '{mod}' installed")
        except ImportError:
            line(False, f"python package '{mod}' missing", f"python -m pip install {pip or mod}")

    if ds:
        try:
            from src.kb import KB
            kb = KB(ds)
            line(len(kb.chunks) > 40, f"knowledge base: {len(kb.chunks)} chunks, {len(kb.products)} products")
            from src.tools_layer import Tools, Session
            t = Tools(kb)
            r = t.lookup_order(Session(), "89782837", None)
            line(r.get("verification_required"), "tools.py: lookup_order works")
        except Exception as e:
            line(False, f"knowledge base / tools: {type(e).__name__}: {e}")

    # live calls last, so every cheap problem above is shown first
    if config.env(key_name):
        try:
            from src.llm import LLM
            out = LLM().chat("You are helpful.", [{"role": "user", "content": "Say hi in 3 words."}], max_tokens=40)
            line(bool(out.strip()), f"LLM answered: {out.strip()[:60]!r}", "model returned empty text")
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:300]}"
            hint = "check the key / LLM_MODEL / LLM_BASE_URL in .env"
            if "404" in msg or "not found" in msg.lower():
                hint = "LLM_MODEL name is not valid for this provider. " + _list_models()
            line(False, "LLM call failed: " + msg, hint)
    if config.env("GNANI_API_KEY"):
        try:
            from src.voice import Voice
            Voice()
            line(True, "Gnani STT/TTS clients created")
        except Exception as e:
            line(False, f"Gnani: {type(e).__name__}: {e}", "python -m pip install gnani-vachana ; check GNANI_API_KEY")
    print("\nALL GOOD - run: python run.py chat" if ok else "\nFix the FAIL lines above, then run doctor again.")


def _list_models():
    try:
        from src.llm import LLM
        names = [m.id for m in LLM().c.models.list()]
        pick = [n.replace("models/", "") for n in names if "gemini" in n or "gpt" in n][:12]
        return "Models you can use: " + ", ".join(pick)
    except Exception:
        return "Open Google AI Studio and copy a valid model name."


# ------------------------------------------------------------------------------------ chat
def cmd_chat(a):
    from src.agent import Agent
    ag = Agent(mode=a.mode, channel="text")
    print("ZentroMart support (type 'quit' to stop, 'new' for a new customer)\n")
    while True:
        try:
            text = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit"):
            break
        if text.lower() == "new":
            ag.reset()
            print("(new conversation)")
            continue
        if not text:
            continue
        t = ag.respond(text)
        print(f"bot > {t.reply}\n      [{t.decision}] chunks: {', '.join(t.chunks_used)}"
              + (f" | tools: {', '.join(c['name'] for c in t.tool_calls)}" if t.tool_calls else "") + "\n")


# ------------------------------------------------------------------------------------ voice (microphone)
def cmd_voice(a):
    from src.agent import Agent
    from src.voice import Voice, record_wav, play
    v, ag = Voice(), Agent(mode="rag", channel="voice")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    print("Press Enter, speak for a few seconds, wait for the reply. Type q + Enter to stop.")
    while input("\n[Enter = record] ").strip().lower() != "q":
        n += 1
        wav_in = out / f"call_turn{n}_in.wav"
        print(f"   recording {a.seconds}s ...")
        record_wav(wav_in, a.seconds)
        t = v.voice_turn(ag, wav_in, out / f"call_turn{n}_reply.wav")
        print(f"   you said : {t.heard}\n   agent    : {t.reply}\n   chunks   : {', '.join(t.chunks_used)}")
        play(out / f"call_turn{n}_reply.wav")


# ------------------------------------------------------------------------------------ batch (submission command)
def cmd_batch(a):
    from src.agent import Agent
    from src.voice import Voice
    src = Path(a.audio_dir)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in AUDIO_EXT)
    if not files:
        sys.exit(f"No audio files found in {src}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v, ag = Voice(), Agent(mode="rag", channel="voice")
    rows = []
    for p in files:
        if not a.sequential:
            ag.reset()                                      # every file = a new customer
        t = v.voice_turn(ag, p, out / f"{p.stem}_reply.wav", a.lang)
        rec = {"file": p.name, "heard": t.heard, "reply": t.reply, "decision": t.decision, "language": t.language,
               "chunks_used": t.chunks_used, "chunks_retrieved": t.chunks_retrieved,
               "tool_calls": t.tool_calls, "reply_audio": f"{p.stem}_reply.wav"}
        (out / f"{p.stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / f"{p.stem}_reply.txt").write_text(t.reply, encoding="utf-8")
        rows.append({**rec, "chunks_used": " ; ".join(t.chunks_used), "chunks_retrieved": " ; ".join(t.chunks_retrieved),
                     "tool_calls": " ; ".join(c["name"] for c in t.tool_calls)})
        print(f"{p.name}\n  heard: {t.heard}\n  reply: {t.reply}\n  chunks: {', '.join(t.chunks_used)}\n")
    with open(out / "results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} replies (text + audio + chunks) in {out}")


# ------------------------------------------------------------------------------------ eval
def cmd_eval(a):
    from src import evaluate as ev
    from src.kb import KB
    kb = KB()
    out = a.out
    summaries = []
    if a.part in ("all", "questions"):
        print("Step 1: LLM only (no documents) ...")
        s1, _ = ev.run_questions("baseline", a.limit, a.asr, a.pause, out, kb)
        print(" ", s1)
        print("Step 2: LLM + RAG + tools ...")
        s2, _ = ev.run_questions("rag", a.limit, a.asr, a.pause, out, kb)
        print(" ", s2)
        summaries += [s1, s2]
    if a.part in ("all", "conversations"):
        print("Step 3: multi-turn conversations ...")
        s3, _ = ev.run_conversations(a.limit, a.pause, out, kb)
        print(" ", s3)
        summaries.append(s3)
    Path(out).mkdir(parents=True, exist_ok=True)
    Path(out, "eval_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nDetails (every reply, missing facts, chunks) are in {out}\\eval_*.csv  -> use these numbers in your report")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    c = sub.add_parser("chat")
    c.add_argument("--mode", default="rag", choices=["rag", "baseline"])
    c.set_defaults(fn=cmd_chat)
    v = sub.add_parser("voice")
    v.add_argument("--out", default="sample_outputs/live_call")
    v.add_argument("--seconds", type=int, default=8)
    v.set_defaults(fn=cmd_voice)
    b = sub.add_parser("batch")
    b.add_argument("--audio_dir", required=True)
    b.add_argument("--out", default="sample_outputs")
    b.add_argument("--lang", default=None, help="STT language code, default STT_LANGUAGE from .env")
    b.add_argument("--sequential", action="store_true", help="treat all files as ONE conversation, in name order")
    b.set_defaults(fn=cmd_batch)
    e = sub.add_parser("eval")
    e.add_argument("--part", default="all", choices=["all", "questions", "conversations"])
    e.add_argument("--limit", type=int, default=None, help="only the first N rows / conversations (quick test)")
    e.add_argument("--asr", action="store_true", help="use the asr_text column instead of question")
    e.add_argument("--pause", type=float, default=0.0, help="seconds to wait between calls (rate limits)")
    e.add_argument("--out", default="sample_outputs")
    e.set_defaults(fn=cmd_eval)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
