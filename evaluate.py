"""Score the agent on practice/questions.csv and practice/conversations.csv (Step 1 vs Step 2 vs Step 3)."""
import csv
import re
import time
from collections import defaultdict
from pathlib import Path

from . import config
from .agent import Agent, normalize_asr
from .tools_layer import Session

HI_MONTHS = ["जनवरी", "फ़रवरी|फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर|सितम्बर",
             "अक्टूबर", "नवंबर|नवम्बर", "दिसंबर|दिसम्बर"]
EN_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
             "october", "november", "december"]
DEV = str.maketrans("०१२३४५६७८९", "0123456789")


def norm(t):
    t = (t or "").lower().translate(DEV).replace("₹", "")
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def date_variants(iso):
    y, m, d = (int(x) for x in iso.split("-"))
    en = EN_MONTHS[m - 1]
    out = [f"{d} {en} {y}", f"{d} {en}", f"{en} {d}", f"{d} of {en}", f"{d} {en[:3]} {y}", iso]
    out += [f"{d} {h} {y}" for h in HI_MONTHS[m - 1].split("|")] + [f"{d} {h}" for h in HI_MONTHS[m - 1].split("|")]
    return out


def alt_matches(alt, reply_n):
    alt = alt.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", alt):
        return any(v in reply_n for v in date_variants(alt))
    a = norm(alt)
    if not a:
        return False
    if re.fullmatch(r"[\d.]+", a):
        return re.search(rf"(?<![\d.]){re.escape(a)}(?![\d])", reply_n) is not None
    return a in reply_n


def facts_ok(must_include, reply):
    """' ; ' = every fact required;  ' / ' = any one alternative is enough."""
    r = norm(reply)
    missing = []
    for fact in [f for f in (must_include or "").split(" ; ") if f.strip()]:
        if not any(alt_matches(alt, r) for alt in fact.split(" / ")):
            missing.append(fact)
    return missing


def forbidden_hit(must_not, reply):
    r = norm(reply)
    hits = []
    for fact in [f for f in (must_not or "").split(" ; ") if f.strip()]:
        if any(alt_matches(alt, r) for alt in fact.split(" / ")):
            hits.append(fact)
    return hits


def ids(s):
    return [x.strip() for x in (s or "").split(";") if x.strip()]


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


AMBIENT = {"KB-00.1", "KB-00.3", "KB-00.2"}     # always in the prompt


def run_questions(mode, limit=None, asr=False, pause=0.0, out_dir=None, kb=None):
    ds = config.dataset_dir()
    rows = read(ds / "practice" / "questions.csv")[:limit]
    agent = Agent(mode=mode, kb=kb)
    res, ok_n, rec_hit, rec_tot = [], 0, 0, 0
    for r in rows:
        agent.reset()
        text = normalize_asr(r["asr_text"]) if asr else r["question"]
        try:
            t = agent.respond(text)
            err = ""
        except Exception as e:
            t, err = None, f"{type(e).__name__}: {e}"
        if t is None:
            res.append({"id": r["id"], "pass": False, "error": err})
            continue
        acceptable = set(ids(r["acceptable_decisions"])) | {r["expected_decision"]}
        dec_ok = t.decision in acceptable
        missing = facts_ok(r["must_include"], t.reply)
        bad = forbidden_hit(r["must_not_include"], t.reply)
        gold = [g for g in ids(r["gold_evidence"]) if g not in AMBIENT]
        seen = set(t.chunks_used) | set(t.chunks_retrieved)
        got = [g for g in gold if g in seen]
        rec_hit, rec_tot = rec_hit + len(got), rec_tot + len(gold)
        passed = dec_ok and not missing and not bad
        ok_n += passed
        res.append({"id": r["id"], "lang": r["language"], "pass": passed, "decision_ok": dec_ok,
                    "decision": t.decision, "expected": r["expected_decision"], "missing_facts": " ; ".join(missing),
                    "forbidden": " ; ".join(bad), "gold": " ; ".join(gold), "retrieved_gold": " ; ".join(got),
                    "chunks_used": " ; ".join(t.chunks_used), "question": text, "reply": t.reply, "error": ""})
        if pause:
            time.sleep(pause)
    n = len(rows)
    summary = {"mode": mode, "asr_text": asr, "questions": n, "passed": ok_n, "pass_rate": round(ok_n / max(1, n), 3),
               "retrieval_recall": round(rec_hit / rec_tot, 3) if (rec_tot and mode != "baseline") else None}
    _save(res, out_dir, f"eval_questions_{mode}{'_asr' if asr else ''}.csv")
    return summary, res


def _tool_names(s):
    return re.findall(r"(\w+)\(", s or "")


def run_conversations(limit=None, pause=0.0, out_dir=None, kb=None):
    ds = config.dataset_dir()
    rows = read(ds / "practice" / "conversations.csv")
    convs = defaultdict(list)
    for r in rows:
        convs[r["conversation_id"]].append(r)
    agent = Agent(mode="rag", kb=kb)
    res, turn_ok, turn_n, conv_ok = [], 0, 0, 0
    for cid, turns in list(convs.items())[:limit]:
        turns.sort(key=lambda r: int(r["turn"]))
        agent.reset()
        all_ok = True
        for r in turns:
            try:
                t = agent.respond(r["customer_message"])
                err = ""
            except Exception as e:
                t, err = None, f"{type(e).__name__}: {e}"
            turn_n += 1
            if t is None:
                res.append({"conversation": cid, "turn": r["turn"], "pass": False, "error": err})
                all_ok = False
                continue
            missing = facts_ok(r["must_include"], t.reply)
            bad = forbidden_hit(r["must_not_include"], t.reply)
            called = [c["name"] for c in t.tool_calls]
            need = _tool_names(r["expected_tool_calls"])
            tools_ok = all(n in called for n in need)
            if "verification_value" in (r["expected_tool_calls"] or ""):
                tools_ok = tools_ok and any(c["name"] == "lookup_order" and "verification_value" in c["args"]
                                            for c in t.tool_calls)
            dec_ok = t.decision == r["expected_decision"]
            passed = dec_ok and not missing and not bad and tools_ok
            turn_ok += passed
            all_ok &= passed
            res.append({"conversation": cid, "turn": r["turn"], "scenario": r["scenario"], "pass": passed,
                        "decision": t.decision, "expected": r["expected_decision"], "missing_facts": " ; ".join(missing),
                        "forbidden": " ; ".join(bad), "tools_called": " ; ".join(called),
                        "tools_expected": r["expected_tool_calls"], "customer": r["customer_message"],
                        "reply": t.reply, "chunks_used": " ; ".join(t.chunks_used), "error": ""})
            if pause:
                time.sleep(pause)
        conv_ok += all_ok
    summary = {"mode": "conversations", "conversations": min(len(convs), limit or len(convs)), "conversations_fully_passed": conv_ok,
               "turns": turn_n, "turns_passed": turn_ok, "turn_pass_rate": round(turn_ok / max(1, turn_n), 3)}
    _save(res, out_dir, "eval_conversations.csv")
    return summary, res


def _save(rows, out_dir, name):
    if not rows or not out_dir:
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(Path(out_dir) / name, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
