"""Offline tests (no internet, fake LLM): tools, verification, dates, ASR cleanup, agent loop, scoring."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.kb import KB, spoken_sku
from src.tools_layer import Tools, Session
from src.agent import Agent, normalize_asr, speakable
from src import evaluate as ev

kb = KB(); tools = Tools(kb)
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m) or (c or sys.exit(1))

# ---- date rules (README: business days Mon-Sat, skip holidays; start day not counted)
ok(tools.cal.calc("add_business_days", "2026-10-17", 5)["result"] == "2026-10-24", "17 Oct + 5 business days = 24 Oct")
ok(tools.cal.calc("add_business_days", "2026-10-17", 7)["result"] == "2026-10-27", "17 Oct + 7 business days = 27 Oct")
ok(tools.cal.calc("add_calendar_days", "2026-10-12", 15)["result"] == "2026-10-27", "12 Oct + 15 calendar days = 27 Oct")

# ---- ASR cleanup + spoken SKU
ok(normalize_asr("refund two zero two six ho gaya") == "refund 2026 ho gaya", "spoken digits -> 2026")
ok(normalize_asr("order eight nine seven eight two eight three seven") == "order 89782837", "spoken order id")
ok(spoken_sku("sku ele-zero zero zero to pe") == ["ELE-0002"], "ASR sku 'ele-zero zero zero to' -> ELE-0002")
ok(spoken_sku("ELE-0012 pe") == ["ELE-0012"], "typed SKU")
ok(speakable("Order 89782837 refund ₹49") == "Order 8978 2837 refund 49 rupees", "speakable()")

# ---- verification enforced in code
s = Session()
r = tools.lookup_order(s, "8978 2837", None)
ok(r.get("verification_required") and "status" not in r, "no details before verification")
r = tools.lookup_order(s, "89782837", "0000")
ok(r.get("verified") is False and "ONE retry" in r["message"], "wrong value -> one retry allowed")
r = tools.lookup_order(s, "89782837", "1111")
ok(r.get("verified") is False and "Do NOT" in r["message"], "second failure -> no details")
r = tools.lookup_order(s, "89782837", "4533")
ok(r.get("locked"), "after 2 failures the order stays locked even with the right value")
s = Session()
r = tools.lookup_order(s, "89782837", "4533")
ok(r.get("verified") and "delivery_pincode" not in r and r["items"], "right value -> details, pincode hidden")
print("   hints:", r["computed_hints"])
t = tools.create_ticket(s, {"summary": "Customer verified with 4533 reports damaged TV worth over 10000", "order_id": "89782837"})
ok(t.get("created"), "ticket created")
import csv
rows = list(csv.DictReader(open(config.dataset_dir()/"tools"/"tickets_created.csv", encoding="utf-8")))
ok("4533" not in rows[-1]["summary"], "verification value redacted from ticket")
ok(tools.create_ticket(Session(), {"summary": "short"}).get("created") is False, "short summary rejected")

# ---- scoring helpers
ok(ev.facts_ok("2026-10-27 / 27 October 2026 ; 49", "Deadline is 27th of October 2026 and fee Rs 49") == [], "date + number facts")
ok(ev.facts_ok("instant / 2 hours", "Refund is instant") == [], "alternatives")
ok(ev.facts_ok("12 ; My Orders / app", "12 months warranty, claim in the app") == [], "mixed")
ok(ev.facts_ok("2026-10-27", "27 अक्टूबर 2026 तक") == [], "hindi date")
ok(ev.forbidden_hit("5 to 7 days", "refund takes 5 to 7 days") != [], "forbidden detected")

# ---- agent loop with a scripted fake LLM
class Fake:
    def __init__(self, script): self.script, self.i = script, 0; self.seen = []
    def chat(self, system, messages, **kw):
        if kw.get("max_tokens") == 80: return "return window deadline"
        self.seen.append(messages[0]["content"]); out = self.script[self.i]; self.i += 1; return json.dumps(out)
    def json(self, system, messages, **kw):
        from src.llm import extract_json; return extract_json(self.chat(system, messages, **kw))
script = [
    {"action": "reply", "reply": "Hello, I'm ZentroMart's AI assistant. What is your order ID?", "decision": "clarify", "language": "en"},
    {"action": "lookup_order", "args": {"order_id": "89782837"}},
    {"action": "reply", "reply": "Thanks. Please tell me the last four digits of your registered phone number.", "decision": "clarify"},
    {"action": "lookup_order", "args": {"order_id": "89782837", "verification_value": "4533"}},
    {"action": "reply", "reply": "Thanks, you are verified.", "decision": "answer", "used_chunks": ["IDV-2"]},
    {"action": "reply", "reply": "A colleague will contact you about the replacement.", "decision": "escalate", "handover_summary": "Customer reports damaged TV over 10000, wants replacement approved. Verified. ESC-1.1."},
]
fake = Fake(script); ag = Agent(llm=fake, kb=kb)
t1 = ag.respond("Hello, mujhe order ke baare mein help chahiye.");  ok(t1.decision == "clarify", "turn1 clarify")
t2 = ag.respond("Order number hai 8978 2837.");  ok([c["name"] for c in t2.tool_calls] == ["lookup_order"] and t2.decision == "clarify", "turn2 lookup (unverified) + ask")
t3 = ag.respond("Last four digits 4533 hain.");  ok(ag.session.verified and t3.tool_calls[0]["args"]["verification_value"] == "[provided]", "turn3 verified, value masked in log")
ok("4533" not in json.dumps(t3.tool_calls), "verification value not in tool log")
t4 = ag.respond("Replacement approve kar do");  ok(t4.decision == "escalate" and ag.session.tickets, "turn4 escalate => ticket really created (code guard)")
ok("delivery_pincode" not in fake.seen[-1] and "4533" not in fake.seen[-1].split("CONVERSATION SO FAR")[0], "LLM prompt has no pincode/phone digits in order data")
ok("customer: Last four digits 4533 hain." in fake.seen[-1], "conversation memory is passed to the LLM")

# ---- eval harness end to end with a trivial fake (checks wiring only)
class Always:
    def chat(self, *a, **k): return json.dumps({"reply": "Sorry, I don't have information about that. I can connect you to a colleague.", "decision": "answer", "language": "en"})
    def json(self, *a, **k): return json.loads(self.chat())
kb2 = kb
import src.evaluate as E
orig = E.Agent
E.Agent = lambda mode, kb: orig(llm=Always(), kb=kb, mode=mode)
s, res = E.run_questions("rag", limit=10, kb=kb2, out_dir="/tmp/claude-0/-home-claude/8e0a0709-6bb2-5e90-ad48-0e7e984bf8d0/scratchpad/evalout")
print("   eval wiring:", s); ok(s["questions"] == 10, "eval harness runs")
s, res = E.run_conversations(limit=3, kb=kb2, out_dir="/tmp/claude-0/-home-claude/8e0a0709-6bb2-5e90-ad48-0e7e984bf8d0/scratchpad/evalout")
print("   conv wiring:", s); ok(s["turns"] > 0, "conversation harness runs")
print("\nALL OFFLINE TESTS PASSED")
