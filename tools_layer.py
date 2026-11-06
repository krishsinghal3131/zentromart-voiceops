"""Tool layer between the LLM and the dataset's tools/tools.py.

Safety rules are enforced HERE in code, not just in the prompt:
 - order details only appear after lookup_order verified the caller
 - at most 1 retry (2 attempts) per order, then no details
 - verification values are never put in tickets or logs
 - "ticket created" is only reported when create_ticket returned created=True
"""
import csv
import re
from datetime import date, datetime, timedelta

from . import config

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def _d(s):
    return datetime.strptime(str(s).strip()[:10], "%Y-%m-%d").date()


def human(d):
    return f"{d.day} {MONTHS[d.month - 1]} {d.year} ({d.strftime('%A')})"


class Calendar:
    """Business day = Monday to Saturday, skipping holidays.csv. Return windows use calendar days."""

    def __init__(self, dataset):
        with open(dataset / "data" / "holidays.csv", encoding="utf-8", newline="") as f:
            self.holidays = {_d(r["date"]) for r in csv.DictReader(f)}

    def is_business(self, d):
        return d.weekday() != 6 and d not in self.holidays

    def add_business_days(self, start, n):
        d, left = start, n
        while left > 0:
            d += timedelta(days=1)
            if self.is_business(d):
                left -= 1
        return d

    def business_days_between(self, a, b):
        """Business days after a up to and including b (negative if b is before a)."""
        if b < a:
            return -self.business_days_between(b, a)
        d, n = a, 0
        while d < b:
            d += timedelta(days=1)
            n += self.is_business(d)
        return n

    def calc(self, op, start=None, n=None, end=None):
        try:
            if op == "add_business_days":
                r = self.add_business_days(_d(start), int(n))
                return {"result": r.isoformat(), "readable": human(r)}
            if op == "add_calendar_days":
                r = _d(start) + timedelta(days=int(n))
                return {"result": r.isoformat(), "readable": human(r)}
            if op == "business_days_between":
                return {"business_days": self.business_days_between(_d(start), _d(end))}
            if op == "calendar_days_between":
                return {"calendar_days": (_d(end) - _d(start)).days}
            if op == "weekday":
                return {"readable": human(_d(start))}
        except Exception as e:
            return {"error": f"bad date_calc arguments: {e}"}
        return {"error": "op must be add_business_days | add_calendar_days | business_days_between | "
                         "calendar_days_between | weekday"}


class Session:
    """Everything remembered about one customer conversation."""

    def __init__(self):
        self.history = []            # [(role, text)]
        self.verified = {}           # order_id -> order dict (verified, safe to discuss)
        self.attempts = {}           # order_id -> failed attempts
        self.pending_order = None    # order id the customer gave, not verified yet
        self.secrets = set()         # verification values typed by the customer (never logged)
        self.tickets = []            # successful create_ticket results

    def transcript(self, n=12):
        return "\n".join(f"{r}: {t}" for r, t in self.history[-n:]) or "(no earlier messages)"


class Tools:
    def __init__(self, kb):
        config.ensure_tools_importable()
        import tools as ds_tools                      # the dataset's tools/tools.py
        self.ds_tools = ds_tools
        self.kb = kb
        self.cal = Calendar(kb.dataset)
        self.pincodes = {}
        with open(kb.dataset / "data" / "pincodes.csv", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                self.pincodes[r["pincode"]] = r
        self.today = _d(config.TODAY_ISO)

    # ---------- dispatcher ----------
    def run(self, name, args, session):
        args = args or {}
        try:
            if name == "lookup_order":
                return self.lookup_order(session, args.get("order_id"), args.get("verification_value"))
            if name == "create_ticket":
                return self.create_ticket(session, args)
            if name == "date_calc":
                return self.cal.calc(args.get("op"), args.get("start"), args.get("n"), args.get("end"))
            if name == "check_pincode":
                p = self.pincodes.get(re.sub(r"\D", "", str(args.get("pincode", ""))))
                return ({k: p[k] for k in ("city", "state", "serviceable", "cod_available", "express_available")}
                        if p else {"found": False, "note": "pincode not in ZentroMart's list"})
        except Exception as e:                       # never crash the conversation on a tool error
            return {"error": f"{type(e).__name__}: {e}"}
        return {"error": f"unknown tool {name}"}

    # ---------- lookup_order with verification enforced in code ----------
    def lookup_order(self, session, order_id, verification_value):
        oid = re.sub(r"\D", "", str(order_id or ""))
        if not oid:
            return {"error": "order_id missing"}
        if oid in session.verified:                  # already verified in this conversation
            return {"already_verified": True, **self.view(session.verified[oid])}
        if session.attempts.get(oid, 0) >= 2:
            return {"verified": False, "locked": True,
                    "message": "Verification failed twice. Do NOT share any order details. Offer general "
                               "policy help and suggest contacting support from the registered phone number."}
        session.pending_order = oid
        proof = None if verification_value in (None, "") else str(verification_value)
        res = self.ds_tools.lookup_order(oid, proof)
        if proof is None:
            return {"verification_required": True, "order_id": oid,
                    "message": "Ask the customer for the last 4 digits of the registered phone number OR the "
                               "delivery pincode (one thing at a time). Never say the expected value."}
        session.secrets.add(re.sub(r"\D", "", proof))
        if res.get("found") and res.get("verified"):
            session.verified[oid] = res
            session.pending_order = None
            return self.view(res)
        session.attempts[oid] = session.attempts.get(oid, 0) + 1
        left = 2 - session.attempts[oid]
        return {"verified": False,
                "message": ("Verification failed. Allow ONE retry: ask again for the last 4 phone digits or the "
                            "delivery pincode." if left else
                            "Verification failed again. Do NOT share order details; offer general policy help.")}

    def view(self, o):
        """What the LLM may see for a verified order: no pincode, plus computed helpers."""
        v = {k: val for k, val in o.items() if k not in ("delivery_pincode", "items")}
        pin = self.pincodes.get(o.get("delivery_pincode") or "")
        if pin:
            v["delivery_area"] = {k: pin[k] for k in ("city", "serviceable", "cod_available", "express_available")}
        items = []
        for it in o.get("items", []):
            p = self.kb.products.get(it["sku"], {})
            items.append({**{k: it[k] for k in ("line_no", "sku", "title", "qty", "unit_price_inr", "return_category")},
                          "fulfilled_by": p.get("fulfilled_by"), "assured": p.get("assured"),
                          "warranty_months": p.get("warranty_months")})
        v["items"] = items
        hints = {"today": self.today.isoformat()}
        for key, label in (("delivered_on", "days_since_delivery"), ("placed_at", "days_since_placed")):
            if o.get(key):
                hints[label] = (self.today - _d(o[key])).days
        c7 = o.get("contacts_last_7_days") or 0
        hints["repeat_contact_rule_ESC-1.4"] = ("TRIGGERED: escalate" if c7 >= 2 else "not triggered")
        if o.get("damage_sku"):
            line = next((i for i in o.get("items", []) if i["sku"] == o["damage_sku"]), None)
            if line:
                val = line["unit_price_inr"] * line["qty"]
                hints["damaged_item_value_inr"] = val
                hints["damage_over_10000_ESC-1.1"] = "TRIGGERED: escalate" if val > 10000 else "no"
        v["computed_hints"] = hints
        return v

    # ---------- create_ticket ----------
    def create_ticket(self, session, a):
        summary = str(a.get("summary") or "")
        for s in session.secrets:                    # never leak verification values
            if s:
                summary = re.sub(rf"(?<!\d){re.escape(s)}(?!\d)", "[redacted]", summary)
        oid = re.sub(r"\D", "", str(a.get("order_id") or "")) or None
        if oid and oid not in session.verified:      # unverified callers: no order link in the ticket
            summary += " (Caller identity NOT verified.)"
        res = self.ds_tools.create_ticket(summary=summary, order_id=oid,
                                          reason=a.get("reason"), intent=a.get("intent"))
        if res.get("created"):
            session.tickets.append(res)
        return res
