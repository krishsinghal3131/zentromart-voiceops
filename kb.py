"""Knowledge base: chunk kb/*.md by [SECTION-ID], index with BM25, plus product catalog rows."""
import csv
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from . import config

STOP = set("""a an the is are was were be been am do does did of to in on at for from by with and or but if
so than that this these those it its i me my we our you your can could would should will shall may might
please what which who whom how when where why not no yes about as into out up down over under again then
there here also just very hi hello thanks thank get got have has had""".split())

DEVA = "ऀ-ॿ"
# Doc type precedence from KB-00.1: policy > agent guide > help > faq > marketing
TYPE_WEIGHT = {"policy": 1.2, "agent_guide": 1.05, "help": 1.0, "faq": 0.9, "marketing": 0.75, "meta": 0.8}
DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "to": "2", "too": "2", "three": "3",
               "four": "4", "for": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}


def stem(t):
    if t.isascii() and len(t) > 4:
        for suf in ("ing", "ed", "es", "s"):
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                return t[: -len(suf)]
    return t


def tokenize(text):
    toks = re.findall(rf"[\w{DEVA}]+", (text or "").lower())
    return [stem(t) for t in toks if t not in STOP and len(t) > 1 or t.isdigit()]


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    doc_type: str
    status: str
    effective_from: str
    heading: str
    text: str
    source: str
    score: float = 0.0
    tokens: list = field(default_factory=list, repr=False)

    def render(self):
        head = f"[{self.id}] {self.heading} (doc: {self.title}; type: {self.doc_type}; " \
               f"status: {self.status}; effective_from: {self.effective_from})"
        return head + "\n" + self.text.strip()


def _parse_doc(path):
    raw = path.read_text(encoding="utf-8")
    meta, body = {}, raw
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip()
        body = m.group(2)
    return meta, body


def load_chunks(kb_dir):
    chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        meta, body = _parse_doc(path)
        parts = re.split(r"^## \[([^\]]+)\]\s*(.*)$", body, flags=re.M)
        # parts = [preface, id1, heading1, text1, id2, heading2, text2, ...]
        for i in range(1, len(parts) - 2, 3):
            cid, heading, text = parts[i].strip(), parts[i + 1].strip(), parts[i + 2].strip()
            chunks.append(Chunk(
                id=cid, doc_id=meta.get("doc_id", path.stem), title=meta.get("title", path.stem),
                doc_type=meta.get("doc_type", ""), status=meta.get("status", "current"),
                effective_from=meta.get("effective_from", ""), heading=heading,
                text=text, source=path.name))
    return chunks


def load_products(data_dir):
    rows = {}
    with open(data_dir / "products.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["sku"]] = r
    return rows


class BM25:
    def __init__(self, docs, k1=1.4, b=0.7):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(1, len(docs))
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query_tokens, i):
        d, s = self.docs[i], 0.0
        for t in set(query_tokens):
            f = d.get(t, 0)
            if f:
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s

# Customers rarely use the documents' words. These generic paraphrase hints (NOT answers) widen the search;
# the LLM query-rewrite in agent.py does the heavy lifting, this is the offline safety net.
PARAPHRASES = [
    (r"send (it )?back|give back|take back|wapas|vapas|लौटा|वापस|लौट|रिटर्न|return", "return window days policy"),
    (r"fit|size|don.?t like|not happy|change of mind|पसंद", "return change of mind window"),
    (r"opened|open kar|used|tried|खोल|इस्तेमाल", "condition unopened seal return"),
    (r"cancel|कैंसिल|रद्द|stop (it|my order)|on its way|shipped ho|शिप", "cancel order before shipped cancellation"),
    (r"refund|paise|पैसे|रिफ़ंड|रिफंड|money back|vapas aayega|kab tak aa", "refund timeline business days method"),
    (r"upi|card|net ?banking|wallet|cod|cash on delivery|वॉलेट", "refund payment method timeline"),
    (r"late|delay|nahi aaya|not arrived|hasn.?t arrived|didn.?t arrive|देर|नहीं आया|ghar nahi|still waiting", "delivery delay late promised date"),
    (r"track|kahan hai|where is my|ट्रैक|status", "track order status shipping"),
    (r"damage|broken|torn|crack|defect|टूट|खराब|kharab|toota|wrong item|galat|missing|nahi mila|not received item", "damaged defective wrong missing item report 48 hours photos"),
    (r"pickup|pick up|collect|nobody came|koi nahi aaya|reverse", "reverse pickup missed fee"),
    (r"human|agent|manager|insaan|executive|supervisor|complaint|consumer court|legal|police|lawyer|वकील|शिकायत", "escalation human handover legal"),
    (r"never (placed|ordered)|didn.?t (place|order)|not me|fraud|unauthori[sz]ed|hacked|नहीं किया", "order not placed by customer escalate account activity"),
    (r"battery|swell|burn|smell|spark|smoke|injur|आग|जल", "product safety risk swelling battery burning escalate"),
    (r"warranty|guarantee|वारंटी|assured|claim|repair", "warranty assured claim months"),
    (r"plus|member|मेंबर|express free", "zentromart plus membership benefits"),
    (r"coupon|promo|discount|offer|code|कूपन|student", "coupon discount offer terms"),
    (r"cod|cash on delivery|pincode|pin code|serviceable|deliver to|deliver kar", "cod availability pincode serviceable delivery"),
    (r"express|same day|next day|slot|midnight|grocery", "delivery express slot standard timeline"),
    (r"seller|marketplace|fulfilled", "marketplace seller returns warranty"),
    (r"verify|verification|last (4|four)|otp|phone number", "verify identity last four digits pincode"),
    (r"compensation|goodwill|₹ ?\d+ right now|extra|reward", "compensation not allowed policy handover"),
    (r"kitne din|how (many|long)|kab tak|कितने|कब तक|deadline|window", "days window timeline"),
]
CATEGORY_WORDS = {
    "electronics": "electronics accessories return window", "fashion": "clothing footwear bags return window reverse pickup fee",
    "home_kitchen": "home kitchen furniture return window", "beauty_sealed": "beauty personal care unopened seal return",
    "books_media": "books stationery return window", "grocery": "grocery fresh perishables not returnable",
    "innerwear": "innerwear lingerie swimwear not returnable", "digital": "gift cards digital codes not returnable",
}


def expand_query(query):
    extra = [words for pat, words in PARAPHRASES if re.search(pat, query, re.I)]
    return query + " " + " ".join(extra)


def spoken_sku(text):
    """Find SKUs such as ELE-0012, also when speech-to-text wrote 'ele zero zero one two' / 'ele-zero zero zero to'."""
    found = [m.upper() for m in re.findall(r"\b([A-Za-z]{3}-\d{4})\b", text)]
    words = re.findall(r"[A-Za-z]+|\d", text.lower())
    for i, w in enumerate(words):
        if len(w) == 3 and w.isalpha() and w not in DIGIT_WORDS:
            digits = []
            for nxt in words[i + 1: i + 5]:
                d = DIGIT_WORDS.get(nxt, nxt if nxt.isdigit() else None)
                if d is None:
                    break
                digits.append(d)
            if len(digits) == 4:
                found.append(f"{w.upper()}-{''.join(digits)}")
    return list(dict.fromkeys(found))


class KB:
    def __init__(self, dataset=None):
        ds = dataset or config.dataset_dir()
        self.dataset = ds
        self.chunks = load_chunks(ds / "kb")
        self.by_id = {c.id: c for c in self.chunks}
        self.products = load_products(ds / "data")
        for c in self.chunks:
            c.tokens = tokenize(c.heading) * 3 + tokenize(c.title) + tokenize(c.text)
        self.index = BM25([c.tokens for c in self.chunks])

    # ---- policy chunks ----
    def search(self, query, k=6, include_superseded=False):
        q = tokenize(expand_query(query))
        scored = []
        for i, c in enumerate(self.chunks):
            if c.doc_id == "KB-00":                    # meta info is always in the system prompt
                continue
            if c.status == "superseded" and not include_superseded:
                continue
            w = TYPE_WEIGHT.get(c.doc_type, 1.0)
            s = self.index.score(q, i) * w
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        out = []
        for s, c in scored[:k]:
            c2 = Chunk(**{**c.__dict__})
            c2.score = round(s, 3)
            out.append(c2)
        return out

    # ---- catalog rows ----
    def product_chunk(self, sku):
        p = self.products.get(sku)
        if not p:
            return None
        text = (f"{p['title']} by {p['brand']}. return_category={p['return_category']}; price_inr={p['price_inr']}; "
                f"fulfilled_by={p['fulfilled_by']}; assured={p['assured']}; warranty_months={p['warranty_months']}. "
                f"{p['description']}")
        return Chunk(id=f"products.csv:{sku}", doc_id="products.csv", title="Product catalog", doc_type="catalog",
                     status="current", effective_from="", heading=p["title"], text=text, source="products.csv")

    def products_for(self, query, extra_skus=()):
        skus = list(spoken_sku(query)) + list(extra_skus)
        q = set(tokenize(query))
        for sku, p in self.products.items():
            t = set(tokenize(p["title"]))
            if len(t) >= 2 and len(t & q) / len(t) >= 0.8:
                skus.append(sku)
        out, seen = [], set()
        for s in skus:
            if s in self.products and s not in seen:
                seen.add(s)
                out.append(self.product_chunk(s))
        return out[:4]

    def retrieve(self, query, k=8, extra_skus=()):
        prods = self.products_for(query, extra_skus)
        cats = " ".join(CATEGORY_WORDS.get(self.products[p.id.split(":")[1]]["return_category"], "") for p in prods)
        found = self.search(query + " " + cats, k=k)
        if any(c.id == "RET-3" for c in found):          # RET-3 says v3 still applies to older electronics
            v3 = self.by_id.get("RETv3-2")
            if v3 and all(c.id != v3.id for c in found):
                found.append(Chunk(**{**v3.__dict__}))
        return found + prods

    def meta_text(self):
        """Precedence rules and holiday table (KB-00) for the system prompt."""
        return "\n\n".join(c.render() for c in self.chunks if c.doc_id == "KB-00")

    def agent_guides_text(self):
        return "\n\n".join(c.render() for c in self.chunks if c.doc_type == "agent_guide")
