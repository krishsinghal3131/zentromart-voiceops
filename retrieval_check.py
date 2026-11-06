import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kb import KB
from pathlib import Path
ds = Path(os.environ["DS"])
kb = KB(ds)
rows = list(csv.DictReader(open(ds/"practice/questions.csv", encoding="utf-8")))
tot = hit = full = 0
misses = []
for r in rows:
    gold = [g.strip() for g in r["gold_evidence"].split(";") if g.strip()]
    if not gold: continue
    ids = [c.id for c in kb.retrieve(r["question"], k=int(os.environ.get("K", 8)))] + ["KB-00.1","KB-00.3"]
    ok = [g in ids for g in gold]
    tot += len(gold); hit += sum(ok); full += all(ok)
    if not all(ok): misses.append((r["id"], r["language"], r["question"][:70], [g for g,o in zip(gold,ok) if not o], ids[:6]))
print(f"gold-chunk recall {hit}/{tot} = {hit/tot:.2f};  questions fully covered {full}/{sum(1 for r in rows if r['gold_evidence'].strip())}")
for m in misses[:25]: print(m)
