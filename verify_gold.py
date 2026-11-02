"""goldset.csv doğrulaması: her cevaplanabilir sorunun cevabı, işaretli sayfanın İNDEKSTEKİ metninde (chunks.jsonl) geçiyor mu?
Kullanım: python verify_gold.py --index index_bge --gold goldset.csv
"""
import argparse, csv, json, re, collections
ap = argparse.ArgumentParser(); ap.add_argument("--index", required=True); ap.add_argument("--gold", required=True)
a = ap.parse_args()
norm = lambda s: re.sub(r"\s+", " ", s).strip().rstrip(".").lower()
pg = collections.defaultdict(str)
for l in open(f"{a.index}/chunks.jsonl", encoding="utf-8"):
    c = json.loads(l); pg[(c["doc"], int(c["page"]))] += " " + c["text"]
bad = 0; n = 0
for r in csv.DictReader(open(a.gold, encoding="utf-8-sig")):
    if r["answerable"].strip() == "0": continue
    n += 1
    pages = [int(x) for x in re.split(r"[;,\s]+", r["gold_pages"]) if x]
    ok = any(norm(r["answer"]) in norm(pg.get((r["gold_doc"], p), "")) for p in pages)
    if not ok:
        bad += 1; print("EŞLEŞMEDİ", r["id"], r["gold_doc"][:30], r["gold_pages"], "|", r["answer"][:70])
print(f"{n} cevaplanabilir soru, {bad} eşleşmeyen")
