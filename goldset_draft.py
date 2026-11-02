"""Kontrol kümesi taslağı: dizinden rastgele sayfalar seçer, LLM'e soru-cevap önerttirir.
Çıktı verified=0'dır; her satırı sayfayla elle doğrulayıp verified=1 yapmadan hiçbir sonuç raporlanmaz.
Cevapsız (answerable=0) sorular elle eklenmelidir (şablondaki örnek satırlar gibi)."""
import argparse
import csv
import random
from collections import defaultdict

from common import GOLD_COLUMNS, load_jsonl, sha
from llm import Cache, GroqLLM, StubLLM, parse_json

P = """Aşağıdaki Sayıştay rapor sayfasından, yalnızca bu sayfadaki bilgiyle yanıtlanabilen TEK bir somut soru ve kısa cevabı üret.
Cevap sayfada aynen geçen kısa bir ifade veya sayı olsun. Yalnızca JSON döndür: {{"soru": "...", "cevap": "..."}}

Sayfa:
{text}"""
ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--llm", choices=["groq", "stub"], default="groq")
ap.add_argument("--model", default="openai/gpt-oss-120b")
ap.add_argument("--dev-frac", type=float, default=0.3)
ap.add_argument("--out", default="goldset_draft.csv")
a = ap.parse_args()
pages = defaultdict(list)
for c in load_jsonl(f"{a.index}/chunks.jsonl"):
    pages[(c["doc"], c["page"])].append(c["text"])
keys = [k for k, v in pages.items() if sum(map(len, v)) > 400]
random.Random(42).shuffle(keys)
llm = StubLLM() if a.llm == "stub" else GroqLLM(a.model, Cache("goldset_cache.jsonl"))
rows = []
for doc, pg in keys[:a.n]:
    j = parse_json(llm.complete(P.format(text=" ".join(pages[(doc, pg)])[:3500])))
    if not j or "soru" not in j:
        continue
    split = "dev" if int(sha(f"{doc}{pg}")[:8], 16) % 100 < a.dev_frac * 100 else "test"
    rows.append({"id": f"q{len(rows) + 1:03d}", "question": j["soru"], "answer": j["cevap"], "gold_doc": doc,
                 "gold_pages": pg, "answerable": 1, "qtype": "kisa", "split": split, "verified": 0})
with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=GOLD_COLUMNS)
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} taslak soru -> {a.out} (hepsi verified=0; elle doğrula)")
