"""Erişim değerlendirmesi: vektör, BM25 ve hibrit (RRF) için Recall@k ve MRR@10.

Bir bölüt, belge adı ve sayfası kontrol kümesindeki doğru belge/sayfa(lar)la eşleşirse "isabet" sayılır.
Yalnızca answerable=1 ve verified=1 satırlar kullanılır.

Örnek:
  python eval_retrieval.py --index index_bge --gold goldset.csv --split test --out results/retrieval_bge
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import Retriever, bootstrap_ci, is_hit, load_gold, mcnemar_exact, wilson


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--split", choices=["dev", "test"], default=None,
                    help="ayarlar dev'de seçilir, sonuçlar test'te raporlanır")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--methods", nargs="+", default=list(Retriever.METHODS))
    ap.add_argument("--paired-k", type=int, default=10, help="eşleştirilmiş karşılaştırma için k")
    ap.add_argument("--stem5-bm25", action="store_true", help="BM25'te ilk 5 harf kökleştirme")
    ap.add_argument("--include-unverified", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gold = [g for g in load_gold(args.gold, args.split, args.include_unverified) if g["answerable"]]
    if not gold:
        raise SystemExit("HATA: uygun (verified=1, answerable=1) soru yok")
    r = Retriever(args.index, args.stem5_bm25)
    kmax = max(args.ks + [10])
    hits_at: dict[str, dict[int, list[bool]]] = {m: {k: [] for k in args.ks} for m in args.methods}
    rr: dict[str, list[float]] = {m: [] for m in args.methods}
    runs = []
    for g in gold:
        for m in args.methods:
            res = r.search(g["question"], m, kmax)
            first = next((rank for rank, (i, _) in enumerate(res, 1) if is_hit(r.chunks[i], g)), None)
            for k in args.ks:
                hits_at[m][k].append(first is not None and first <= k)
            rr[m].append(1.0 / first if first and first <= 10 else 0.0)
            runs.append({"id": g["id"], "method": m, "first_hit_rank": first,
                         "top": [f'{r.chunks[i]["doc"]}#{r.chunks[i]["page"]}' for i, _ in res[:10]]})

    summary = {"n_questions": len(gold), "split": args.split, "index_meta": r.meta, "rows": [], "mrr@10": {},
               "paired": []}
    for m in args.methods:
        lo, hi = bootstrap_ci(rr[m])
        summary["mrr@10"][m] = {"mrr": sum(rr[m]) / len(rr[m]), "ci95": [lo, hi]}
        for k in args.ks:
            h = hits_at[m][k]
            lo, hi = wilson(sum(h), len(h))
            summary["rows"].append({"method": m, "k": k, "n": len(h), "recall": sum(h) / len(h),
                                    "ci95_lo": lo, "ci95_hi": hi})
    pk = args.paired_k if args.paired_k in args.ks else max(args.ks)
    for i, a in enumerate(args.methods):
        for b in args.methods[i + 1:]:
            summary["paired"].append({"a": a, "b": b, "k": pk, **mcnemar_exact(hits_at[a][pk], hits_at[b][pk])})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary["rows"][0].keys()))
        w.writeheader()
        w.writerows(summary["rows"])
    with open(out / "runs.jsonl", "w", encoding="utf-8") as f:
        for x in runs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"n={len(gold)} soru, index={r.meta['embedder']}, chunk={r.meta['chunk_size']}")
    for m in args.methods:
        cells = "  ".join(f"R@{k}={sum(hits_at[m][k]) / len(gold):.3f}" for k in args.ks)
        print(f"{m:7s} {cells}  MRR@10={summary['mrr@10'][m]['mrr']:.3f}")
    for p in summary["paired"]:
        print(f"eşleştirilmiş {p['a']} vs {p['b']} (k={p['k']}): yalnız {p['a']}={p['a_only']}, "
              f"yalnız {p['b']}={p['b_only']}, p={p['p_value']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
