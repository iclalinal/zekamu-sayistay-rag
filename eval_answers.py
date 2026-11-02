"""Yanıt değerlendirmesi: EM, F1, kaynak (atıf) doğruluğu, groundedness (LLM hakem), cevapsız sorularda çekimserlik.

Koşullar: norag (belge verilmez), vector, bm25, hybrid. Aynı model, aynı istem, temperature=0.
Yalnızca verified=1 satırlar kullanılır. Ayarları dev'de seç, sonucu test'te raporla.

Örnek:
  set GROQ_API_KEY=...
  python eval_answers.py --index index_bge --gold goldset.csv --split test --out results/answers_bge
  python eval_answers.py ... --llm stub        # ağ olmadan akış testi
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from common import (Retriever, bootstrap_ci, doc_key, exact_match, f1_score, load_gold, normalize_answer)
from llm import ABSTAIN, Cache, GroqLLM, StubLLM, parse_json

PROMPT_VERSION = "v1"
PROMPT_RAG = """Sen Sayıştay denetim raporlarını inceleyen bir asistansın.
Aşağıdaki belge parçalarını kullanarak soruyu yanıtla. Yalnızca bu parçalardaki bilgiye dayan.
Yanıtın sonuna dayandığın kaynağı [Kaynak: DOSYA_ADI, s. SAYFA] biçiminde ekle.
Yanıt belge parçalarında yoksa tam olarak şunu yaz: {abstain}
Yanıtı kısa ve doğrudan tut.

Belge parçaları:
{context}

Soru: {question}

Yanıt:"""
PROMPT_NORAG = """Sayıştay denetim raporlarıyla ilgili aşağıdaki soruyu kısa ve doğrudan yanıtla.
Kesin olarak bilmiyorsan tam olarak şunu yaz: {abstain}

Soru: {question}

Yanıt:"""
PROMPT_JUDGE = """Aşağıda belge parçaları ve bir yanıt var. Yanıttaki her iddianın belge parçalarıyla
desteklenip desteklenmediğini değerlendir.
Puan: 2 = tüm iddialar destekleniyor; 1 = bazı iddialar destekleniyor, bazıları desteklenmiyor;
0 = iddialar desteklenmiyor veya parçalarla çelişiyor.
Yanıt "{abstain}" ise puan 2 ver.
Yalnızca JSON döndür: {{"puan": 0, "gerekce": "kısa gerekçe"}}

Belge parçaları:
{context}

Yanıt: {answer}"""

CITE = re.compile(r"\[Kaynak:\s*([^\],]+?)\s*,\s*s\.\s*(\d+)\s*\]")


def build_context(r: Retriever, hits) -> str:
    return "\n\n".join(f'[Kaynak: {r.chunks[i]["doc"]}, s. {r.chunks[i]["page"]}]\n{r.chunks[i]["text"]}'
                       for i, _ in hits)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    ap.add_argument("--conditions", nargs="+", default=["norag", "vector", "bm25", "hybrid"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--llm", choices=["groq", "stub"], default="groq")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--judge-model", default=None, help="boşsa üretim modeli (öz-tercih yanlılığına dikkat)")
    ap.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"],
                    help="gpt-oss için akıl yürütme düzeyi; düşük düzey token tüketimini azaltır")
    ap.add_argument("--judge-conds", nargs="+", default=["hybrid"],
                    help="hakem (groundedness) yalnızca bu koşullarda çalışır (token tasarrufu)")
    ap.add_argument("--em-qtypes", nargs="+", default=["kisa", "sayisal"],
                    help="EM yalnızca bu soru türlerinde raporlanır (uzun yanıtlarda EM anlamsızdır)")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--human-sample", type=int, default=30)
    ap.add_argument("--stem5-bm25", action="store_true")
    ap.add_argument("--include-unverified", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Cache(out / "llm_cache.jsonl")
    if args.llm == "stub":
        gen = judge = StubLLM()
    else:
        gen = GroqLLM(args.model, cache, reasoning_effort=args.reasoning_effort)
        judge = GroqLLM(args.judge_model, cache, reasoning_effort=args.reasoning_effort) \
            if args.judge_model else gen

    gold = load_gold(args.gold, args.split, args.include_unverified)
    if not gold:
        raise SystemExit("HATA: uygun (verified=1) soru yok")
    r = Retriever(args.index, args.stem5_bm25)
    rows = []
    for g in gold:
        for cond in args.conditions:
            if cond == "norag":
                context = ""
                prompt = PROMPT_NORAG.format(abstain=ABSTAIN, question=g["question"])
            else:
                hits = r.search(g["question"], cond, args.top_k)
                context = build_context(r, hits)
                prompt = PROMPT_RAG.format(abstain=ABSTAIN, context=context, question=g["question"])
            ans = gen.complete(prompt)
            body = CITE.sub("", ans).strip()
            abstained = normalize_answer(ABSTAIN) in normalize_answer(ans)
            cites = [(doc_key(d), int(p)) for d, p in CITE.findall(ans)]
            correct_cites = [c for c in cites if c[0] == g["gold_doc_key"] and c[1] in g["gold_pages_set"]]
            grounded = None
            if cond != "norag" and not args.no_judge and cond in args.judge_conds:
                j = parse_json(judge.complete(PROMPT_JUDGE.format(abstain=ABSTAIN, context=context, answer=ans)))
                grounded = (int(j["puan"]) / 2) if j and str(j.get("puan")) in ("0", "1", "2") else None
            rows.append({
                "id": g["id"], "condition": cond, "qtype": g.get("qtype", ""), "answerable": g["answerable"],
                "question": g["question"], "gold_answer": g.get("answer", ""), "model_answer": ans,
                "abstained": abstained,
                "em": exact_match(body, g.get("answer", "")) if g["answerable"] else None,
                "f1": f1_score(body, g.get("answer", "")) if g["answerable"] else None,
                "f1_stem5": f1_score(body, g.get("answer", ""), stem5=True) if g["answerable"] else None,
                "cited": bool(cites),
                "cite_any_correct": (bool(correct_cites) if g["answerable"] else None),
                "cite_precision": (len(correct_cites) / len(cites) if cites and g["answerable"] else None),
                "grounded": grounded,
            })

    with open(out / "runs.jsonl", "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    summary = {"prompt_version": PROMPT_VERSION, "model": args.model, "judge_model": args.judge_model or args.model,
               "top_k": args.top_k, "split": args.split, "n_questions": len(gold),
               "index_meta": r.meta, "conditions": {}}
    print(f"{'koşul':8s} {'EM*':>6s} {'F1':>6s} {'F1s5':>6s} {'atıf':>6s} {'ground':>7s} {'çekim.(cevapsız)':>17s}")
    for cond in args.conditions:
        cr = [x for x in rows if x["condition"] == cond]
        ans = [x for x in cr if x["answerable"]]
        una = [x for x in cr if not x["answerable"]]
        em = [x["em"] for x in ans if x["qtype"] in args.em_qtypes]
        s = {
            "n_answerable": len(ans), "n_unanswerable": len(una),
            "em_short": mean(em), "em_short_n": len(em),
            "f1": mean([x["f1"] for x in ans]), "f1_ci95": bootstrap_ci([x["f1"] for x in ans]),
            "f1_stem5": mean([x["f1_stem5"] for x in ans]),
            "cite_any_correct": mean([x["cite_any_correct"] for x in ans if cond != "norag"]),
            "cite_precision": mean([x["cite_precision"] for x in ans]),
            "grounded": mean([x["grounded"] for x in ans if not x["abstained"]]),
            "grounded_all_answerable": mean([x["grounded"] for x in ans]),
            "abstain_rate_unanswerable": mean([float(x["abstained"]) for x in una]) if una else None,
            "wrong_abstain_rate_answerable": mean([float(x["abstained"]) for x in ans]),
        }
        summary["conditions"][cond] = s
        ab = f'{s["abstain_rate_unanswerable"]:.3f}' if una else "-"
        print(f'{cond:8s} {s["em_short"]:6.3f} {s["f1"]:6.3f} {s["f1_stem5"]:6.3f} '
              f'{s["cite_any_correct"]:6.3f} {s["grounded"]:7.3f} {ab:>17s}')
    print("* EM yalnızca kısa/sayısal sorularda; ground yalnızca cevaplanabilir ve çekimser olmayan yanıtlarda")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # İnsan kontrol örneği: hakem uyumunu ve doğruluğu elle kontrol etmek için (Zheng vd. önerisi)
    main_cond = "hybrid" if "hybrid" in args.conditions else args.conditions[-1]
    pool = [x for x in rows if x["condition"] == main_cond]
    random.Random(42).shuffle(pool)
    with open(out / "human_check.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "condition", "question", "gold_answer", "model_answer", "judge_grounded_0_1",
                    "human_correct_0_1_2", "human_grounded_0_1_2"])
        for x in pool[:args.human_sample]:
            w.writerow([x["id"], x["condition"], x["question"], x["gold_answer"], x["model_answer"],
                        "" if x["grounded"] is None else x["grounded"], "", ""])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
