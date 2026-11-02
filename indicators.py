"""Rapor düzeyinde A/C/R/H göstergeleri ve bileşik skor T = 0.25A + 0.35C + 0.20R + 0.20H.

ÖNEMLİ: Aşağıdaki kurallar ÖNERİ niteliğindedir. Deneyden ÖNCE onaylayıp makalede sabitle,
sonuçlara bakarak değiştirme (ön onay formundaki "önceden sabitlenmiş kural" taahhüdü).
  A (Açıklık/erişilebilirlik): metin katmanı olan sayfa oranı (>=200 karakter) x 100
  C (Tutarlılık): hedef geçen sayfalardan, aynı sayfada gerçekleşme/sonuç terimi de geçenlerin oranı x 100
  R (Güncellik): 100 - 25 x (referans yıl - rapor yılı), [0,100] aralığına kırpılır
  H (Hesap verebilirlik): bulgu/sorumluluk terimi VE sayı içeren sayfa oranı x 100
Rapor yılı: dosya adındaki 20xx, yoksa ilk 3 sayfadaki en sık 20xx.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from common import read_pdf_pages, tr_lower

WEIGHTS = {"A": 0.25, "C": 0.35, "R": 0.20, "H": 0.20}
TARGET = ["hedef", "performans programı", "stratejik plan", "amaç"]
RESULT = ["gerçekleş", "sonuç", "ulaşıl", "tamamlan"]
ACCOUNT = ["bulgu", "sorumlu", "hesap verebilir", "tavsiye", "öneri", "kamu zararı", "görüş", "yönetim"]
NUM = re.compile(r"\d[\d.,]*")


def report_year(name: str, pages: list[str], manifest: dict | None = None) -> int | None:
    """Denetim yılı (mali yıl). Öncelik: manifest (liste sayfasındaki yıl), sonra kapaktaki "20xx YILI",
    sonra dosya adı. Kapaktaki yayın tarihi (ör. "Eylül 2026") denetim yılı DEĞİLDİR."""
    if manifest and name in manifest:
        return int(manifest[name])
    m = re.findall(r"\b(20\d\d)\s*YILI", " ".join(pages[:8]).upper().replace("İ", "I").replace("ı", "I"))
    if m:
        return int(Counter(m).most_common(1)[0][0])
    m = re.findall(r"20\d\d", name)
    if m:
        return int(m[0])
    c = Counter(re.findall(r"\b(20\d\d)\b", " ".join(pages[:3])))
    return int(c.most_common(1)[0][0]) if c else None


def score(pdf: Path, ref_year: int, manifest: dict | None = None) -> dict:
    pages = read_pdf_pages(pdf)
    n = max(len(pages), 1)
    low = [tr_lower(p) for p in pages]
    A = 100 * sum(len(p) >= 200 for p in pages) / n
    tgt = [p for p in low if any(t in p for t in TARGET)]
    C = 100 * sum(any(r in p for r in RESULT) for p in tgt) / len(tgt) if tgt else 0.0
    H = 100 * sum(any(a in p for a in ACCOUNT) and bool(NUM.search(p)) for p in low) / n
    y = report_year(pdf.name, pages, manifest)
    R = max(0.0, min(100.0, 100 - 25 * (ref_year - y))) if y else 0.0
    T = sum(w * v for w, v in zip(WEIGHTS.values(), (A, C, R, H)))
    return {"doc": pdf.name, "year": y or "", "pages": len(pages), "A": round(A, 2), "C": round(C, 2),
            "R": round(R, 2), "H": round(H, 2), "T": round(T, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--ref-year", type=int, default=2026, help="güncellik için sabit referans yıl")
    ap.add_argument("--manifest", default=None, help="sample_manifest.csv (file,year sütunları)")
    ap.add_argument("--out", required=True, help="indicators.csv")
    a = ap.parse_args()
    man = None
    if a.manifest:
        with open(a.manifest, encoding="utf-8-sig", newline="") as f:
            man = {r["file"]: r["year"] for r in csv.DictReader(f) if r.get("year")}
    rows = [score(p, a.ref_year, man) for p in sorted(Path(a.pdf_dir).rglob("*.pdf"))]
    if not rows:
        raise SystemExit("HATA: PDF yok")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in sorted(rows, key=lambda r: -r["T"])[:5]:
        print(r)
    print(f"{len(rows)} rapor -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
