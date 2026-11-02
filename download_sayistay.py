"""Sayıştay "Kamu İdareleri Denetim Raporları" için tabakalı örneklem indirici.

Kendi bilgisayarında çalıştır (Sayıştay sitesine erişimi olan ağdan). Standart kütüphane + requests yeter.

  pip install requests
  python download_sayistay.py --out data_kurum --years 2023 2024 2025 --seed 42
  python download_sayistay.py --out data_kurum --dry-run      # yalnızca listeyi ve sayıları göster

Ne yapar:
  1) Seçili kategorilerin liste sayfalarını (?page=N) gezer, rapor bağlantılarını ve yılını toplar.
  2) Yıl filtresinden geçenlerden, kategori başına kota kadar rastgele (sabit seed) seçer.
  3) /reports/download/<id>-<slug> adresinden PDF'i indirir (1.5 sn arayla, var olanı atlar).
  4) sample_manifest.csv yazar (makaledeki örneklem tablosu ve tekrar üretilebilirlik için).

Not: Liste sayfasının HTML yapısı test edilemeden yazıldı. Yıl bulunamazsa betik bunu uyarı olarak
yazar; --debug-html ile ilk sayfanın HTML'ini kaydeder, bana gönder, düzeltirim.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
from pathlib import Path

import requests

BASE = "https://sayistay.gov.tr"
UA = "Mozilla/5.0 (akademik arastirma; KTO Karatay Univ. ogrenci projesi)"

# kategori id-slug : varsayılan kota (toplam ~90)
CATS = {
    "6-genel-butce-kapsamindaki-kamu-idareleri": 15,
    "7-ozel-butceli-idareler---a": 20,            # üniversiteler ağırlıklı
    "27-ozel-butceli-idareler---b": 5,
    "28-duzenleyici-ve-denetleyici-kurumlar": 5,
    "29-sosyal-guvenlik-kurumlari": 4,
    "30-ozel-idareler": 12,                       # il özel idareleri
    "31-belediyeler---bagli-idareler": 25,        # belediyeler
    "33-kalkinma-ajanslari": 4,
}

LINK = re.compile(r'href="(/reports/(?!category|download)([A-Za-z0-9]{10})-([^"?#]+))"', re.I)
YEAR = re.compile(r"\b(20[12]\d)\b")


def get(session: requests.Session, url: str, retries: int = 4) -> requests.Response:
    for a in range(retries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503):
                time.sleep(3 * (a + 1))
                continue
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(3 * (a + 1))
    raise RuntimeError(f"alınamadı: {url}")


def parse_listing(html: str) -> list[dict]:
    """Sayfadaki her rapor bağlantısı için (id, slug, yıl) çıkarır; yıl bağlantı çevresindeki metinden."""
    items, seen = [], set()
    ms = list(LINK.finditer(html))
    for i, m in enumerate(ms):
        rid = m.group(2)
        if rid in seen:
            continue
        seen.add(rid)
        end = ms[i + 1].start() if i + 1 < len(ms) else min(len(html), m.end() + 1500)
        chunk = html[m.end():end]
        yrs = YEAR.findall(chunk)
        items.append({"id": rid, "slug": m.group(3), "year": int(yrs[0]) if yrs else None})
    return items


def crawl(session, cat: str, years: set[int], delay: float, max_pages: int, debug_dir: Path | None):
    found, empty_streak = [], 0
    for p in range(1, max_pages + 1):
        url = f"{BASE}/reports/category/{cat}" + (f"?page={p}" if p > 1 else "")
        html = get(session, url).text
        if debug_dir and p == 1:
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / f"{cat[:12]}_page1.html").write_text(html, encoding="utf-8")
        items = parse_listing(html)
        if not items:
            break
        hit = [x for x in items if x["year"] in years]
        found += [{**x, "category": cat} for x in hit]
        nones = sum(1 for x in items if x["year"] is None)
        if nones:
            print(f"  UYARI {cat} s.{p}: {nones} kayıtta yıl bulunamadı", file=sys.stderr)
        # liste yeni yıldan eskiye ise hedef yıllar bittiğinde dur
        empty_streak = empty_streak + 1 if not hit and nones == 0 else 0
        if empty_streak >= 3:
            break
        time.sleep(delay)
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--max-pages", type=int, default=150)
    ap.add_argument("--scale", type=float, default=1.0, help="kotaları çarp (ör. 0.5 → yarı örneklem)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--debug-html", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = UA
    rng = random.Random(args.seed)
    years = set(args.years)

    chosen = []
    for cat, quota in CATS.items():
        print(f"[{cat}] taranıyor…")
        items = crawl(s, cat, years, args.delay, args.max_pages, out / "_debug" if args.debug_html else None)
        by_year = {}
        for x in items:
            by_year[x["year"]] = by_year.get(x["year"], 0) + 1
        k = min(len(items), max(1, round(quota * args.scale)))
        pick = rng.sample(items, k) if items else []
        print(f"  bulunan={len(items)} yıllara göre={by_year} seçilen={k}")
        chosen += pick

    man = out / "sample_manifest.csv"
    with open(man, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "id", "slug", "year", "file", "status"])
        for x in chosen:
            fn = f'{x["id"]}-{x["slug"]}.pdf'
            status = "planlandı"
            if not args.dry_run:
                dest = out / fn
                if dest.exists() and dest.stat().st_size > 10_000:
                    status = "var"
                else:
                    try:
                        r = get(s, f'{BASE}/reports/download/{x["id"]}-{x["slug"]}')
                        if not r.content.startswith(b"%PDF"):
                            status = "pdf-degil"
                        else:
                            dest.write_bytes(r.content)
                            status = "indirildi"
                    except Exception as e:  # noqa: BLE001
                        status = f"hata: {e}"
                    time.sleep(args.delay)
                print(f"  {fn[:70]:70s} {status}")
            w.writerow([x["category"], x["id"], x["slug"], x["year"], fn, status])
    print(f"\nToplam seçilen: {len(chosen)}  →  {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
