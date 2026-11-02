"""PDF'lerden bölüt indeksi üretir (sayfa bilgisi korunur) ve korpus tablosunu yazar.

Örnek:
  python build_index.py --pdf-dir data --out index_minilm \
      --embedder sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --chunk-size 1000 --overlap 150
  python build_index.py --pdf-dir data --out index_bge --embedder BAAI/bge-m3 --chunk-size 1000 --overlap 150

Çıktılar (--out klasörü): chunks.jsonl, emb.npy, meta.json, corpus_manifest.csv (makaledeki korpus tablosu için).
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import platform
import sys
from pathlib import Path

import numpy as np

from common import STEmbedder, TfidfEmbedder, read_pdf_pages, split_text


def lib_versions() -> dict:
    v = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("fitz", "pypdf", "sentence_transformers", "langchain_text_splitters", "sklearn"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "var")
        except ImportError:
            pass
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embedder", required=True,
                    help="HF model adı (ör. BAAI/bge-m3) veya 'tfidf' (yalnızca duman testi)")
    ap.add_argument("--chunk-size", type=int, default=1000, help="karakter")
    ap.add_argument("--overlap", type=int, default=150, help="karakter")
    ap.add_argument("--min-chars", type=int, default=30, help="bundan kısa bölütler atılır")
    args = ap.parse_args()

    pdfs = sorted(Path(args.pdf_dir).rglob("*.pdf"))
    if not pdfs:
        print(f"HATA: {args.pdf_dir} içinde PDF bulunamadı", file=sys.stderr)
        return 1

    chunks, manifest, splitter = [], [], ""
    for pdf in pdfs:
        try:
            pages = read_pdf_pages(pdf)
        except Exception as e:  # bozuk PDF belgeyi atlar ama kayda geçer
            print(f"UYARI: {pdf.name} okunamadı: {e}", file=sys.stderr)
            manifest.append({"doc": pdf.name, "pages": 0, "empty_pages": 0, "chars": 0, "chunks": 0,
                             "status": f"hata: {e}"})
            continue
        n_chunks = 0
        for pno, text in enumerate(pages, start=1):  # sayfa numarası 1-tabanlı
            if len(text) < args.min_chars:
                continue
            parts, splitter = split_text(text, args.chunk_size, args.overlap)
            for part in parts:
                if len(part) < args.min_chars:
                    continue
                chunks.append({"id": len(chunks), "doc": pdf.name, "page": pno, "text": part})
                n_chunks += 1
        manifest.append({
            "doc": pdf.name, "pages": len(pages),
            "empty_pages": sum(1 for t in pages if len(t) < 200),
            "chars": sum(len(t) for t in pages), "chunks": n_chunks, "status": "ok",
        })
        print(f"{pdf.name}: {len(pages)} sayfa, {n_chunks} bölüt")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    texts = [c["text"] for c in chunks]

    if args.embedder == "tfidf":
        emb = TfidfEmbedder()
        emb.fit(texts)
        with open(out / "tfidf.pkl", "wb") as f:
            pickle.dump(emb.vec, f)
    else:
        emb = STEmbedder(args.embedder)
    vectors = emb.encode(texts)
    np.save(out / "emb.npy", vectors)

    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(out / "corpus_manifest.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    meta = {
        "embedder": args.embedder, "chunk_size": args.chunk_size, "overlap": args.overlap,
        "splitter": splitter, "n_docs": len(manifest), "n_pages": sum(m["pages"] for m in manifest),
        "n_chunks": len(chunks), "dim": int(vectors.shape[1]), "versions": lib_versions(),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
