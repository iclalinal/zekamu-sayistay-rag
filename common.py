"""Ortak yardımcılar: Türkçe metin işleme, PDF okuma, bölütleme, BM25, gömme ve erişim.

Bağımlılıklar isteğe bağlıdır:
  PDF okuma  : PyMuPDF (fitz) varsa o, yoksa pypdf
  Bölütleme  : langchain_text_splitters varsa o, yoksa dahili basit bölütleyici
  Gömme      : sentence-transformers (ör. bge-m3, MiniLM); "tfidf" yalnızca duman testi içindir
Kullanılan bölütleyici ve kütüphane sürümleri indeksin meta.json dosyasına yazılır.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Türkçe metin
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[0-9a-zçğıöşüâîû]+")


def tr_lower(s: str) -> str:
    """Türkçe uyumlu küçük harfe çevirme (İ->i, I->ı)."""
    return s.replace("İ", "i").replace("I", "ı").lower()


def tokenize(text: str, stem5: bool = False) -> list[str]:
    toks = _TOKEN.findall(tr_lower(text))
    return [t[:5] for t in toks] if stem5 else toks


def normalize_answer(s: str) -> str:
    """EM/F1 için normalizasyon: küçük harf, noktalama kaldırma, sayı biçimi (1.234,5 -> 1234.5)."""
    s = tr_lower(s)
    s = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", s)  # binlik ayracı
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)  # ondalık virgül
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)  # rakam arası olmayan noktalar
    s = re.sub(r"[^0-9a-zçğıöşüâîû.\s]", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str, stem5: bool = False) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if stem5:
        p, g = [t[:5] for t in p], [t[:5] for t in g]
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


# --------------------------------------------------------------------------- #
# PDF ve bölütleme
# --------------------------------------------------------------------------- #
def read_pdf_pages(path: str | Path) -> list[str]:
    """Sayfa listesi döndürür (indeks 0 = sayfa 1)."""
    try:
        import fitz  # PyMuPDF

        with fitz.open(str(path)) as d:
            pages = [p.get_text("text") for p in d]
    except ImportError:
        from pypdf import PdfReader

        pages = [(p.extract_text() or "") for p in PdfReader(str(path)).pages]
    return [re.sub(r"[ \t]+", " ", t).strip() for t in pages]


def _simple_split(text: str, size: int, overlap: int) -> list[str]:
    pieces = [p for p in re.split(r"(?<=[.!?:;])\s+|\n{2,}", text) if p.strip()]
    chunks, cur = [], ""
    for p in pieces:
        while len(p) > size:  # çok uzun parçayı sert böl
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:size])
            p = p[size - overlap:]
        if len(cur) + len(p) + 1 <= size:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                chunks.append(cur)
            tail = cur[-overlap:] if overlap and cur else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            cur = (tail + " " + p).strip()
    if cur:
        chunks.append(cur)
    return chunks


def split_text(text: str, size: int, overlap: int) -> tuple[list[str], str]:
    """(bölütler, kullanılan bölütleyici adı)."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        sp = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""]
        )
        return sp.split_text(text), "langchain_RecursiveCharacterTextSplitter"
    except ImportError:
        return _simple_split(text, size, overlap), "simple_sentence_splitter"


# --------------------------------------------------------------------------- #
# Gömme
# --------------------------------------------------------------------------- #
class STEmbedder:
    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.model = SentenceTransformer(name, device="cpu")

    def encode(self, texts: list[str], batch: int = 32) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, batch_size=batch, normalize_embeddings=True,
                              show_progress_bar=len(texts) > 200),
            dtype=np.float32,
        )


class TfidfEmbedder:
    """YALNIZCA duman testi içindir; makale sonuçlarında kullanılmaz."""

    name = "tfidf"

    def __init__(self, vec=None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = vec or TfidfVectorizer(max_features=4096, sublinear_tf=True,
                                          tokenizer=tokenize, lowercase=False, token_pattern=None)

    def fit(self, texts: list[str]):
        self.vec.fit(texts)

    def encode(self, texts: list[str], batch: int = 0) -> np.ndarray:
        return self.vec.transform(texts).toarray().astype(np.float32)


# --------------------------------------------------------------------------- #
# BM25 (Okapi)
# --------------------------------------------------------------------------- #
class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.dl = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if self.N else 0.0
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs_tokens):
            for t, c in Counter(d).items():
                self.post[t].append((i, c))
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def scores(self, q_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.N, dtype=np.float32)
        for t in set(q_tokens):
            if t not in self.post:
                continue
            idf = self.idf[t]
            for i, tf in self.post[t]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                s[i] += idf * tf * (self.k1 + 1) / denom
        return s


# --------------------------------------------------------------------------- #
# İndeks ve erişim
# --------------------------------------------------------------------------- #
def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def doc_key(name: str) -> str:
    """Belge adını karşılaştırma için normalize eder (.pdf ve büyük/küçük harf yok sayılır)."""
    n = Path(str(name).strip()).name
    return tr_lower(n[:-4] if n.lower().endswith(".pdf") else n)


class Retriever:
    METHODS = ("vector", "bm25", "hybrid")

    def __init__(self, index_dir: str | Path, stem5_bm25: bool = False):
        d = Path(index_dir)
        self.dir = d
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.chunks = load_jsonl(d / "chunks.jsonl")
        self.emb = np.load(d / "emb.npy")
        self._embedder = None
        self.stem5 = stem5_bm25
        self.bm25 = BM25([tokenize(c["text"], stem5_bm25) for c in self.chunks])

    @property
    def embedder(self):
        if self._embedder is None:
            name = self.meta["embedder"]
            if name == "tfidf":
                with open(self.dir / "tfidf.pkl", "rb") as f:
                    self._embedder = TfidfEmbedder(pickle.load(f))
            else:
                self._embedder = STEmbedder(name)
        return self._embedder

    @staticmethod
    def _top(scores: np.ndarray, k: int) -> list[int]:
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        return [int(i) for i in idx[np.argsort(-scores[idx], kind="stable")]]

    def search(self, query: str, method: str, k: int = 10, pool: int = 100, rrf_k: int = 60):
        """[(chunk_indeksi, skor)] döndürür. hybrid = vektör + BM25 sıralarının RRF birleşimi."""
        if method == "vector":
            s = self.emb @ self.embedder.encode([query])[0]
            return [(i, float(s[i])) for i in self._top(s, k)]
        if method == "bm25":
            s = self.bm25.scores(tokenize(query, self.stem5))
            return [(i, float(s[i])) for i in self._top(s, k) if s[i] > 0]
        if method == "hybrid":
            sv = self.emb @ self.embedder.encode([query])[0]
            sb = self.bm25.scores(tokenize(query, self.stem5))
            fused: dict[int, float] = defaultdict(float)
            for rank, i in enumerate(self._top(sv, pool), 1):
                fused[i] += 1.0 / (rrf_k + rank)
            for rank, i in enumerate([j for j in self._top(sb, pool) if sb[j] > 0], 1):
                fused[i] += 1.0 / (rrf_k + rank)
            return sorted(fused.items(), key=lambda x: -x[1])[:k]
        raise ValueError(f"bilinmeyen yöntem: {method}")


# --------------------------------------------------------------------------- #
# Kontrol kümesi (gold set)
# --------------------------------------------------------------------------- #
GOLD_COLUMNS = ["id", "question", "answer", "gold_doc", "gold_pages", "answerable",
                "qtype", "split", "verified"]


def parse_pages(s: str) -> set[int]:
    pages: set[int] = set()
    for part in re.split(r"[;,\s]+", str(s).strip()):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            pages.update(range(int(a), int(b) + 1))
        else:
            pages.add(int(part))
    return pages


def load_gold(path: str | Path, split: str | None = None, include_unverified: bool = False) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not include_unverified and str(r.get("verified", "")).strip() != "1":
                continue
            if split and r.get("split", "").strip() != split:
                continue
            r["answerable"] = str(r.get("answerable", "1")).strip() != "0"
            r["gold_pages_set"] = parse_pages(r.get("gold_pages", "")) if r["answerable"] else set()
            r["gold_doc_key"] = doc_key(r.get("gold_doc", "")) if r["answerable"] else ""
            rows.append(r)
    return rows


def is_hit(chunk: dict, gold: dict) -> bool:
    return doc_key(chunk["doc"]) == gold["gold_doc_key"] and int(chunk["page"]) in gold["gold_pages_set"]


# --------------------------------------------------------------------------- #
# İstatistik yardımcıları
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def mcnemar_exact(a_hits: list[bool], b_hits: list[bool]) -> dict:
    """Eşleştirilmiş isabet karşılaştırması (iki yönlü tam binom)."""
    from scipy.stats import binomtest

    b = sum(1 for x, y in zip(a_hits, b_hits) if x and not y)
    c = sum(1 for x, y in zip(a_hits, b_hits) if y and not x)
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    return {"a_only": b, "b_only": c, "p_value": float(p)}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
