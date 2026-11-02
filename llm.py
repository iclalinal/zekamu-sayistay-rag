"""LLM istemcileri: Groq (OpenAI uyumlu uç nokta), önbellekli; test için StubLLM.

GROQ_API_KEY ortam değişkeni gerekir (.env dosyasını kendin yükle: `set GROQ_API_KEY=...`).
Aynı (model, istem) çifti önbellekten gelir, bu yüzden bir koşu yarıda kesilirse kaldığı yerden sürer.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from common import sha

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
ABSTAIN = "Belgelerde bu bilgi bulunamadı."


class Cache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, str] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    o = json.loads(line)
                    self.data[o["k"]] = o["v"]

    def get(self, k: str):
        return self.data.get(k)

    def put(self, k: str, v: str):
        self.data[k] = v
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


class GroqLLM:
    def __init__(self, model: str, cache: Cache, sleep: float = 0.5, reasoning_effort: str | None = None,
                 max_tokens: int = 1024, retries: int = 40):
        import requests

        self.requests = requests
        self.model, self.cache, self.sleep = model, cache, sleep
        self.reasoning_effort, self.max_tokens, self.retries = reasoning_effort, max_tokens, retries
        self.key = os.environ.get("GROQ_API_KEY")
        if not self.key:
            raise SystemExit("HATA: GROQ_API_KEY tanımlı değil")

    def complete(self, prompt: str) -> str:
        k = sha(f"{self.model}\n{self.reasoning_effort}\n{prompt}")
        hit = self.cache.get(k)
        if hit is not None:
            return hit
        payload = {"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                   "messages": [{"role": "user", "content": prompt}]}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        for attempt in range(self.retries):
            resp = self.requests.post(GROQ_URL, json=payload, timeout=120,
                                      headers={"Authorization": f"Bearer {self.key}"})
            if resp.status_code == 200:
                text = (resp.json()["choices"][0]["message"].get("content") or "").strip()
                self.cache.put(k, text)
                time.sleep(self.sleep)
                return text
            body = resp.text[:400]
            if resp.status_code == 429:
                low = body.lower()
                if "per day" in low or "(tpd)" in low or "(rpd)" in low:
                    raise RuntimeError("Groq GÜNLÜK limit doldu (önbellek duruyor, limit sıfırlanınca aynı komutu "
                                       "tekrar çalıştır): " + body)
                wait = 0.0
                ra = resp.headers.get("retry-after")
                if ra:
                    try:
                        wait = float(ra)
                    except ValueError:
                        pass
                m = re.search(r"try again in (?:(\d+)m)?(?:([\d.]+)s)?", low)
                if m and not wait:
                    wait = int(m.group(1) or 0) * 60 + float(m.group(2) or 0)
                wait = min(120.0, max(wait, 2.0) + 1.0)
                print(f"[Groq 429, {wait:.0f} sn bekleniyor] {body[:160]}", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code in (500, 502, 503):
                print(f"[Groq {resp.status_code}, yeniden deneniyor]", flush=True)
                time.sleep(min(60, 2 ** attempt * 2))
                continue
            raise RuntimeError(f"Groq HTTP {resp.status_code}: {body}")
        raise RuntimeError("Groq: yeniden deneme sınırı aşıldı")


class StubLLM:
    """Ağ gerektirmeyen test istemcisi: ilk belge parçasının ilk cümlesini kaynakla birlikte döndürür."""

    model = "stub"

    def complete(self, prompt: str) -> str:
        if '"puan"' in prompt:  # hakem istemi
            return '{"puan": 2, "gerekce": "stub"}'
        if '"soru"' in prompt:  # soru üretim istemi
            return '{"soru": "Test sorusu nedir?", "cevap": "test"}'
        m = re.search(r"\[Kaynak: ([^\]]+?), s\. (\d+)\]\n(.+)", prompt)
        if not m:
            return ABSTAIN
        first = re.split(r"(?<=[.!?])\s", m.group(3).strip())[0]
        return f"{first} [Kaynak: {m.group(1)}, s. {m.group(2)}]"


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
