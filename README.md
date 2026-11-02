# ZeKamu değerlendirme paketi (vize makalesi)

Kurulum: `pip install -r requirements.txt`  (API anahtarı: `set CEREBRAS_API_KEY=...`; Groq için `GROQ_API_KEY`; kodda saklama)

Sıra:
1. `python build_index.py --pdf-dir data_kurum --out index_bge --embedder BAAI/bge-m3`  (MiniLM için ikinci indeks de al; karşılaştır)
2. `python goldset_draft.py --index index_bge --n 120` -> taslak; SAYFAYLA ELLE DOĞRULA, verified=1 yap, ~15 cevapsız soru ekle (şablona bak)
3. `python eval_retrieval.py --index index_bge --gold goldset.csv --split test --out results/ret`
4. `python eval_answers.py --index index_bge --gold goldset.csv --split test --out results/ans`
5. `results/ans/human_check.csv` içindeki 30 satırı elle puanla -> `python human_agreement.py --file ...`
6. `python indicators.py --pdf-dir data_kurum --manifest data_kurum/sample_manifest.csv --out results/ind.csv` -> `python sensitivity.py --indicators results/ind.csv --out results/sens`

Kurallar: ayarları yalnız dev'de seç; test'i bir kez çalıştır. indicators.py kuralları öneridir; deneyden önce onayla.
Duman testi: `python make_demo_corpus.py demo_pdfs` + `--embedder tfidf` + `--llm stub` (sonuçlar anlamsızdır, yalnız akış testi).

## Hakem düzeltmeleri (revizyon, Kasım 2026)
Colab defteri: `notebooks/revizyon_analizleri.ipynb` (API çağrısı yapmaz; mevcut `results/*/runs.jsonl` üzerinden çalışır).
Çıktılar `results/revision_v2/` altına yazılır.
1. Çekimserlik ölçütü düzeltmesi, Tablo 2 ve Tablo 3'ün yeniden hesabı
2. Yıl göstergesi (R) çıkarma; sayfa numarası ve genel terim etkisi (H, A)
3. Model puanı gizli, aynı belge parçalarıyla kör insan değerlendirme dosyaları ve κ

## Veri ve yeniden üretim notları
- `data_kurum/`: Sayıştay'ın kamuya açık denetim raporları (88 PDF). Haklar Sayıştay'a aittir; kaynak: https://www.sayistay.gov.tr
- `data_kurum/sample_manifest.csv`: rapor yılları (denetim yılı). Özgün manifest dosyası kaybolduğundan `results/ind.csv` içindeki yıllardan yeniden oluşturuldu; `python indicators.py --manifest data_kurum/sample_manifest.csv` bu yıllarla çalışır.
- `results/ans/llm_cache.jsonl`: model yanıt önbelleği. Aynı (model, akıl yürütme düzeyi, istem) için API çağrısı yapmadan sonuçları yeniden üretir.
- `results/ans/human_check_scored_nonblind.xlsx`: ilk insan puanlaması (hakem puanı görünür halde yapıldı; körleme yok). Kör yeniden puanlama: `results/revision_v2/`.
- Anahtar: `CEREBRAS_API_KEY` (varsayılan sağlayıcı) veya `GROQ_API_KEY` ortam değişkeni; kodda anahtar saklanmaz.

## Sağlayıcı ve önbellek (v1.1 notu)
Makaledeki koşu **Cerebras** üzerinden, model `gpt-oss-120b`, sıcaklık 0, akıl yürütme çabası `low` ile yapılmıştır (`results/ans/summary.json`: `provider: cerebras`).
`llm.py` v1.1'den itibaren sağlayıcı seçilebilir (`--provider cerebras|groq`, varsayılan `cerebras`) ve önbellek anahtarı
`sha("sağlayıcı\nmodel\nçaba\nistem")` biçimindedir; kayıtlı `results/ans/llm_cache.jsonl` bu anahtarla birebir eşleşir
(RAG'siz istemlerin 75/75'i ve kör değerlendirme havuzundaki RAG istemleri doğrulanmıştır). Böylece aynı komut API anahtarı olmadan
kayıtlı yanıtları yeniden üretir. Groq ile koşmak için `--provider groq --model openai/gpt-oss-120b` verilmelidir; bu durumda
anahtar farklı olacağından önbellek kullanılmaz ve çıktılar değişebilir.

`results/ans/runs.jsonl` içindeki `contain` (içerir ölçütü; boşluk/NBSP ile ayrılmış binlik gruplarını da eşler), `abstained` (eski esnek çekimserlik)
ve `abstained_strict` alanları koşu sonrası analizde eklenmiştir; güncel `eval_answers.py` bu alanları yazmaz. Düzeltilmiş çekimserlik tanımı
`notebooks/revizyon_analizleri.ipynb` içinde tanımlıdır ve bu kayıtlı alanları okur.
