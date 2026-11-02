"""Duman testi için sentetik Türkçe 'rapor' PDF'leri üretir (gerçek Sayıştay verisi DEĞİLDİR)."""
import random
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo_pdfs")
out.mkdir(exist_ok=True)
pdfmetrics.registerFont(TTFont("DV", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
rnd = random.Random(1)
S = ["Kurumun {y} yılı performans hedefi {n} milyon TL olarak belirlenmiştir.",
     "Hedefin gerçekleşme sonucu yüzde {p} düzeyinde kalmıştır.",
     "Denetim sonucunda {k} adet bulgu tespit edilmiş ve sorumlu birimlere bildirilmiştir.",
     "Yönetim, tavsiye edilen düzeltici işlemleri {y} yılı sonuna kadar tamamlayacaktır.",
     "Kamu zararı tutarı {n} bin TL olarak hesaplanmıştır.",
     "Stratejik plan kapsamındaki amaçlar yıllık faaliyet raporunda izlenmektedir."]
for i, (kurum, y) in enumerate([("Alfa Bakanlığı", 2023), ("Beta Üniversitesi", 2024), ("Gama Belediyesi", 2022),
                                 ("Delta Kurumu", 2025), ("Epsilon İdaresi", 2021)]):
    c = canvas.Canvas(str(out / f"{kurum.split()[0]}_{y}_Denetim_Raporu.pdf"), pagesize=A4)
    for pg in range(1, 7):
        c.setFont("DV", 10)
        yy = 800
        for _ in range(14):
            line = rnd.choice(S).format(y=y, n=rnd.randint(10, 900), p=rnd.randint(20, 99), k=rnd.randint(2, 20))
            c.drawString(40, yy, f"{kurum}: {line}")
            yy -= 22
        c.showPage()
    c.save()
print("ok")
