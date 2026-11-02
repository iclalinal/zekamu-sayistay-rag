"""human_check.csv doldurulduktan sonra: hakem (LLM) ile insan arasındaki ağırlıklı kappa ve insan doğruluğu."""
import argparse
import csv

import numpy as np


def wkappa(a, b, k=3):
    a, b = np.asarray(a, int), np.asarray(b, int)
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    den = (W * E).sum()
    return 1 - (W * O).sum() / den if den else float("nan")


ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True)
a = ap.parse_args()
rows = [r for r in csv.DictReader(open(a.file, encoding="utf-8-sig"))
        if r["human_grounded_0_1_2"].strip() != "" and r["judge_grounded_0_1"].strip() != ""]
if not rows:
    raise SystemExit("HATA: human_grounded_0_1_2 ve judge sütunları dolu satır yok")
j = [round(float(r["judge_grounded_0_1"]) * 2) for r in rows]
h = [int(r["human_grounded_0_1_2"]) for r in rows]
print(f"n={len(rows)}  ağırlıklı kappa (kuadratik)={wkappa(j, h):.3f}  tam uyum={np.mean(np.array(j) == np.array(h)):.3f}")
hc = [int(r["human_correct_0_1_2"]) for r in rows if r["human_correct_0_1_2"].strip() != ""]
if hc:
    print(f"insan doğruluk ortalaması (0-2 -> 0-1): {np.mean(hc) / 2:.3f} (n={len(hc)})")
