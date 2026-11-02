"""Ağırlık duyarlılığı: her ağırlık x U(1-d, 1+d) ile bozulur, yeniden normalize edilir (Monte Carlo);
temel sıralama ile Kendall tau ve sıra kararlılığı raporlanır."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from indicators import WEIGHTS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indicators", required=True)
    ap.add_argument("--delta", type=float, default=0.20)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.indicators, encoding="utf-8-sig")))
    X = np.array([[float(r[k]) for k in "ACRH"] for r in rows])
    w0 = np.array(list(WEIGHTS.values()))
    base = X @ w0
    rng = np.random.default_rng(a.seed)
    taus, ranks = [], []
    for _ in range(a.n):
        w = w0 * rng.uniform(1 - a.delta, 1 + a.delta, 4)
        w /= w.sum()
        s = X @ w
        t = kendalltau(base, s).statistic
        taus.append(0.0 if np.isnan(t) else t)
        ranks.append((-s).argsort().argsort() + 1)
    ranks = np.array(ranks)
    br = (-base).argsort().argsort() + 1
    top = max(1, len(rows) // 10)
    summ = {"n_reports": len(rows), "delta": a.delta, "n_sim": a.n,
            "tau_mean": float(np.mean(taus)), "tau_p5": float(np.quantile(taus, 0.05)),
            "tau_min": float(np.min(taus)),
            "mean_abs_rank_shift": float(np.mean(np.abs(ranks - br))),
            f"top{top}_stay_rate": float(np.mean([(br <= top)[i] and (ranks[:, i] <= top).mean() or 0
                                                  for i in range(len(rows)) if br[i] <= top]))}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "rank_stability.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc", "T", "base_rank", "rank_mean", "rank_p5", "rank_p95"])
        for i, r in enumerate(rows):
            w.writerow([r["doc"], r["T"], br[i], round(ranks[:, i].mean(), 2),
                        int(np.quantile(ranks[:, i], .05)), int(np.quantile(ranks[:, i], .95))])
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
