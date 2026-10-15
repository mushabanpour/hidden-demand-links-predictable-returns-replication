import os
import numpy as np, pandas as pd, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = pd.read_parquet(f"{R}/data/derived/panel_final.parquet", columns=["permno", "month", "cat"])
D = []
for t in [m for m in sorted(P.month.unique()) if m.month == 12]:
    U = set(P[P.month == t].permno)
    S = pd.read_parquet(f"{R}/data/derived/sim/{t}.parquet", columns=["i", "j", "RPS", "STORE", "HOME"])
    O = pd.read_parquet(f"{R}/data/derived/ovl/{t}.parquet", columns=["i", "j", "A_S"])
    S = S.merge(O, on=["i", "j"], how="left"); S = S[S.i.isin(U) & S.j.isin(U) & (S.i < S.j)]
    D.append(S)
D = pd.concat(D)
import figstyle as fs
fs.apply()
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3))
x = D.A_S[D.A_S > 0] * 100
ax[0].hist(x, bins=60, range=(0, 40), color=fs.BAR_DARK, edgecolor="white", linewidth=0.4)
ax[0].axvline(x.median(), color=fs.BLACK, lw=0.9, ls="--")
ax[0].set_xlabel("Customers in common, percent of customers (linked pairs)"); ax[0].set_ylabel("Firm pairs")
for k, (c, lab) in enumerate([("RPS", "Shared tastes (RPS)"), ("STORE", "Store footprint"), ("HOME", "Customer homes")]):
    v = np.sort(D[c].dropna().values)
    fs.line(ax[1], v, np.arange(1, len(v) + 1) / len(v), k, label=lab, markevery=max(1, len(v) // 9))
ax[1].set_xlabel("Pairwise cosine similarity"); ax[1].set_ylabel("Cumulative share of pairs")
ax[1].legend(loc="upper left")
plt.tight_layout(); plt.savefig(f"{R}/figures/fig_distributions.pdf"); plt.close()
print(len(x), x.describe().round(2).to_dict(), (D.A_S > 0).mean())
