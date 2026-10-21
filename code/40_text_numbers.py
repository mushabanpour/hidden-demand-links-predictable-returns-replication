"""
40_text_numbers.py -- numbers quoted in the text that are not table cells: data coverage,
local-market counts, overlap-network statistics, RPS persistence, the sign of the long-short return.
(Other in-text numbers come from logs/decomp.json, logs/intramonth.json, logs/permutation.txt,
logs/mech_ownership_diff.json, logs/main_results_co.json, and logs/robust_raw.json.)
Output: logs/text_numbers.json
"""
import os
import sys, re, glob, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")
R = ap.R
out = {}

# A. coverage of the raw data (extraction log; 2025-11 re-extracted after de-duplication)
rows = {}
for f in [f"{R}/logs/extract.log", f"{R}/logs/re2511.log"]:
    for line in open(f):
        m = re.match(r"(\d{4}-\d{2}): POIs=([\d,]+) branded=([\d,]+)", line)
        if m:
            rows[m.group(1)] = (int(m.group(2).replace(",", "")), int(m.group(3).replace(",", "")))
D = pd.DataFrame(rows, index=["poi", "branded"]).T.sort_index()
out["data"] = {"months": len(D), "first": D.index[0], "last": D.index[-1],
               **{f"{c}_{s}": (int(getattr(D[c], s)()) if s != "argmin" else D[c].idxmin())
                  for c in ["poi", "branded"] for s in ["min", "median", "max", "argmin"]}}
Mk = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{R}/data/extract/market_month/*.parquet")))
g = Mk.assign(z=Mk.mkt.str.startswith("z")).groupby("month").agg(n=("mkt", "size"), z=("z", "sum"))
out["markets"] = {"mean": g.n.mean(), "min": int(g.n.min()), "max": int(g.n.max()), "zip_mean": g.z.mean(),
                  "county_mean": (g.n - g.z).mean()}

# B. customer-overlap network over the 77 formation months (pairs i < j of sample firms)
P = M.load("co")
P = P[P.month <= P.dropna(subset=["ret_f1"]).month.max()]
st, vals = [], []
for t, d in P.groupby("month"):
    U = d.set_index("permno"); n = len(U)
    O = pd.read_parquet(f"{R}/data/derived/ovl/{t}.parquet", columns=["i", "j", "A_S"])
    O = O[O.i.isin(U.index) & O.j.isin(U.index) & (O.i < O.j)]
    L = O[O.A_S > 0]
    ci, cj = L.i.map(U.cat).values, L.j.map(U.cat).values
    cnt = U.cat.value_counts(); n_same = int((cnt * (cnt - 1) / 2).sum())
    st.append(dict(n_pairs=n * (n - 1) / 2, n_linked=len(L), n_same=n_same, n_same_linked=int((ci == cj).sum()),
                   cross_share=float((ci != cj).mean()), cross_weight=float(L.A_S[ci != cj].sum() / L.A_S.sum())))
    vals.append(L.A_S.values)
S = pd.DataFrame(st); v = np.concatenate(vals)
out["overlap"] = {"pairs_per_month": S.n_pairs.mean(), "linked_per_month": S.n_linked.mean(),
                  "share_linked_all": (S.n_linked / S.n_pairs).mean(), "share_linked_samecat": (S.n_same_linked / S.n_same).mean(),
                  "samecat_pairs_per_month": S.n_same.mean(),
                  "linked_cross_category_share": S.cross_share.mean(), "linked_cross_category_weight": S.cross_weight.mean(),
                  "median_linked_pct": float(np.median(v) * 100), "p75_linked_pct": float(np.quantile(v, 0.75) * 100),
                  "median_peers": float(P.loc[P.co_n.notna(), "co_n"].median()), "firms_per_month": P.groupby("month").size().mean(),
                  "share_firms_with_peers": P.sig.notna().mean()}

# C. persistence of the networks (12-month autocorrelation of pairwise similarity)
rps, store = import_module("18_descriptives").persistence()
out["persistence"] = {"RPS": rps, "STORE": store}

# D. the sign of the long-short return
ew, vw, n = ap.sort_returns(P, "sig", 5)
out["ls_sign"] = {"ew_pos_share": (ew.LS > 0).mean(), "vw_pos_share": (vw.LS > 0).mean()}
out["quintiles"] = {w: [ap.nw_mean(x[q])[0] * 100 for q in range(1, 6)] for w, x in [("ew", ew), ("vw", vw)]}

json.dump(out, open(f"{R}/logs/text_numbers.json", "w"), default=float, indent=1)
print(json.dumps(out, default=float, indent=1))
