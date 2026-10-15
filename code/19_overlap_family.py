"""
19_overlap_family.py -- direct customer-overlap networks and their decomposition.

For firm i and peer j (j's brands B_j), over the window t-12..t-1:
  A^L_ij = share of i's customers who also shop at j's brands        (list L)
  E^L_ij = share predicted if i's customers behaved like other shoppers in i's local
           markets (ZIP/county for PHYS; county x category for SAME)
  X^L_ij = A^L_ij - E^L_ij  (excess, preference-driven overlap)
Lists: SAME (same-category brands, top-20 within category) and PHYS (top-20 brands).
Brand-level shares are combined across j's brands with the probabilistic union
1 - prod(1 - s_b). Matrices are symmetrised: (M_ij + M_ji)/2.
Output: data/derived/ovl/<t>.parquet  (t, i, j, A_S, E_S, X_S, A_P, E_P, X_P)
"""
import os, glob, sys
import numpy as np, pandas as pd, duckdb

R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F = f"{R}/data/derived/firm"; OUT = f"{R}/data/derived/ovl"


def main():
    os.makedirs(OUT, exist_ok=True)
    allm = sorted(os.path.basename(f)[:7] for f in glob.glob(f"{F}/firm_day/*.parquet"))
    mp = pd.read_parquet(f"{R}/data/derived/brand_firm_map.parquet")
    con = duckdb.connect(); con.execute("SET threads=4; SET memory_limit='3GB'")
    for k, t in enumerate(allm):
        win = allm[max(0, k - 12):k]
        if len(win) < 6 or os.path.exists(f"{OUT}/{t}.parquet"):
            continue
        fl = lambda tab: "[" + ",".join(f"'{F}/{tab}/{m}.parquet'" for m in win) + "]"
        wv = con.execute(f"SELECT permno, sum(w_sum) w FROM read_parquet({fl('firm_month')}) GROUP BY permno").df()
        wv = wv[wv.w >= 2000].set_index("permno").w
        mid = pd.Timestamp(win[-1] + "-15")
        m = mp[(mp.start <= mid) & (mp.end >= mid) & mp.permno.isin(wv.index)]
        b2p = dict(zip(m.brand, m.permno))
        res = None
        for L, tag in [("SAME", "S"), ("PHYS", "P")]:
            d = con.execute(f"SELECT permno, b, sum(act_num) a, sum(exp_num) e FROM read_parquet({fl('firm_rel_' + L)}) GROUP BY permno, b").df()
            d = d[d.permno.isin(wv.index)]
            d["j"] = d.b.map(b2p); d = d.dropna(subset=["j"]); d["j"] = d.j.astype(int)
            d = d[d.j != d.permno]
            d["A"] = (d.a / d.permno.map(wv) / 100).clip(0, 1); d["E"] = (d.e / d.permno.map(wv) / 100).clip(0, 1)
            g = d.groupby(["permno", "j"]).agg(A=("A", lambda s: 1 - np.prod(1 - s)), E=("E", lambda s: 1 - np.prod(1 - s))).reset_index()
            g = g.rename(columns={"permno": "i"})
            # symmetrise
            gt = g.rename(columns={"i": "j", "j": "i", "A": "At", "E": "Et"})
            s = g.merge(gt, on=["i", "j"], how="outer").fillna(0.0)
            s[f"A_{tag}"] = (s.A + s.At) / 2; s[f"E_{tag}"] = (s.E + s.Et) / 2; s[f"X_{tag}"] = s[f"A_{tag}"] - s[f"E_{tag}"]
            s = s[["i", "j", f"A_{tag}", f"E_{tag}", f"X_{tag}"]]
            res = s if res is None else res.merge(s, on=["i", "j"], how="outer")
        res["t"] = t
        res.to_parquet(f"{OUT}/{t}.parquet", index=False)
        print(t, len(res), flush=True)


if __name__ == "__main__":
    main()
