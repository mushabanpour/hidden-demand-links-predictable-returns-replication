"""26_co_variants.py -- customer-overlap network variants: windows (6/24 months), frozen 2019
network, one-directional overlap (share of i's customers at j, not symmetrised)."""
import os, glob
import numpy as np, pandas as pd, duckdb
R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); F = f"{R}/data/derived/firm"; OUT = f"{R}/data/derived/ovl_var"


def overlap(win, con, mp):
    fl = lambda tab: "[" + ",".join(f"'{F}/{tab}/{m}.parquet'" for m in win) + "]"
    wv = con.execute(f"SELECT permno, sum(w_sum) w FROM read_parquet({fl('firm_month')}) GROUP BY permno").df()
    wv = wv[wv.w >= 2000].set_index("permno").w
    mid = pd.Timestamp(win[-1] + "-15")
    m = mp[(mp.start <= mid) & (mp.end >= mid) & mp.permno.isin(wv.index)]
    b2p = dict(zip(m.brand, m.permno))
    d = con.execute(f"SELECT permno, b, sum(act_num) a FROM read_parquet({fl('firm_rel_SAME')}) GROUP BY permno, b").df()
    d = d[d.permno.isin(wv.index)]; d["j"] = d.b.map(b2p); d = d.dropna(subset=["j"]); d["j"] = d.j.astype(int)
    d = d[d.j != d.permno]; d["A"] = (d.a / d.permno.map(wv) / 100).clip(0, 1)
    g = d.groupby(["permno", "j"]).A.agg(lambda s: 1 - np.prod(1 - s)).reset_index().rename(columns={"permno": "i"})
    gt = g.rename(columns={"i": "j", "j": "i", "A": "At"})
    s = g.merge(gt, on=["i", "j"], how="outer").fillna(0.0)
    s["sym"] = (s.A + s.At) / 2
    return s[["i", "j", "A", "sym"]]


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    allm = sorted(os.path.basename(f)[:7] for f in glob.glob(f"{F}/firm_day/*.parquet"))
    mp = pd.read_parquet(f"{R}/data/derived/brand_firm_map.parquet")
    con = duckdb.connect(); con.execute("SET threads=4; SET memory_limit='2GB'")
    frozen = overlap([m for m in allm if m.startswith("2019")], con, mp).rename(columns={"sym": "CO_FROZEN"})[["i", "j", "CO_FROZEN"]]
    for k, t in enumerate(allm):
        if k < 6 or os.path.exists(f"{OUT}/{t}.parquet"):
            continue
        a = overlap(allm[max(0, k - 12):k], con, mp).rename(columns={"A": "CO_ASYM"})[["i", "j", "CO_ASYM"]]
        b = overlap(allm[max(0, k - 6):k], con, mp).rename(columns={"sym": "CO_W6"})[["i", "j", "CO_W6"]]
        c = overlap(allm[max(0, k - 24):k], con, mp).rename(columns={"sym": "CO_W24"})[["i", "j", "CO_W24"]]
        res = a.merge(b, on=["i", "j"], how="outer").merge(c, on=["i", "j"], how="outer")
        if t >= "2020-01":
            res = res.merge(frozen, on=["i", "j"], how="outer")
        res["t"] = t
        res.to_parquet(f"{OUT}/{t}.parquet", index=False)
        print(t, len(res), flush=True)
