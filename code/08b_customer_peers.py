"""08b_customer_peers.py -- Cohen-Frazzini (2008) customer momentum: month-t return of the
firm's principal customers (Compustat segment disclosures), weighted by the share of the
supplier's sales, using the latest disclosure that is at least six months old."""
import os
import numpy as np, pandas as pd
R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
lk = pd.read_parquet(f"{R}/data/derived/cust_supplier_links.parquet")
lk["avail"] = (pd.to_datetime(lk.srcdate) + pd.offsets.MonthEnd(6)).dt.to_period("M")
lk = lk.groupby(["gvkey", "cust_gvkey", "avail"], as_index=False).salecs.max()
sm = pd.read_parquet(f"{R}/data/derived/stock_month.parquet", columns=["permno", "month", "ret", "gvkey"])
sm = sm[sm.month >= pd.Period("2018-12", "M")]
g2p = sm.dropna(subset=["gvkey"]).sort_values("month").drop_duplicates("gvkey", keep="last").set_index("gvkey").permno
out = []
for t, d in sm.groupby("month"):
    l = lk[(lk.avail <= t) & (lk.avail > t - 24)].sort_values("avail").drop_duplicates(["gvkey", "cust_gvkey"], keep="last")
    r = d.dropna(subset=["gvkey"]).groupby("gvkey").ret.first()
    l = l[l.cust_gvkey.isin(r.index)].copy()
    l["rc"] = l.cust_gvkey.map(r); l["w"] = l.salecs.fillna(0).clip(lower=0) + 1e-6
    a = l.groupby("gvkey").apply(lambda x: (x.w * x.rc).sum() / x.w.sum(), include_groups=False).rename("cust_ret")
    a = a.reset_index(); a["permno"] = a.gvkey.map(g2p); a["month"] = t
    out.append(a.dropna(subset=["permno"])[["permno", "month", "cust_ret"]])
C = pd.concat(out); C["permno"] = C.permno.astype(int)
cp = pd.read_parquet(f"{R}/data/derived/control_peers.parquet")
cp = cp.drop(columns=[c for c in ["cust_ret"] if c in cp]).merge(C, on=["permno", "month"], how="left")
cp.to_parquet(f"{R}/data/derived/control_peers.parquet", index=False)
print(C.shape, cp.cust_ret.notna().sum())
