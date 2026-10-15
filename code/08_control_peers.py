"""
08_control_peers.py -- month-t peer returns from competing (non-consumer) networks,
for every CRSP common stock, using only information available at the end of t:

  tnic_ret  Hoberg-Phillips TNIC-3 product-market peers (EW); TNIC year y-1 for
            Jul-Dec of y, y-2 for Jan-Jun (10-Ks must be filed); 2023 vintage carried forward
  hq_ret    Parsons-Sabbatucci-Titman HQ-geography peers: EW return of all stocks
            headquartered in the same CBSA, excluding the focal's FF48 industry
  cov_ret   Ali-Hirshleifer shared-analyst peers: return weighted by # of analysts
            covering both firms in calendar year y-1 (IBES detail)
Output: data/derived/control_peers.parquet  (permno, month, tnic_ret, hq_ret, cov_ret, n_*)
"""
import os
import numpy as np, pandas as pd

R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = f"{R}/data/raw"


def main():
    sm = pd.read_parquet(f"{R}/data/derived/stock_month.parquet",
                         columns=["permno", "month", "ret", "ff48", "gvkey"])
    sm = sm[sm.month >= pd.Period("2018-12", "M")]
    rets = sm[["permno", "month", "ret"]].dropna()

    # ---------------- TNIC ----------------
    l = pd.read_parquet(f"{RAW}/ccm_link.parquet")
    l["linkenddt"] = pd.to_datetime(l.linkenddt).fillna(pd.Timestamp("2099-12-31"))
    l = l[l.linkenddt >= "2019-01-01"].sort_values("linkenddt").drop_duplicates("gvkey", keep="last")
    g2p = dict(zip(l.gvkey, l.permno.astype(int)))
    tn = pd.read_parquet(f"{RAW}/tnic/tnic3_2016plus.parquet")
    tn["g1"] = tn.gvkey1.str.zfill(6).map(g2p); tn["g2"] = tn.gvkey2.str.zfill(6).map(g2p)
    tn = tn.dropna(subset=["g1", "g2"]).astype({"g1": int, "g2": int})
    out = []
    months = sorted(rets.month.unique())
    for t in months:
        y = t.year - 1 if t.month >= 7 else t.year - 2
        y = min(y, 2023)
        e = tn[tn.year == y][["g1", "g2"]].rename(columns={"g1": "permno", "g2": "j"})
        if t.month == 12:
            e.assign(month=t).to_parquet(f"{R}/data/derived/tnic_pairs_{t.year}.parquet", index=False)
        r = rets[rets.month == t].set_index("permno").ret
        e = e[e.j.isin(r.index)]
        e["rj"] = e.j.map(r)
        a = e.groupby("permno").rj.agg(["mean", "count"]).rename(columns={"mean": "tnic_ret", "count": "n_tnic"})
        a["month"] = t
        out.append(a.reset_index())
    tnic = pd.concat(out)

    # ---------------- HQ CBSA (exclude same FF48) ----------------
    co = pd.read_parquet(f"{RAW}/comp_company.parquet")
    co["zip"] = co.addzip.astype(str).str.extract(r"(\d{5})")[0]
    zc = pd.read_parquet(f"{RAW}/zip_county.parquet"); cb = pd.read_parquet(f"{RAW}/county_cbsa.parquet")
    co = co.merge(zc, on="zip", how="left").merge(cb[["county", "cbsa"]], on="county", how="left")
    co = co[co.loc_ == "USA"] if "loc_" in co else co[co["loc"] == "USA"]
    hq = co.dropna(subset=["cbsa"])[["gvkey", "cbsa"]]
    s2 = sm.merge(hq, on="gvkey", how="left").dropna(subset=["ret", "cbsa"])
    tot = s2.groupby(["month", "cbsa"]).ret.agg(["sum", "count"]).reset_index()
    ind = s2.groupby(["month", "cbsa", "ff48"]).ret.agg(["sum", "count"]).reset_index().rename(columns={"sum": "si", "count": "ci"})
    s2 = s2.merge(tot, on=["month", "cbsa"]).merge(ind, on=["month", "cbsa", "ff48"])
    s2["n_hq"] = s2["count"] - s2.ci
    s2["hq_ret"] = (s2["sum"] - s2.si) / s2.n_hq.where(s2.n_hq > 0)
    hqr = s2[["permno", "month", "hq_ret", "n_hq", "cbsa"]]

    # ---------------- analyst co-coverage ----------------
    d = pd.read_parquet(f"{RAW}/ibes_det_coverage.parquet")
    lk = pd.read_parquet(f"{RAW}/ibes_crsp_link.parquet"); lk = lk[lk.score <= 2]
    lk = lk.sort_values("edate").drop_duplicates("ticker", keep="last")
    d["permno"] = d.ticker.map(dict(zip(lk.ticker, lk.permno.astype(int))))
    d = d.dropna(subset=["permno"]).astype({"permno": int})
    d = d[d.analys > 0].drop_duplicates(["permno", "analys", "yr"])
    out = []
    for yr in sorted(d.yr.unique()):
        x = d[d.yr == yr][["permno", "analys"]]
        p = x.merge(x, on="analys", suffixes=("", "_j"))
        p = p[p.permno != p.permno_j].groupby(["permno", "permno_j"]).size().rename("ncov").reset_index()
        p["yr"] = yr + 1                    # used in the following calendar year
        out.append(p)
    cc = pd.concat(out)
    cc.to_parquet(f"{R}/data/derived/cocoverage_pairs.parquet", index=False)
    out = []
    for t in months:
        e = cc[cc.yr == t.year]
        r = rets[rets.month == t].set_index("permno").ret
        e = e[e.permno_j.isin(r.index)].copy()
        e["wr"] = e.ncov * e.permno_j.map(r)
        a = e.groupby("permno").agg(sw=("ncov", "sum"), swr=("wr", "sum"), n_cov=("ncov", "size"))
        a["cov_ret"] = a.swr / a.sw; a["month"] = t
        out.append(a[["cov_ret", "n_cov", "month"]].reset_index())
    cov = pd.concat(out)

    res = rets[["permno", "month"]].merge(tnic, on=["permno", "month"], how="left") \
        .merge(hqr, on=["permno", "month"], how="left").merge(cov, on=["permno", "month"], how="left")
    res.to_parquet(f"{R}/data/derived/control_peers.parquet", index=False)
    print(res.describe().T)


if __name__ == "__main__":
    main()
