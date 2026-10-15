"""
07_characteristics.py -- monthly stock panel for ALL CRSP common stocks 2017-2025:
returns, size, price, industry (FF48 / FF12 / SIC3), accounting characteristics
(lagged so they are public), analyst coverage, HQ CBSA, TNIC peers, analyst
co-coverage, and industry / geographic / product / co-coverage peer returns.
Output: data/derived/stock_month.parquet, data/derived/peer_returns_controls.parquet
"""
import numpy as np, pandas as pd, duckdb, re, os

R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = f"{R}/data/raw"


def ff48_map():
    # Ken French 48-industry SIC definitions (compact ranges)
    txt = open(f"{RAW}/Siccodes48.txt").read()
    rows, cur = [], None
    for line in txt.splitlines():
        m = re.match(r"^\s*(\d+)\s+(\w+)\s+(.*)$", line)
        if m:
            cur = int(m.group(1)); continue
        m = re.match(r"^\s*(\d{4})-(\d{4})", line)
        if m and cur is not None:
            rows.append((cur, int(m.group(1)), int(m.group(2))))
    return rows


def assign_ff48(sic, rows):
    out = np.full(len(sic), 48)
    s = np.asarray(sic, dtype=float)
    for k, lo, hi in rows:
        out[(s >= lo) & (s <= hi)] = k
    out[np.isnan(s)] = -1
    return out


def main():
    msf = pd.read_parquet(f"{RAW}/crsp_msf_v2.parquet")
    msf = msf[(msf.sharetype == "NS") & (msf.securitytype == "EQTY") & (msf.securitysubtype == "COM")
              & msf.primaryexch.isin(["N", "A", "Q"])].copy()
    msf["month"] = pd.to_datetime(msf.mthcaldt).dt.to_period("M")
    msf = msf.sort_values(["permno", "month"]).drop_duplicates(["permno", "month"])
    msf["ret"] = msf.mthret.astype(float)
    msf["me"] = msf.mthcap.astype(float) / 1e3             # $ millions
    msf["prc"] = msf.mthprc.abs()
    rows = ff48_map()
    msf["ff48"] = assign_ff48(msf.siccd, rows)
    msf["sic3"] = (msf.siccd // 10).astype("Int64")
    # momentum r(t-11,t-1) and lagged returns
    g = msf.groupby("permno")
    msf["lr"] = np.log1p(msf.ret.clip(lower=-0.99))
    msf["mom"] = np.expm1(g.lr.transform(lambda s: s.shift(1).rolling(11, min_periods=8).sum()))
    msf["ret_l1"] = g.ret.shift(1)
    msf["nyse"] = msf.primaryexch.eq("N")

    # ---------- industry return (VW, leave-focal-out), FF48 ----------
    msf["wlag"] = g.me.shift(1)
    msf["rw"] = msf.ret * msf.wlag
    tot = msf.groupby(["month", "ff48"]).agg(S=("rw", "sum"), Wt=("wlag", "sum"), N=("ret", "count")).reset_index()
    msf = msf.merge(tot, on=["month", "ff48"], how="left")
    msf["indret"] = (msf.S - msf.rw.fillna(0)) / (msf.Wt - msf.wlag.fillna(0))
    msf.loc[msf.N < 3, "indret"] = np.nan
    msf = msf.drop(columns=["S", "Wt", "N", "rw"])

    # ---------- Compustat annual characteristics (public 6 months after FYE) ----------
    a = pd.read_parquet(f"{RAW}/comp_funda.parquet")
    a["datadate"] = pd.to_datetime(a.datadate)
    ps = a.pstkrv.fillna(a.pstkl).fillna(a.pstk).fillna(0)
    se = a.seq.fillna(a.ceq + a.pstk.fillna(0)).fillna(a["at"] - a["lt"])
    a["be"] = se + a.txditc.fillna(0) - ps
    a["gpa"] = (a.revt - a.cogs) / a["at"]
    a = a.sort_values(["gvkey", "datadate"])
    a["ag"] = a.groupby("gvkey")["at"].pct_change(fill_method=None)
    a["ik"] = a.capx / a.ppent.where(a.ppent > 0)
    a["sale_ann"] = a.sale
    l = pd.read_parquet(f"{RAW}/ccm_link.parquet")
    l["linkdt"] = pd.to_datetime(l.linkdt); l["linkenddt"] = pd.to_datetime(l.linkenddt).fillna(pd.Timestamp("2099-12-31"))
    a = a.merge(l[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey")
    a = a[(a.datadate >= a.linkdt) & (a.datadate <= a.linkenddt)].copy()
    a["permno"] = a.permno.astype("int64")
    a["avail"] = (a.datadate + pd.offsets.MonthEnd(6)).dt.to_period("M")
    a = a.sort_values(["permno", "avail", "datadate"]).drop_duplicates(["permno", "avail"], keep="last")
    acc = a[["permno", "avail", "be", "gpa", "ag", "ik", "sale_ann", "gvkey", "datadate"]].rename(columns={"avail": "month"})
    msf = msf.sort_values(["month", "permno"])
    acc = acc.sort_values(["month", "permno"])
    msf = merge_asof_period(msf, acc)
    msf["bm"] = np.log(msf.be / msf.me.where(msf.me > 0)).where(msf.be > 0)
    msf["size"] = np.log(msf.me)

    # ---------- IBES coverage (FY1 numest at month end) ----------
    st = pd.read_parquet(f"{RAW}/ibes_statsum.parquet")
    st = st[st.fpi == "1"].copy()
    st["month"] = pd.to_datetime(st.statpers).dt.to_period("M")
    lk = pd.read_parquet(f"{RAW}/ibes_crsp_link.parquet")
    lk = lk[lk.score <= 2]
    st = st.merge(lk[["ticker", "permno", "sdate", "edate"]], on="ticker")
    st["permno"] = st.permno.astype("int64")
    st = st[(pd.to_datetime(st.statpers) >= pd.to_datetime(st.sdate)) & (pd.to_datetime(st.statpers) <= pd.to_datetime(st.edate))]
    cov = st.groupby(["permno", "month"]).numest.max().reset_index()
    msf = msf.merge(cov, on=["permno", "month"], how="left")
    msf["numest"] = msf.numest.fillna(0)

    keep = ["permno", "month", "ret", "ret_l1", "mom", "me", "size", "prc", "nyse", "siccd", "sic3", "ff48",
            "indret", "bm", "gpa", "ag", "ik", "sale_ann", "gvkey", "numest", "wlag"]
    msf[keep].to_parquet(f"{R}/data/derived/stock_month.parquet", index=False)
    print("stock_month", msf.shape, msf.month.min(), msf.month.max())


def merge_asof_period(left, right):
    L = left.copy(); Rr = right.copy()
    L["_t"] = L.month.dt.to_timestamp(); Rr["_t"] = Rr.month.dt.to_timestamp()
    L["permno"] = L.permno.astype("int64"); Rr["permno"] = Rr.permno.astype("int64")
    L = L.sort_values("_t"); Rr = Rr.sort_values("_t").drop(columns=["month"])
    out = pd.merge_asof(L, Rr, on="_t", by="permno", direction="backward", tolerance=pd.Timedelta(days=550))
    return out.drop(columns=["_t"])


if __name__ == "__main__":
    main()
