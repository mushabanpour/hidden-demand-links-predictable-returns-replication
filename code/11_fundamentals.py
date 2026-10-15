"""
11_fundamentals.py -- inputs for the mechanism tests.

 firm_spend.parquet  monthly card spend per firm, normalised by total panel spend
                     (SafeGraph's institution panel changes over time), with year-
                     over-year log growth of the firm's share of panel spend
 earnings.parquet    quarterly announcements: rdq; SUE used in the paper = sue_sd (seasonal
                     change in EPS / its s.d. over the prior 8 quarters, at least 4; sue_p,
                     price-scaled, is kept for reference); IBES forecast error fe_p (actual -
                     last median forecast, / price at the end of the announcement month);
                     revenue surprise sur_sale (seasonal change in sales / its s.d. over the
                     prior 8 quarters, at least 4); announcement abnormal returns CAR[-1,+1]
                     (market-adjusted, CRSP daily)
"""
import glob, os
import numpy as np, pandas as pd, duckdb

R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = f"{R}/data/raw"


def spend():
    con = duckdb.connect()
    tot = con.execute(f"""SELECT month, sum(spend) tot, sum(w) totw FROM read_parquet('{R}/data/extract/market_month/*.parquet')
                          GROUP BY month""").df()
    fm = con.execute(f"""SELECT month, permno, spend, w_sum, txn, n_poi, ospend FROM read_parquet('{R}/data/derived/firm/firm_month/*.parquet')""").df()
    fm = fm.merge(tot, on="month")
    fm["month"] = pd.PeriodIndex(fm.month, freq="M")
    fm["share"] = fm.spend / fm.tot
    fm = fm.sort_values(["permno", "month"])
    lag = fm[["permno", "month", "share", "w_sum"]].copy(); lag["month"] = lag.month + 12
    fm = fm.merge(lag.rename(columns={"share": "share_l12", "w_sum": "w_l12"}), on=["permno", "month"], how="left")
    fm["g_yoy"] = np.log(fm.share / fm.share_l12)
    fm.loc[(fm.w_sum < 500) | (fm.w_l12 < 500), "g_yoy"] = np.nan
    fm.to_parquet(f"{R}/data/derived/firm_spend.parquet", index=False)
    print("firm_spend", fm.shape, fm.g_yoy.describe().round(3).to_dict())


def earnings():
    q = pd.read_parquet(f"{RAW}/comp_fundq.parquet")
    q["datadate"] = pd.to_datetime(q.datadate); q["rdq"] = pd.to_datetime(q.rdq)
    q = q.dropna(subset=["rdq"]).sort_values(["gvkey", "datadate"]).drop_duplicates(["gvkey", "datadate"], keep="last")
    q["eps_adj"] = q.epspxq / q.ajexq
    g = q.groupby("gvkey")
    q["d_eps"] = q.eps_adj - g.eps_adj.shift(4)
    q["sd_d"] = g.d_eps.transform(lambda s: s.shift(1).rolling(8, min_periods=4).std())
    q["prc_adj"] = q.prccq / q.ajexq
    q["sue_p"] = q.d_eps / g.prc_adj.shift(4)           # Livnat-Mendenhall style, price-scaled
    q["sue_sd"] = q.d_eps / q.sd_d
    q["d_sale"] = q.saleq - g.saleq.shift(4)
    q["sd_ds"] = g.d_sale.transform(lambda s: s.shift(1).rolling(8, min_periods=4).std())
    q["sur_sale"] = q.d_sale / q.sd_ds
    q["sale_g"] = np.log(q.saleq / g.saleq.shift(4))
    l = pd.read_parquet(f"{RAW}/ccm_link.parquet")
    l["linkdt"] = pd.to_datetime(l.linkdt); l["linkenddt"] = pd.to_datetime(l.linkenddt).fillna(pd.Timestamp("2099-12-31"))
    q = q.merge(l[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey")
    q = q[(q.datadate >= q.linkdt) & (q.datadate <= q.linkenddt)].copy()
    q["permno"] = q.permno.astype(int)
    q = q[q.rdq >= "2019-01-01"]

    # IBES forecast error: actual minus the last consensus (FY quarter, fpi 6) before the announcement
    st = pd.read_parquet(f"{RAW}/ibes_statsum.parquet"); st = st[st.fpi == "6"].copy()
    lk = pd.read_parquet(f"{RAW}/ibes_crsp_link.parquet"); lk = lk[lk.score <= 2]
    lk = lk.sort_values("edate").drop_duplicates("ticker", keep="last")
    st["permno"] = st.ticker.map(dict(zip(lk.ticker, lk.permno.astype(int))))
    st = st.dropna(subset=["permno", "anndats_act", "actual", "medest"])
    st["statpers"] = pd.to_datetime(st.statpers); st["anndats_act"] = pd.to_datetime(st.anndats_act)
    st = st[st.statpers < st.anndats_act].sort_values("statpers").groupby(["permno", "fpedats"]).tail(1)
    st["fe_raw"] = st.actual - st.medest
    st["permno"] = st.permno.astype(int)
    st = st[["permno", "anndats_act", "fe_raw", "numest"]].rename(columns={"anndats_act": "rdq_ibes"})
    q = pd.merge_asof(q.sort_values("rdq"), st.sort_values("rdq_ibes"), left_on="rdq", right_on="rdq_ibes",
                      by="permno", direction="nearest", tolerance=pd.Timedelta(days=3))

    # announcement CAR[-1,+1] (market-adjusted)
    d = pd.read_parquet(f"{RAW}/crsp_dsf_sample.parquet")
    d["date"] = pd.to_datetime(d.dlycaldt)
    ffd = pd.read_parquet(f"{RAW}/ff_daily.parquet"); ffd["date"] = pd.to_datetime(ffd.date)
    d = d.merge(ffd[["date", "mktrf", "rf"]], on="date", how="left")
    d["ar"] = d.dlyret - (d.mktrf + d.rf)
    d = d.sort_values(["permno", "date"])
    d["k"] = d.groupby("permno").cumcount()
    q = q[q.permno.isin(d.permno.unique())]
    # trading-day index of the announcement (first trading day >= rdq)
    dd = d[["permno", "date", "k"]]
    q = pd.merge_asof(q.sort_values("rdq"), dd.sort_values("date"), left_on="rdq", right_on="date", by="permno",
                      direction="forward", tolerance=pd.Timedelta(days=5))
    ar = d.set_index(["permno", "k"]).ar
    def car(row, a, b):
        if pd.isna(row.k):
            return np.nan
        idx = [(row.permno, int(row.k) + s) for s in range(a, b + 1)]
        v = ar.reindex(idx)
        return v.sum() if v.notna().sum() == len(idx) else np.nan
    q["car_m1p1"] = [car(r, -1, 1) for r in q.itertuples()]
    q["car_p2p60"] = [car(r, 2, 60) for r in q.itertuples()]
    q["ann_month"] = q.date.dt.to_period("M")
    q = q.merge(pd.read_parquet(f"{R}/data/derived/stock_month.parquet", columns=["permno", "month", "prc"])
                .rename(columns={"month": "ann_month", "prc": "prc_m"}), on=["permno", "ann_month"], how="left")
    q["fe_p"] = q.fe_raw / q.prc_m
    keep = ["permno", "gvkey", "datadate", "rdq", "date", "ann_month", "sue_p", "sue_sd", "sur_sale", "sale_g",
            "fe_p", "numest", "car_m1p1", "car_p2p60"]
    q[keep].to_parquet(f"{R}/data/derived/earnings.parquet", index=False)
    print("earnings", q.shape, q[keep].describe().T[["count", "mean", "std"]])


if __name__ == "__main__":
    spend()
    earnings()
