"""
31_common_ownership.py -- Anton and Polk (2014) common ownership between sample firms.

For quarter q and firms i, j:
   FCAP_ijq = sum_{f holds both i and j} (S_if P_i + S_jf P_j) / (ME_i + ME_j)
the value of the two firms' shares held by 13F institutions that hold BOTH, relative to
the two firms' combined market capitalization. Holdings for quarter q are used from the
second month after the quarter end (13F filings are due within 45 days).
Output: data/derived/fcap_pairs.parquet (q, avail, i, j, fcap)
"""
import os
import numpy as np, pandas as pd, psycopg2, warnings
from scipy import sparse
warnings.filterwarnings("ignore")
R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pull():
    P = pd.read_parquet(f"{R}/data/derived/panel_final.parquet", columns=["permno"])
    c = psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu', port=9737, dbname='wrds', user=os.environ["WRDS_USER"],
                         sslmode='require', connect_timeout=90)
    n = pd.read_sql("select permno, cusip, nameenddt from crsp.stocknames_v2 where nameenddt >= '2018-06-01'", c)
    n = n[n.permno.isin(P.permno.unique())]
    cus = sorted(set(n.cusip.dropna()))
    lst = ",".join(f"'{x}'" for x in cus)
    h = pd.read_sql(f"""select fdate, mgrno, cusip, sum(shares) as shares
                        from tfn.s34 where fdate >= '2018-09-30' and fdate = rdate and cusip in ({lst})
                          and shares > 0
                        group by fdate, mgrno, cusip""", c)
    cmap = n.sort_values("nameenddt").drop_duplicates("cusip", keep="last")[["cusip", "permno"]]
    h = h.merge(cmap, on="cusip")
    h.to_parquet(f"{R}/data/raw/s34_holdings_sample.parquet", index=False)
    print("holdings", h.shape, h.fdate.min(), h.fdate.max(), h.mgrno.nunique())
    return h


def fcap(h):
    msf = pd.read_parquet(f"{R}/data/raw/crsp_msf_v2.parquet", columns=["permno", "mthcaldt", "mthprc", "shrout", "mthcap"])
    msf["q"] = pd.to_datetime(msf.mthcaldt).dt.to_period("Q")
    msf = msf.sort_values("mthcaldt").groupby(["permno", "q"]).tail(1)
    h["q"] = pd.to_datetime(h.fdate).dt.to_period("Q")
    out = []
    for q, d in h.groupby("q"):
        px = msf[msf.q == q].set_index("permno")
        d = d[d.permno.isin(px.index)].groupby(["permno", "mgrno"], as_index=False).shares.sum()
        d["val"] = d.shares * d.permno.map(px.mthprc.abs())
        me = px.mthcap.astype(float)                          # $ thousands in CIZ
        firms = pd.Index(sorted(d.permno.unique())); mgrs = pd.Index(sorted(d.mgrno.unique()))
        r = firms.get_indexer(d.permno); cidx = mgrs.get_indexer(d.mgrno)
        m = me.reindex(firms).values
        den = m[:, None] + m[None, :]
        top = d.groupby("mgrno").val.sum().nlargest(10).index        # largest holders ~ index-fund families
        res = {}
        for nm, dd in [("fcap", d), ("fcap_x", d[~d.mgrno.isin(top)])]:
            r = firms.get_indexer(dd.permno); cidx = mgrs.get_indexer(dd.mgrno)
            H = sparse.csr_matrix((dd.val.values / 1e3, (r, cidx)), shape=(len(firms), len(mgrs)))   # $ thousands
            B = sparse.csr_matrix((np.ones(len(dd)), (r, cidx)), shape=H.shape)
            F = (H @ B.T + B @ H.T).toarray() / den
            np.fill_diagonal(F, np.nan)
            res[nm] = F
        io = np.asarray(sparse.csr_matrix((d.val.values / 1e3, (firms.get_indexer(d.permno), mgrs.get_indexer(d.mgrno))),
                                          shape=(len(firms), len(mgrs))).sum(1)).ravel() / m
        ii, jj = np.where(~np.isnan(res["fcap"]))
        out.append(pd.DataFrame({"q": q, "i": firms.values[ii], "j": firms.values[jj], "fcap": res["fcap"][ii, jj],
                                 "fcap_x": res["fcap_x"][ii, jj], "io_i": io[ii], "io_j": io[jj]}))
        print(q, len(firms), len(mgrs), flush=True)
    X = pd.concat(out, ignore_index=True)
    X["avail"] = X.q.dt.end_time.dt.to_period("M") + 2           # usable from two months after quarter end
    X.to_parquet(f"{R}/data/derived/fcap_pairs.parquet", index=False)
    print(X[['fcap','fcap_x','io_i']].describe())


if __name__ == "__main__":
    import os, sys
    h = pd.read_parquet(f"{R}/data/raw/s34_holdings_sample.parquet") if os.path.exists(f"{R}/data/raw/s34_holdings_sample.parquet") else pull()
    fcap(h)
