"""
09_panel.py -- focal-firm x formation-month panel for the asset-pricing tests.

Formation month t (end of month t):
  * networks: data/derived/sim/<t>.parquet  (card data from t-12 .. t-1 only)
  * universe U_t: firm has a measured profile, CRSP common stock with a month-t
    return, price >= $1 at t, and card-panel spend >= MIN_COV of its latest public
    annual sales (consumer-facing firms)
  * signal for network m: peer return in month t,
        PR_m(i,t) = sum_j w_ij r_jt / sum_j w_ij,  j in U_t \\ {i}
    weights: POS = max(S_ij, 0); TOPk = S_ij among the k most similar peers
  * outcome: r_i,t+1 (CRSP CIZ, delisting returns included)
Output: data/derived/panel.parquet
"""
import os, glob
import numpy as np, pandas as pd

R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIN_COV, MIN_PRC = 0.001, 1.0
MEAS = ["RPS", "RPS_P", "RPS_D", "RPSX", "RAW", "GEOX", "STORE", "HOME", "INC", "INCX", "OVL", "SAMEC"]
TOPK = [10, 20, 50]


def wavg(df, wcol, key="i"):
    g = df.assign(_wr=df[wcol] * df.rj).groupby(key)
    s = g[wcol].sum()
    return (g._wr.sum() / s.where(s > 0)), g[wcol].apply(lambda x: (x > 0).sum())


def main():
    sm = pd.read_parquet(f"{R}/data/derived/stock_month.parquet")
    sm = sm[sm.month >= pd.Period("2018-12", "M")]
    cp = pd.read_parquet(f"{R}/data/derived/control_peers.parquet")
    nxt = sm[["permno", "month", "ret"]].copy(); nxt["month"] = nxt.month - 1
    nxt = nxt.rename(columns={"ret": "ret_f1"})
    cc = pd.read_parquet(f"{R}/data/derived/cocoverage_pairs.parquet")
    rows = []
    for f in sorted(glob.glob(f"{R}/data/derived/sim/*.parquet")):
        t = pd.Period(os.path.basename(f)[:7], "M")
        S = pd.read_parquet(f)
        fw = pd.read_parquet(f"{R}/data/derived/sim_firm/{t}.parquet")
        st = sm[sm.month == t].set_index("permno")
        u = fw.set_index("permno").join(st[["ret", "prc", "me", "sale_ann", "ff48", "size"]], how="inner")
        u["cov_ratio"] = (u.spend / u.nmon * 12) / (u.sale_ann * 1e6)
        u = u[u.ret.notna() & (u.prc >= MIN_PRC) & (u.cov_ratio >= MIN_COV)]
        U = u.index
        S = S[S.i.isin(U) & S.j.isin(U)].copy()
        S["rj"] = S.j.map(u.ret)
        # pure-taste similarity: RPS orthogonalised (pair level, month by month) to income and geography networks
        Xc = ["INC", "INCX", "STORE", "HOME"]
        ok = S[["RPS"] + Xc].notna().all(1)
        X = np.column_stack([np.ones(ok.sum())] + [S.loc[ok, c].values for c in Xc])
        b = np.linalg.lstsq(X, S.loc[ok, "RPS"].values, rcond=None)[0]
        S["RPSX"] = np.nan; S.loc[ok, "RPSX"] = S.loc[ok, "RPS"].values - X @ b
        # link visibility: TNIC rival, shared analyst
        tp = pd.read_parquet(f"{R}/data/derived/tnic_pairs_{min(t.year if t.month >= 7 else t.year - 1, 2025)}.parquet")
        tk = set(zip(tp.permno, tp.j))
        S["tnic_link"] = [(a, b_) in tk for a, b_ in zip(S.i, S.j)]
        ck = cc[cc.yr == t.year]; ck = set(zip(ck.permno, ck.permno_j))
        S["cov_link"] = [(a, b_) in ck for a, b_ in zip(S.i, S.j)]
        S["same_ind"] = S.i.map(u.ff48).values == S.j.map(u.ff48).values
        med = u.me.median()
        S["big_j"] = S.j.map(u.me) >= med
        out = {}
        for m in MEAS:
            x = S[["i", "j", "rj", "same_ind", "big_j", "tnic_link", "cov_link", m]].dropna(subset=[m])
            x["wp"] = x[m].clip(lower=0)
            out[f"{m}_pos"], out[f"{m}_npos"] = wavg(x, "wp")
            x["rk"] = x.groupby("i")[m].rank(ascending=False, method="first")
            for k in TOPK:
                x[f"w{k}"] = np.where(x.rk <= k, x.wp, 0.0)
                out[f"{m}_top{k}"], _ = wavg(x, f"w{k}")
            if m in ("RPS", "RPSX", "RPS_D", "STORE", "HOME", "INC", "RAW"):
                # equal-weighted top-20 and splits used in the mechanism / attention tests
                x["e20"] = (x.rk <= 20).astype(float)
                out[f"{m}_eq20"], _ = wavg(x, "e20")
                for nm, msk in [("same", x.same_ind), ("cross", ~x.same_ind), ("big", x.big_j), ("small", ~x.big_j),
                                ("tnic", x.tnic_link), ("ntnic", ~x.tnic_link), ("cov", x.cov_link), ("ncov", ~x.cov_link)]:
                    x["_w"] = np.where(msk, x.w20, 0.0)
                    out[f"{m}_top20_{nm}"], _ = wavg(x, "_w")
                    x["_w"] = np.where(msk, x.wp, 0.0)
                    out[f"{m}_pos_{nm}"], _ = wavg(x, "_w")
        out = pd.DataFrame(out).reindex(U)
        out["n_univ"] = len(U)
        # share of RPS weight on visible links
        x = S[["i", "RPS", "same_ind", "tnic_link", "cov_link"]].dropna(subset=["RPS"]); x["wp"] = x.RPS.clip(lower=0)
        for nm in ["same_ind", "tnic_link", "cov_link"]:
            out[f"rpsw_{nm}"] = (x.wp * x[nm]).groupby(x.i).sum() / x.wp.groupby(x.i).sum()
        out = out.join(u[["cov_ratio", "w", "n_poi", "spend", "nmon", "inc_hi", "inc_lo"]])
        out["month"] = t
        rows.append(out.reset_index().rename(columns={"index": "permno", "i": "permno"}))
        print(t, "universe", len(U), flush=True)
    P = pd.concat(rows, ignore_index=True)
    P = P.merge(sm, on=["permno", "month"], how="left").merge(cp, on=["permno", "month"], how="left") \
         .merge(nxt, on=["permno", "month"], how="left")
    P.to_parquet(f"{R}/data/derived/panel.parquet", index=False)
    print(P.shape, P.month.min(), P.month.max())


if __name__ == "__main__":
    main()
