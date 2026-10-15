"""
22_weekly.py -- weekly lead-lag with the same real-time networks.

Weeks run Wednesday close to Wednesday close (Lo-MacKinlay 1990). For a week ending in
calendar month m, the network is the one formed at the end of month m-1 (card data
through m-2), and the universe is that month's test universe. Signal: network-weighted
peer return in week w; outcome: focal return in week w+1 (and w+2 to skip a week).
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
R = ap.R


def weekly_returns():
    d = pd.read_parquet(f"{R}/data/raw/crsp_dsf_sample.parquet")
    d["date"] = pd.to_datetime(d.dlycaldt)
    d = d.dropna(subset=["dlyret"])
    d["wk"] = (d.date - pd.to_timedelta((d.date.dt.weekday - 2) % 7, unit="D"))   # week ends on Wednesday
    d["wk"] = d.wk.where(d.date.dt.weekday == 2, d.wk + pd.Timedelta(days=7))
    d["lr"] = np.log1p(d.dlyret.clip(lower=-0.99))
    w = d.groupby(["permno", "wk"]).agg(lr=("lr", "sum"), n=("lr", "size"), cap=("dlycap", "last")).reset_index()
    w["r"] = np.expm1(w.lr)
    ffd = pd.read_parquet(f"{R}/data/raw/ff_daily.parquet"); ffd["date"] = pd.to_datetime(ffd.date)
    ffd["wk"] = (ffd.date - pd.to_timedelta((ffd.date.dt.weekday - 2) % 7, unit="D"))
    ffd["wk"] = ffd.wk.where(ffd.date.dt.weekday == 2, ffd.wk + pd.Timedelta(days=7))
    for c in ["mktrf", "smb", "hml", "umd", "rf"]:
        ffd["l_" + c] = np.log1p(ffd[c])
    fw = ffd.groupby("wk")[[c for c in ffd if c.startswith("l_")]].sum().apply(np.expm1)
    fw.columns = [c[2:] for c in fw.columns]
    return w, fw


def main(nets=("RPS", "STORE", "HOME", "INC", "RPS_D", "RAW")):
    w, fw = weekly_returns()
    P = pd.read_parquet(f"{R}/data/derived/panel.parquet", columns=["permno", "month", "ff48", "me"])
    weeks = sorted(w.wk.unique())
    W = w.set_index(["wk", "permno"]).r
    rows = []
    for k in range(len(weeks) - 2):
        wk = pd.Timestamp(weeks[k]); m = wk.to_period("M") - 1
        f = f"{R}/data/derived/sim/{m}.parquet"
        if not os.path.exists(f):
            continue
        U = P[P.month == m]
        if U.empty:
            continue
        if k == 0 or getattr(main, "_m", None) != m:
            S = pd.read_parquet(f, columns=["i", "j"] + list(nets))
            S = S[S.i.isin(U.permno) & S.j.isin(U.permno)]
            main._m, main._S = m, S
        S = main._S
        r0 = W.loc[wk] if wk in W.index.get_level_values(0) else None
        if r0 is None:
            continue
        r1 = W.loc[pd.Timestamp(weeks[k + 1])]; r2 = W.loc[pd.Timestamp(weeks[k + 2])]
        x = S.copy(); x["rj"] = x.j.map(r0)
        x = x.dropna(subset=["rj"])
        o = pd.DataFrame(index=U.permno.values)
        for n in nets:
            wp = x[n].clip(lower=0)
            o[n] = (wp * x.rj).groupby(x.i).sum() / wp.groupby(x.i).sum()
        o["own"] = o.index.map(r0); o["f1"] = o.index.map(r1); o["f2"] = o.index.map(r2)
        ff = dict(zip(U.permno, U.ff48)); o["ff48"] = o.index.map(ff); o["me"] = o.index.map(dict(zip(U.permno, U.me)))
        g = o.groupby("ff48").own
        o["ind_u"] = ((g.transform("sum") - o.own) / (g.transform("count") - 1)).where(g.transform("count") >= 2)
        o["wk"] = wk
        rows.append(o.reset_index().rename(columns={"index": "permno"}))
    D = pd.concat(rows, ignore_index=True)
    D["size"] = np.log(D.me)
    for c in list(nets) + ["own", "ind_u"]:
        D[c] = D.groupby("wk")[c].transform(ap.winsor)
    D.to_parquet(f"{R}/data/derived/weekly_panel.parquet", index=False)
    res = {}
    for n in nets:
        for y in ["f1", "f2"]:
            r, r2, nobs, T = ap.fama_macbeth(D, y, [n, "own", "ind_u", "size"], by="wk", lags=6, min_obs=30)
            ew, vw, _ = ap.sort_returns(D.rename(columns={"wk": "month"}), n, 5, ret=y)
            Fw = fw.copy(); Fw.index = Fw.index - pd.Timedelta(days=7 * (1 if y == "f1" else 2))
            m_e, t_e, _ = ap.nw_mean(ew.LS, 6); m_v, t_v, _ = ap.nw_mean(vw.LS, 6)
            import statsmodels.api as sm
            dd = pd.concat([ew.LS.rename("y"), Fw[["mktrf", "smb", "hml", "umd"]]], axis=1).dropna()
            a = sm.OLS(dd.y, sm.add_constant(dd[["mktrf", "smb", "hml", "umd"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
            res[(n, y)] = (r[n], (m_e * 100, t_e), (m_v * 100, t_v), (a.params["const"] * 100, a.tvalues["const"]), T)
            print(n, y, "FM", np.round(r[n], 3), "EW LS", round(m_e * 100, 3), round(t_e, 2), "VW LS", round(m_v * 100, 3),
                  round(t_v, 2), "EW a4", round(a.params["const"] * 100, 3), round(a.tvalues["const"], 2), "weeks", T, flush=True)
    return res


if __name__ == "__main__":
    main()
