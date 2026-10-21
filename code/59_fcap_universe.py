"""
59_fcap_universe.py -- customer overlap against common-ownership momentum computed over all stocks.

From 58_fcap_universe_pull.py, for sample firm i and every other CRSP common stock j with at least one common 13F holder,
   FCAP_ij = (si P_i + sj P_j) / (ME_i + ME_j)                         (Anton and Polk, 2014)
with prices and market capitalizations at the end of the 13F quarter, and FCAPx_ij the same excluding the ten largest
holders of sample firms (Table 12). Each quarter's holdings are used from two months after the quarter end. The
common-ownership peer return of firm i in month t weights the month-t returns of all connected stocks by FCAP
(other share classes of i's issuer excluded):
   ucap_ret     all connected CRSP common stocks, FCAP weights (as in Anton and Polk, 2014, and Ying, 2024)
   ucapx_ret    the same with FCAPx weights
   ucap_out     connected stocks outside the sample universe only
Tests: quintile sorts on each (six-factor alphas) and Fama-MacBeth regressions with the controls of Table 8, column (5),
alone and together with the customer-overlap peer return.
Output: data/derived/fcap_universe_signals.parquet, logs/fcap_universe.json
"""
import os, sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")
R = ap.R


def signals():
    msf = pd.read_parquet(f"{R}/data/raw/crsp_msf_v2.parquet",
                          columns=["permno", "mthcaldt", "mthret", "mthprc", "mthcap", "sharetype", "securitytype"])
    msf["month"] = pd.to_datetime(msf.mthcaldt).dt.to_period("M")
    msf = msf[(msf.sharetype == "NS") & (msf.securitytype == "EQTY")]
    msf["q"] = msf.month.dt.asfreq("Q")
    qend = msf.sort_values("mthcaldt").groupby(["permno", "q"]).tail(1).set_index(["permno", "q"])
    cmap = pd.read_parquet(f"{R}/data/raw/crsp_cusip_permno.parquet").sort_values("nameenddt").drop_duplicates("cusip", keep="last")
    c2p = dict(zip(cmap.cusip, cmap.permno))
    names = pd.read_parquet(f"{R}/data/raw/crsp_stocknames_v2.parquet", columns=["permno", "permco"]).drop_duplicates("permno")
    permco = dict(zip(names.permno, names.permco))
    P = pd.read_parquet(f"{R}/data/derived/panel_final.parquet", columns=["permno", "month"])
    sample = set(P.permno)
    Q = {}
    for f in sorted(glob.glob(f"{R}/data/raw/s34_pairs_universe/*.parquet")):
        q = pd.Period(os.path.basename(f)[:10], "Q")
        d = pd.read_parquet(f)
        d["i"] = d.ci.map(c2p); d["j"] = d.cj.map(c2p)
        d = d.dropna(subset=["i", "j"]); d[["i", "j"]] = d[["i", "j"]].astype(int)
        d = d[d.i.isin(sample) & (d.i != d.j)]
        d = d.groupby(["i", "j"], as_index=False)[["si", "sj", "si_x", "sj_x", "nmgr"]].sum()
        px = qend.xs(q, level="q") if q in qend.index.get_level_values("q") else None
        if px is None:
            continue
        pi, pj = d.i.map(px.mthprc.abs()), d.j.map(px.mthprc.abs())
        mi, mj = d.i.map(px.mthcap), d.j.map(px.mthcap)
        den = (mi + mj) * 1e3                                    # CIZ market cap is in $ thousands
        d["fcap"] = (d.si * pi + d.sj * pj) / den
        d["fcapx"] = (d.si_x * pi + d.sj_x * pj) / den
        d = d.dropna(subset=["fcap"])
        d = d[d.i.map(permco) != d.j.map(permco)]
        Q[q] = d[["i", "j", "fcap", "fcapx"]]
        print(q, len(d), d.i.nunique(), d.j.nunique(), flush=True)
    avail = {q: q.asfreq("M", "end") + 2 for q in Q}
    ret = msf.set_index(["month", "permno"]).mthret
    rows = []
    for t in sorted(P.month.unique()):
        ok = [q for q in Q if avail[q] <= t]
        if not ok:
            continue
        d = Q[max(ok)].copy()
        rt = ret.xs(t, level="month") if t in ret.index.get_level_values("month") else None
        if rt is None:
            continue
        d["rj"] = d.j.map(rt)
        d = d.dropna(subset=["rj"])
        out = {}
        for nm, w, msk in (("ucap_ret", "fcap", None), ("ucapx_ret", "fcapx", None), ("ucap_out", "fcap", ~d.j.isin(sample))):
            x = d if msk is None else d[msk]
            ww = x[w].clip(lower=0)
            out[nm] = (ww * x.rj).groupby(x.i).sum() / ww.groupby(x.i).sum().where(lambda s: s > 0)
        o = pd.DataFrame(out); o["month"] = t
        rows.append(o.reset_index().rename(columns={"i": "permno"}))
    S = pd.concat(rows, ignore_index=True)
    S.to_parquet(f"{R}/data/derived/fcap_universe_signals.parquet", index=False)
    return S


def main():
    S = signals() if not os.environ.get("REUSE") else pd.read_parquet(f"{R}/data/derived/fcap_universe_signals.parquet")
    P = M.load("co").merge(S, on=["permno", "month"], how="left")
    X32 = import_module("32_mechanism_extra")
    *_, Pn = X32.ownership(M.load("co"))
    P = P.merge(Pn[["permno", "month", "fcap_ret", "fcap_ret_cat"]], on=["permno", "month"], how="left")
    for c in ["ucap_ret", "ucapx_ret", "ucap_out", "fcap_ret", "fcap_ret_cat"]:
        P[c] = P.groupby("month")[c].transform(lambda s: ap.winsor(s.replace([np.inf, -np.inf], np.nan)))
    Fn, CTL = M.Fn, M.CONTROLS
    res = {"coverage": float(P.ucap_ret.notna().mean()), "sorts": {}, "fm": {}}
    for s in ["ucap_ret", "ucapx_ret", "ucap_out"]:
        ew, vw, n = ap.sort_returns(P, s, 5)
        res["sorts"][s] = {w: (lambda a: (a["alpha"], a["t_alpha"]))(ap.ts_reg(x.LS, Fn, "FF6")) for w, x in (("ew", ew), ("vw", vw))}
        print("sort", s, {w: (round(100 * v[0], 2), round(v[1], 2)) for w, v in res["sorts"][s].items()})
    D = P[P.sig.notna()].copy()
    for c in ["ucap_ret", "ucapx_ret", "ucap_out", "fcap_ret", "fcap_ret_cat"]:
        D[c] = D[c].fillna(D.groupby("month")[c].transform("mean"))
    specs = {"co": ["sig"], "ucap": ["ucap_ret"], "ucapx": ["ucapx_ret"], "co_ucap": ["sig", "ucap_ret"],
             "co_ucapx": ["sig", "ucapx_ret"], "co_ucap_out": ["sig", "ucap_out"],
             "co_all_own": ["sig", "ucap_ret", "fcap_ret", "fcap_ret_cat"]}
    for k, s in specs.items():
        r, r2, nobs, T = ap.fama_macbeth(D, "ret_f1", s + CTL)
        res["fm"][k] = {"full": {v: r[v] for v in s}, "n": nobs}
        print("fm", k, {v: (round(r[v][0], 3), round(r[v][1], 2)) for v in s})
    res["corr"] = D[["sig", "ucap_ret", "ucapx_ret", "ucap_out", "fcap_ret"]].corr().round(3).to_dict()
    json.dump(res, open(f"{R}/logs/fcap_universe.json", "w"), default=float, indent=1)


if __name__ == "__main__":
    main()
