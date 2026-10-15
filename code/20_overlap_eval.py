"""20_overlap_eval.py -- peer signals from the overlap family + evaluation (exploratory)."""
import sys, glob, os
import numpy as np, pandas as pd, duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")
R = ap.R


def signals(P):
    rows = []
    for t, d in P.groupby("month"):
        f = f"{R}/data/derived/ovl/{t}.parquet"
        if not os.path.exists(f):
            continue
        S = pd.read_parquet(f)
        U = d.set_index("permno")
        S = S[S.i.isin(U.index) & S.j.isin(U.index)].copy()
        S["rj"] = S.j.map(U.ret)
        S["A_M"] = S[["A_S", "A_P"]].max(1); S["E_M"] = S[["E_S", "E_P"]].max(1); S["X_M"] = S.A_M - S.E_M
        S["same_ind"] = S.i.map(U.ff48).values == S.j.map(U.ff48).values
        o = {}
        for c in ["A_S", "E_S", "X_S", "A_P", "E_P", "X_P", "A_M", "E_M", "X_M"]:
            w = S[c].clip(lower=0).fillna(0)
            for nm, msk in [("", np.ones(len(S), bool)), ("_cross", ~S.same_ind.values), ("_same", S.same_ind.values)]:
                ww = w * msk
                num = (ww * S.rj).groupby(S.i).sum(); den = ww.groupby(S.i).sum()
                o[f"ov_{c}{nm}"] = num / den.where(den > 0)
                if nm == "":
                    o[f"deg_{c}"] = den
        o = pd.DataFrame(o); o["month"] = t
        rows.append(o.reset_index().rename(columns={"i": "permno"}))
    return P.merge(pd.concat(rows), on=["permno", "month"], how="left")


def within_cat(P):
    con = duckdb.connect()
    fm = con.execute(f"SELECT month, permno, cat FROM read_parquet('{R}/data/derived/firm/firm_month/*.parquet')").df()
    fm["month"] = pd.PeriodIndex(fm.month, freq="M")
    if os.environ.get("CAT_METHOD") == "full":          # full-sample modal category (old rule, for comparison only)
        cat = fm.groupby("permno").cat.agg(lambda s: s.mode().iat[0])
        P["cat"] = P.permno.map(cat)
    else:                                                # real-time: most frequent monthly category over t-12..t-1
        out = []
        for t in sorted(P.month.unique()):
            w = fm[(fm.month >= t - 12) & (fm.month <= t - 1)]
            c = w.groupby(["permno", "cat"]).month.agg(["size", "max"]).reset_index()
            c = c.sort_values(["permno", "size", "max"]).groupby("permno").tail(1)   # ties -> most recent month
            out.append(c[["permno", "cat"]].assign(month=t))
        P = P.drop(columns=["cat"], errors="ignore").merge(pd.concat(out), on=["permno", "month"], how="left")
    g = P.groupby(["month", "cat"])
    n = g.ret.transform("count"); s = g.ret.transform("sum")
    P["cat_ew"] = ((s - P.ret) / (n - 1)).where(n >= 2)
    P["_w"] = P.me; P["_wr"] = P.me * P.ret
    sw = g._w.transform("sum"); swr = g._wr.transform("sum")
    P["cat_vw"] = ((swr - P._wr) / (sw - P._w)).where(n >= 2)
    return P


if __name__ == "__main__":
    P = M.load("RPS_pos", src="panel")
    P = within_cat(signals(P))
    sig = [c for c in P.columns if c.startswith("ov_")]
    for c in sig + ["cat_ew", "cat_vw"]:
        P[c] = P.groupby("month")[c].transform(lambda s: ap.winsor(s.replace([np.inf, -np.inf], np.nan)))
    for c in ["cat_ew", "cat_vw"]:
        P[c] = P[c].fillna(P.groupby("month")[c].transform("mean"))
    P.to_parquet(f"{R}/data/derived/panel_ovl.parquet", index=False)
    if "eval" not in sys.argv:
        sys.exit(0)
    base = ["ret", "ind_u", "indret"]
    full = base + ["cat_ew", "cat_vw", "STORE_pos", "tnic_ret", "size", "bm", "gpa", "ik", "mom"]
    Fn = M.Fn
    out = []
    for s in sig + ["sig"]:
        ew, vw, n = ap.sort_returns(P, s, 5)
        a_e = ap.ts_reg(ew.LS, Fn, "FF6"); a_v = ap.ts_reg(vw.LS, Fn, "FF6")
        r1, *_ = ap.fama_macbeth(P, "ret_f1", [s] + base)
        r2, *_ = ap.fama_macbeth(P, "ret_f1", [s] + full)
        out.append(dict(sig=s, N=int(n.sum(1).mean()), ew_a=a_e["alpha"] * 100, ew_t=a_e["t_alpha"], vw_a=a_v["alpha"] * 100,
                        vw_t=a_v["t_alpha"], fm_b=r1[s][0], fm_bt=r1[s][1], fm_f=r2[s][0], fm_ft=r2[s][1]))
    O = pd.DataFrame(out).round(3)
    pd.set_option("display.width", 200)
    print(O.to_string(index=False))
    for xs in [["ov_X_S", "ov_E_S"], ["ov_X_M", "ov_E_M"], ["ov_A_S", "sig"], ["ov_X_S", "sig"]]:
        r, *_ = ap.fama_macbeth(P, "ret_f1", xs + full)
        print(xs, {k: (round(r[k][0], 3), round(r[k][1], 2)) for k in xs})
    P.to_parquet(f"{R}/data/derived/panel_ovl.parquet", index=False)
