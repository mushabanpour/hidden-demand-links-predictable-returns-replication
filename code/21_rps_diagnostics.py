"""21_rps_diagnostics.py -- dose-response (link strength terciles), by-year performance."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")
R = ap.R
P = M.load("RPS_pos")
# dose-response: split each firm's positive-RPS peers into within-firm terciles of RPS
rows = []
for t, d in P.groupby("month"):
    S = pd.read_parquet(f"{R}/data/derived/sim/{t}.parquet", columns=["i", "j", "RPS"])
    U = d.set_index("permno")
    S = S[S.i.isin(U.index) & S.j.isin(U.index) & (S.RPS > 0)].copy()
    S["rj"] = S.j.map(U.ret)
    S["k"] = S.groupby("i").RPS.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
    o = {}
    for k, nm in [(2, "strong"), (1, "medium"), (0, "weak")]:
        x = S[S.k == k]; w = x.RPS
        o[f"d_{nm}"] = (w * x.rj).groupby(x.i).sum() / w.groupby(x.i).sum()
        o[f"m_{nm}"] = x.RPS.groupby(x.i).mean()
    o = pd.DataFrame(o); o["month"] = t
    rows.append(o.reset_index().rename(columns={"i": "permno"}))
P = P.merge(pd.concat(rows), on=["permno", "month"], how="left")
for c in ["d_strong", "d_medium", "d_weak"]:
    P[c] = P.groupby("month")[c].transform(ap.winsor)
CTL = ["ret", "ind_u", "indret", "STORE_pos", "tnic_ret", "size", "bm", "gpa", "ik", "mom"]
r, *_ = ap.fama_macbeth(P, "ret_f1", ["d_strong", "d_medium", "d_weak"] + CTL)
print("dose-response:", {k: (round(r[k][0], 3), round(r[k][1], 2)) for k in ["d_strong", "d_medium", "d_weak"]})
print("mean RPS by tercile:", P[["m_strong", "m_medium", "m_weak"]].mean().round(3).to_dict())
for c in ["d_strong", "d_medium", "d_weak"]:
    ew, vw, n = ap.sort_returns(P, c, 5)
    print(c, "EW a6", round(ap.ts_reg(ew.LS, M.Fn, "FF6")["alpha"] * 100, 2), round(ap.ts_reg(ew.LS, M.Fn, "FF6")["t_alpha"], 2),
          "VW a6", round(ap.ts_reg(vw.LS, M.Fn, "FF6")["alpha"] * 100, 2), round(ap.ts_reg(vw.LS, M.Fn, "FF6")["t_alpha"], 2))
# by year (main signal)
ew, vw, n = ap.sort_returns(P, "sig", 5)
for nm, x in [("EW", ew), ("VW", vw)]:
    print(nm, x.LS.groupby(x.index.year).agg(["mean", "count"]).assign(mean=lambda d: d["mean"] * 100).round(2).T.to_string())
# FM by year
_, _, _, _ = ap.fama_macbeth(P, "ret_f1", ["sig"] + CTL)
P.to_parquet(f"{R}/data/derived/panel_dose.parquet", index=False)
