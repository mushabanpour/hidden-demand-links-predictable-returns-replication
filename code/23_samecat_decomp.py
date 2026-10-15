"""23_samecat_decomp.py -- is same-category overlap geography (co-location) or popularity/preference?"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")
R = ap.R
P = pd.read_parquet(f"{R}/data/derived/panel_ovl.parquet")
rows = []
for t, d in P.groupby("month"):
    S = pd.read_parquet(f"{R}/data/derived/sim/{t}.parquet", columns=["i", "j", "STORE", "HOME", "RPS"])
    O = pd.read_parquet(f"{R}/data/derived/ovl/{t}.parquet", columns=["i", "j", "A_S", "E_S"])
    U = d.set_index("permno")
    S = S[S.i.isin(U.index) & S.j.isin(U.index)].merge(O, on=["i", "j"], how="left")
    S["rj"] = S.j.map(U.ret)
    S["samecat"] = (S.i.map(U.cat).values == S.j.map(U.cat).values)
    S["bigj"] = S.j.map(U.me)
    o = {}
    for nm, w in [("sc_store", S.STORE.clip(lower=0) * S.samecat), ("sc_home", S.HOME.clip(lower=0) * S.samecat),
                  ("sc_rps", S.RPS.clip(lower=0) * S.samecat), ("sc_me", S.bigj * S.samecat),
                  ("sc_store_me", S.STORE.clip(lower=0) * S.bigj * S.samecat)]:
        w = w.fillna(0)
        o[nm] = (w * S.rj).groupby(S.i).sum() / w.groupby(S.i).sum().where(lambda s: s > 0)
    o = pd.DataFrame(o); o["month"] = t
    rows.append(o.reset_index().rename(columns={"i": "permno"}))
P = P.merge(pd.concat(rows), on=["permno", "month"], how="left")
new = ["sc_store", "sc_home", "sc_rps", "sc_me", "sc_store_me"]
for c in new:
    P[c] = P.groupby("month")[c].transform(ap.winsor)
    P[c] = P[c].fillna(P.groupby("month")[c].transform("mean"))
P.to_parquet(f"{R}/data/derived/panel_ovl2.parquet", index=False)
if "eval" not in sys.argv:
    sys.exit(0)
CTL = ["ret", "ind_u", "indret", "cat_ew", "cat_vw", "STORE_pos", "tnic_ret", "size", "bm", "gpa", "ik", "mom"]
D = P[P.ov_A_S.notna()]
for xs in [["ov_A_S"], ["sc_store"], ["sc_home"], ["sc_rps"], ["sc_me"], ["sc_store_me"], ["ov_E_S"],
           ["ov_A_S", "sc_store"], ["ov_A_S", "sc_store_me"], ["ov_E_S", "sc_store_me"], ["ov_A_S", "sc_me"],
           ["ov_X_S", "ov_E_S", "sc_store_me"]]:
    r, *_ = ap.fama_macbeth(D, "ret_f1", xs + CTL)
    print(xs, {k: (round(r[k][0], 3), round(r[k][1], 2)) for k in xs})
for c in ["ov_A_S", "sc_store", "sc_store_me", "sc_me"]:
    ew, vw, n = ap.sort_returns(D, c, 5)
    print(c, "EW a6 %.2f (%.2f)" % (ap.ts_reg(ew.LS, M.Fn, "FF6")["alpha"] * 100, ap.ts_reg(ew.LS, M.Fn, "FF6")["t_alpha"]),
          "VW a6 %.2f (%.2f)" % (ap.ts_reg(vw.LS, M.Fn, "FF6")["alpha"] * 100, ap.ts_reg(vw.LS, M.Fn, "FF6")["t_alpha"]),
          "by year EW", (ew.LS.groupby(ew.index.year).mean() * 100).round(2).to_dict())
P.to_parquet(f"{R}/data/derived/panel_ovl2.parquet", index=False)
