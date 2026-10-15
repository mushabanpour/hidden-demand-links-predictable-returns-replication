import os
import sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")
R = ap.R
P = pd.read_parquet(f"{R}/data/derived/panel_ovl2.parquet")
sm = pd.read_parquet(f"{R}/data/derived/stock_month.parquet", columns=["permno", "month", "ret", "me", "nyse"])
f2 = sm[["permno", "month", "ret"]].copy(); f2["month"] = f2.month - 2
P = P.merge(f2.rename(columns={"ret": "ret_f2"}), on=["permno", "month"], how="left")
bp = sm[sm.nyse].groupby("month").me.quantile(0.2).rename("nyse20")
P = P.merge(bp, left_on="month", right_index=True, how="left")
F2 = ap.factors(); F2.index = F2.index - 2
CTL = ["ret", "ind_u", "indret", "cat_ew", "cat_vw", "STORE_pos", "tnic_ret", "size", "bm", "gpa", "ik", "mom"]
specs = [("all", P, "ret_f1"), ("ex2020", P[P.month.dt.year != 2020], "ret_f1"), ("ex2020-21", P[~P.month.dt.year.isin([2020, 2021])], "ret_f1"),
         ("h1", P[P.month <= pd.Period("2022-06", "M")], "ret_f1"), ("h2", P[P.month > pd.Period("2022-06", "M")], "ret_f1"),
         ("prc>=5", P[P.prc >= 5], "ret_f1"), ("exmicro", P[P.me >= P.nyse20], "ret_f1"), ("skip", P, "ret_f2")]
for s in ["ov_A_S", "ov_E_S", "sig", "RPS_D_pos", "RPSX_pos"]:
    for nm, D, y in specs:
        D = D[D[s].notna()]
        r, *_ = ap.fama_macbeth(D, y, [s] + CTL)
        ew, vw, n = ap.sort_returns(D, s, 5, ret=y)
        Fx = M.Fn if y == "ret_f1" else F2
        ae = ap.ts_reg(ew.LS, Fx, "FF6"); av = ap.ts_reg(vw.LS, Fx, "FF6")
        print(f"{s:10s} {nm:9s} FM {r[s][0]:.3f} ({r[s][1]:.2f})   EW a6 {ae['alpha']*100:.2f} ({ae['t_alpha']:.2f})  VW a6 {av['alpha']*100:.2f} ({av['t_alpha']:.2f})  T={len(ew)}")
