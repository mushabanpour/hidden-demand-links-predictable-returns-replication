"""27_intramonth.py -- where in month t+1 is the predictable return earned? (stale-price check)"""
import os
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")
R = ap.R
P = M.load("co")
d = pd.read_parquet(f"{R}/data/raw/crsp_dsf_sample.parquet"); d["date"] = pd.to_datetime(d.dlycaldt)
d = d.dropna(subset=["dlyret"]); d["month"] = d.date.dt.to_period("M")
d["k"] = d.groupby(["permno", "month"]).date.rank(method="first")
d["lr"] = np.log1p(d.dlyret.clip(lower=-0.99))
seg = d.assign(part=np.select([d.k <= 1, d.k <= 5], ["d1", "w1"], "rest")).groupby(["permno", "month", "part"]).lr.sum().unstack().apply(np.expm1)
seg = seg.reset_index(); seg["month"] = seg.month - 1          # align with formation month t
P = P.merge(seg, on=["permno", "month"], how="left")
CTL = ["ret", "ind_u", "indret", "cat_ew", "cat_vw", "STORE_pos", "tnic_ret", "size", "bm", "gpa", "ik", "mom"]
out = {}
for y in ["d1", "w1", "rest", "ret_f1"]:
    r, *_ = ap.fama_macbeth(P, y, ["sig"] + CTL)
    out[y] = r["sig"]
    ew, vw, n = ap.sort_returns(P, "sig", 5, ret=y)
    print(y, "FM %.3f (%.2f)" % r["sig"], "EW LS %.2f (%.2f)" % (ap.nw_mean(ew.LS)[0] * 100, ap.nw_mean(ew.LS)[1]),
          "VW LS %.2f (%.2f)" % (ap.nw_mean(vw.LS)[0] * 100, ap.nw_mean(vw.LS)[1]))
json.dump(out, open(f"{R}/logs/intramonth.json", "w"), default=float, indent=1)
