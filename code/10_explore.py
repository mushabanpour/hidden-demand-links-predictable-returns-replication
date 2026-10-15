"""10_explore.py -- first-pass evaluation of every network/weighting signal."""
import os
import sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap

R = ap.R
P = pd.read_parquet(f"{R}/data/derived/panel.parquet")
F = ap.factors()
P["exret_f1"] = P.ret_f1 - P.month.map(F.rf.shift(-1))
sigs = [c for c in P.columns if any(c.startswith(m + "_") for m in
        ["RPS", "RPS_P", "RPS_D", "RAW", "GEOX", "STORE", "HOME", "INC", "INCX", "OVL", "SAMEC"])
        and not c.endswith("npos")] + ["tnic_ret", "hq_ret", "cov_ret", "indret"]
for c in sigs + ["ret", "mom", "size", "bm", "gpa", "ag"]:
    P[c] = P.groupby("month")[c].transform(lambda s: ap.winsor(s.replace([np.inf, -np.inf], np.nan)))
rows = []
Fn = F.copy(); Fn.index = Fn.index - 1          # factor of t+1 aligned to formation month t
for s in sigs:
    ew, vw, n = ap.sort_returns(P, s, 5)
    r = {"signal": s, "N": int(n.sum(1).mean()) if len(n) else 0, "T": len(ew)}
    for nm, x in [("ew", ew), ("vw", vw)]:
        if x.empty:
            continue
        m, t, _ = ap.nw_mean(x.LS)
        a = ap.ts_reg(x.LS, Fn, "FF6")
        r[f"LS_{nm}"] = m * 100; r[f"t_{nm}"] = t; r[f"a6_{nm}"] = a["alpha"] * 100; r[f"ta6_{nm}"] = a["t_alpha"]
    try:
        res, _, _, _ = ap.fama_macbeth(P, "ret_f1", [s])
        r["fm"] = res[s][0]; r["t_fm"] = res[s][1]
        res, _, _, _ = ap.fama_macbeth(P, "ret_f1", [s, "ret", "indret", "size", "bm", "mom"])
        r["fm_c"] = res[s][0]; r["t_fm_c"] = res[s][1]
    except Exception as e:
        pass
    rows.append(r)
out = pd.DataFrame(rows).round(3)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(out.to_string(index=False))
out.to_csv(f"{R}/logs/explore_signals.csv", index=False)
print("months:", P.month.min(), P.month.max(), "avg universe", P.groupby("month").size().mean())
