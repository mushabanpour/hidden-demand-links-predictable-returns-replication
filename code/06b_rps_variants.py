"""
06b_rps_variants.py -- alternative constructions of the RPS network (robustness).

  W6, W24     trailing windows of 6 / 24 months (still ending at t-1)
  K025, K5    kappa = 0.25 / 5 percentage points in the Pearson residual
  LIFT        log-lift profile  log((A+1)/(E+1))  instead of the Pearson residual
  WSAME       profile also includes the same-category merchant list (county x category baseline)
  FROZEN      one network estimated on 2019 data only (Jan-Dec 2019) and held fixed
              for all formation months >= 2020-01 (pure out-of-sample network)
Output: data/derived/sim_var/<t>.parquet (t, i, j, RPS_<variant>)
"""
import os, sys, glob
import numpy as np, pandas as pd, duckdb
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
S6 = import_module("06_similarity")

R = S6.R; F = S6.F
OUT = f"{R}/data/derived/sim_var"


def profiles(win, con, kappa=1.0, lift=False, same=False):
    fl = lambda tab: "[" + ",".join(f"'{F}/{tab}/{m}.parquet'" for m in win) + "]"
    fm = con.execute(f"SELECT permno, sum(w_sum) w FROM read_parquet({fl('firm_month')}) GROUP BY permno").df()
    fm = fm[fm.w >= S6.MIN_CUST].sort_values("permno").reset_index(drop=True)
    idx = pd.Index(fm.permno.values); wv = fm.set_index("permno").w
    parts = []
    lists = [("PHYS", "P:"), ("ONL", "O:"), ("SERV", "S:")] + ([("SAME", "C:")] if same else [])
    for L, pre in lists:
        parts.append(con.execute(f"""SELECT permno, '{pre}' || b AS b, sum(act_num) a, sum(exp_num) e
                                      FROM read_parquet({fl('firm_rel_' + L)}) GROUP BY permno, b""").df())
    rel = pd.concat(parts, ignore_index=True); rel = rel[rel.permno.isin(idx)]
    A = rel.a / rel.permno.map(wv); E = rel.e / rel.permno.map(wv)
    rel["z"] = np.log((A + 1) / (E + 1)) if lift else (A - E) / np.sqrt((E + kappa) * (100 - E).clip(lower=1))
    Z, _ = S6.pivot(rel, "permno", "b", "z", idx)
    return idx, S6.cos_matrix(Z)


def stack(t, idx, M, name):
    n = len(idx); ii, jj = np.where(~np.eye(n, dtype=bool))
    return pd.DataFrame({"t": t, "i": idx.values[ii], "j": idx.values[jj], name: M[ii, jj].astype("float32")})


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    allm = S6.months_all()
    con = duckdb.connect(); con.execute("SET threads=4; SET memory_limit='3GB'")
    frozen = None
    for t in [m for m in allm if os.path.exists(f"{S6.OUT}/{m}.parquet")]:
        if os.path.exists(f"{OUT}/{t}.parquet"):
            continue
        k = allm.index(t)
        res = None
        specs = {"W6": dict(win=allm[max(0, k - 6):k]), "W24": dict(win=allm[max(0, k - 24):k]),
                 "K025": dict(win=allm[max(0, k - 12):k], kappa=0.25), "K5": dict(win=allm[max(0, k - 12):k], kappa=5.0),
                 "LIFT": dict(win=allm[max(0, k - 12):k], lift=True),
                 "WSAME": dict(win=allm[max(0, k - 12):k], same=True)}
        for nm, sp in specs.items():
            win = sp.pop("win")
            if len(win) < 6 and nm != "W6":
                continue
            idx, M = profiles(win, con, **sp)
            d = stack(t, idx, M, f"RPS_{nm}")
            res = d if res is None else res.merge(d, on=["t", "i", "j"], how="outer")
        if t >= "2020-01":
            if frozen is None:
                fidx, FM = profiles([m for m in allm if m.startswith("2019")], con)
                frozen = stack(None, fidx, FM, "RPS_FROZEN").drop(columns="t")
            res = res.merge(frozen, on=["i", "j"], how="left")
        res.to_parquet(f"{OUT}/{t}.parquet", index=False)
        print(t, res.shape, flush=True)
