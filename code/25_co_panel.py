"""
25_co_panel.py -- final analysis panel built around CUSTOMER OVERLAP (CO).

CO_ijt = (A_ij + A_ji)/2, where A_ij is the share of firm i's cardholders who also bought
from firm j's brands over months t-12..t-1 (same-category cross-shopping lists; brand
shares combined with the probabilistic union). Decomposition: A = E + X with E the share
predicted by other same-category shoppers in i's counties (market-level patronage) and X
the firm-specific excess.

Signals (month t): CO-weighted peer return and its splits by link visibility
(same/different FF48 industry, TNIC-3 rival or not, shared analyst or not), by peer size,
and by link strength; plus the RPS (shared tastes) and geography networks for horse races.
Output: data/derived/panel_final.parquet
"""
import os
import numpy as np, pandas as pd

R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def wavg(S, w):
    w = w.fillna(0.0)
    num = (w * S.rj).groupby(S.i).sum(); den = w.groupby(S.i).sum()
    return num / den.where(den > 0), den


def main():
    P = pd.read_parquet(f"{R}/data/derived/panel_ovl2.parquet")
    cc = pd.read_parquet(f"{R}/data/derived/cocoverage_pairs.parquet")
    rows = []
    for t, d in P.groupby("month"):
        U = d.set_index("permno")
        O = pd.read_parquet(f"{R}/data/derived/ovl/{t}.parquet", columns=["i", "j", "A_S", "E_S", "X_S", "A_P"])
        S = pd.read_parquet(f"{R}/data/derived/sim/{t}.parquet", columns=["i", "j", "RPS", "STORE", "HOME"])
        S = O.merge(S, on=["i", "j"], how="left")
        S = S[S.i.isin(U.index) & S.j.isin(U.index) & (S.A_S > 0)].copy()
        S["rj"] = S.j.map(U.ret)
        S["same_ind"] = S.i.map(U.ff48).values == S.j.map(U.ff48).values
        yr = t.year if t.month >= 7 else t.year - 1
        tp = pd.read_parquet(f"{R}/data/derived/tnic_pairs_{min(yr, 2025)}.parquet"); tk = set(zip(tp.permno, tp.j))
        S["tnic"] = [(a, b) in tk for a, b in zip(S.i, S.j)]
        ck = cc[cc.yr == t.year]; ck = set(zip(ck.permno, ck.permno_j))
        S["cov"] = [(a, b) in ck for a, b in zip(S.i, S.j)]
        S["big_j"] = S.j.map(U.me) >= U.me.median()
        S["k"] = S.groupby("i").A_S.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) if len(s) >= 3 else 2)
        o = {}
        o["co"], o["co_deg"] = wavg(S, S.A_S)
        o["co_n"] = S.groupby("i").size()
        o["co_E"], _ = wavg(S, S.E_S.clip(lower=0))
        o["co_X"], _ = wavg(S, S.X_S.clip(lower=0))
        o["co_eq"], _ = wavg(S, pd.Series(1.0, index=S.index))
        for nm, m in [("same", S.same_ind), ("cross", ~S.same_ind), ("tnic", S.tnic), ("ntnic", ~S.tnic),
                      ("cov", S["cov"]), ("ncov", ~S["cov"]), ("big", S.big_j), ("small", ~S.big_j),
                      ("strong", S.k == 2), ("medium", S.k == 1), ("weak", S.k == 0)]:
            o[f"co_{nm}"], _ = wavg(S, S.A_S * m)
        for nm, c in [("tnic", "tnic"), ("cov", "cov"), ("same", "same_ind")]:
            o[f"cow_{nm}"] = (S.A_S * S[c]).groupby(S.i).sum() / S.A_S.groupby(S.i).sum()
        o = pd.DataFrame(o); o["month"] = t
        rows.append(o.reset_index().rename(columns={"i": "permno"}))
    C = pd.concat(rows)
    P = P.drop(columns=[c for c in C.columns if c in P.columns and c not in ("permno", "month")]).merge(C, on=["permno", "month"], how="left")
    P.to_parquet(f"{R}/data/derived/panel_final.parquet", index=False)
    print(P.shape, P.co.notna().mean(), P.groupby("month").co.count().mean(), P.co_n.median())


if __name__ == "__main__":
    main()
