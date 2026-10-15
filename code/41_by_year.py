"""
41_by_year.py -- predictability by calendar year of the return month: average monthly
Fama-MacBeth slope on the customer-overlap peer return (controls of Table fm, column 5) and
the equal- and value-weighted quintile long-short returns; plus the full sample and the
sample excluding January 2021 and/or return months in 2020.
Output: tables/tab_byyear.tex, logs/byyear.json
"""
import os
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")
R = ap.R
P = M.load("co")
C = ap.fama_macbeth(P, "ret_f1", ["sig"] + M.CONTROLS, series=True)["sig"]
ew, vw, n = ap.sort_returns(P, "sig", 5)
D = pd.DataFrame({"fm": C, "ew": ew.LS * 100, "vw": vw.LS * 100}).dropna()
D["ry"] = (D.index + 1).year                         # calendar year of the return month t+1
jan21 = pd.Period("2020-12", "M")                     # formation month of the January 2021 return
out, rows = {}, []
for y, d in D.groupby("ry"):
    lab = f"{y}" + (" (Aug--Dec)" if y == 2019 else "")
    out[str(y)] = {"months": len(d), **{c: d[c].mean() for c in ["fm", "ew", "vw"]}}
    rows.append([lab, str(len(d)), f"{d.fm.mean():.3f}", f"{d.ew.mean():.2f}", f"{d.vw.mean():.2f}"])
    if y == 2021:
        e = d.drop(jan21)
        out["2021_exjan"] = {"months": len(e), **{c: e[c].mean() for c in ["fm", "ew", "vw"]}}
        rows.append(["2021, excluding January", str(len(e)), f"{e.fm.mean():.3f}", f"{e.ew.mean():.2f}", f"{e.vw.mean():.2f}"])
rows.append("\\addlinespace")
for lab, key, d in [("All months", "all", D), ("Excluding January 2021", "ex_jan21", D.drop(jan21)),
                    ("Excluding return months in 2020", "ex_2020", D[D.ry != 2020]),
                    ("Excluding 2020 and January 2021", "ex_2020_jan21", D[(D.ry != 2020)].drop(jan21))]:
    st = {c: ap.nw_mean(d[c])[:2] for c in ["fm", "ew", "vw"]}
    out[key] = {"months": len(d), **st}
    rows.append([lab, str(len(d))] + [f"{st[c][0]:.{3 if c == 'fm' else 2}f}{ap.stars(st[c][1])}" for c in ["fm", "ew", "vw"]])
    rows.append(["", ""] + [f"({st[c][1]:.2f})" for c in ["fm", "ew", "vw"]])
out["jan21"] = {c: D.loc[jan21, c] for c in ["fm", "ew", "vw"]}
out["share_of_fm_from_jan21"] = D.loc[jan21, "fm"] / D.fm.sum()
out["share_of_fm_from_2020"] = D.loc[D.ry == 2020, "fm"].sum() / D.fm.sum()
json.dump(out, open(f"{R}/logs/byyear.json", "w"), default=float, indent=1)
notes = ("Averages of monthly quantities by calendar year of the return month $t+1$. Fama-MacBeth slope: the monthly "
         "cross-sectional coefficient on the customer-overlap peer return with the controls of Table~\\ref{tab:fm}, column (5). "
         "EW and VW long-short: the return of the top minus the bottom quintile of Table~\\ref{tab:sorts}, in percent per month. "
         "The bottom panel reports full-sample averages and averages excluding January 2021 (formation month December 2020) "
         "and/or the twelve return months of 2020, with Newey-West $t$-statistics (four lags) in parentheses.")
tex = ap.latex_table([["Return year", "Months", "Fama-MacBeth slope", "EW long-short (\\%)", "VW long-short (\\%)"]], rows,
                     "Return predictability by year", "tab:byyear", notes, colspec="lcccc")
open(f"{R}/tables/tab_byyear.tex", "w").write(tex)
print(json.dumps(out, default=float, indent=1))
