"""29_decomp_table.py -- Table: demand versus geography; shared customers versus shared tastes."""
import os
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")
R = ap.R
P = M.load("co")
P = P[P.sig.notna()].copy()               # firms with at least one customer-overlap peer
for c in ["co_E", "co_X", "co_strong", "co_medium", "co_weak", "sc_store", "sc_home", "RPS_D_pos", "RPSX_pos"]:
    P[c] = P.groupby("month")[c].transform(lambda s: ap.winsor(s.replace([np.inf, -np.inf], np.nan)))
for c in ["co_strong", "co_medium", "co_weak", "sc_store", "sc_home", "co_E", "co_X"]:
    P[c] = P[c].fillna(P.groupby("month")[c].transform("mean"))
CTL = M.CONTROLS
specs = [["sig"], ["co_E", "co_X"], ["sig", "sc_store", "sc_home"], ["co_strong", "co_medium", "co_weak"],
         ["sig", "RPS_pos", "HOME_pos"], ["RPS_pos"], ["RPS_D_pos"], ["RPSX_pos"]]
lab = {"sig": "Customer overlap (CO)", "co_E": "CO: local-market component", "co_X": "CO: firm-specific (excess) component",
       "sc_store": "Same-category store-footprint peers", "sc_home": "Same-category customer-home peers",
       "co_strong": "CO: strongest third of links", "co_medium": "CO: middle third", "co_weak": "CO: weakest third",
       "RPS_pos": "Shared tastes (RPS)", "HOME_pos": "Customer homes (all firms)", "RPS_D_pos": "Shared tastes, location-free channels",
       "RPSX_pos": "Shared tastes, orthogonal to income and geography"}
P = P.dropna(subset=["ret_f1", "sig", "RPS_pos", "HOME_pos", "RPS_D_pos", "RPSX_pos"] + CTL)   # one sample for every column
res = [ap.fama_macbeth(P, "ret_f1", s + CTL) for s in specs]
order = ["sig", "co_E", "co_X", "sc_store", "sc_home", "co_strong", "co_medium", "co_weak", "RPS_pos", "HOME_pos", "RPS_D_pos", "RPSX_pos"]
rows = []
for v in order:
    rb, rt = [lab[v]], [""]
    for (r, *_), s in zip(res, specs):
        if v in s:
            b, t = r[v]; rb.append(f"{b:.3f}{ap.stars(t)}"); rt.append(f"({t:.2f})")
        else:
            rb.append(""); rt.append("")
    rows += [rb, rt]
rows.append("\\addlinespace")
rows.append(["Controls"] + ["Yes"] * len(specs))
rows.append(["Observations"] + [f"{x[2]:,}" for x in res])
notes = ("Fama-MacBeth regressions of month-$t+1$ returns on month-$t$ peer returns built from alternative networks, with the "
         "controls of Table~\\ref{tab:fm}, column (5). The local-market component weights same-category peers by the share of "
         "the focal firm's customers that would also shop at the peer if they behaved like other same-category shoppers in the "
         "focal firm's local markets (counties); the firm-specific component weights by the excess of actual over predicted overlap (positive "
         "part). Same-category store-footprint (customer-home) peers weight same-category firms by the cosine similarity of "
         "store counts across counties (customer counts across home cities). Link-strength terciles split each firm's "
         "customer-overlap peers by the size of the overlap. Shared-taste returns weight all sample firms by RPS. "
         "All columns use the same firm-months: firms with at least one customer-overlap peer and all controls. When a "
         "firm has no peer in a component or link-strength third, that peer return is set to its monthly cross-sectional mean. "
         "Newey-West $t$-statistics (four lags) in parentheses.")
open(f"{R}/tables/tab_decomp.tex", "w").write(ap.latex_table([["", *[f"({k})" for k in range(1, len(specs) + 1)]]], rows,
                                                            "Shared customers, shared places, and shared tastes", "tab:decomp",
                                                            notes, size="\\scriptsize"))
ex = [ap.fama_macbeth(P[P.month.dt.year != 2020], "ret_f1", s + CTL) for s in specs]          # formation months in 2020
exr = [ap.fama_macbeth(P[(P.month + 1).dt.year != 2020], "ret_f1", s + CTL) for s in specs]   # return months in 2020
json.dump({"specs": specs, "full": [{k: r[k] for k in s} for (r, *_), s in zip(res, specs)],
           "ex2020": [{k: r[k] for k in s} for (r, *_), s in zip(ex, specs)],
           "ex2020_return": [{k: r[k] for k in s} for (r, *_), s in zip(exr, specs)], "nobs": [x[2] for x in res]},
          open(f"{R}/logs/decomp.json", "w"), default=float, indent=1)
for (r, *_), (r2, *_), s in zip(res, ex, specs):
    print(s, {k: (round(r[k][0], 3), round(r[k][1], 2)) for k in s}, "ex2020", {k: (round(r2[k][0], 3), round(r2[k][1], 2)) for k in s})
