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
P = P.merge(pd.read_parquet(f"{R}/data/derived/geo_category.parquet", columns=["permno", "month", "geo_all"]),
            on=["permno", "month"], how="left")       # 54_geo_category.py
P = P[P.sig.notna()].copy()               # firms with at least one customer-overlap peer
for c in ["co_E", "co_X", "co_strong", "co_medium", "co_weak", "sc_store", "sc_home", "geo_all"]:
    P[c] = P.groupby("month")[c].transform(lambda s: ap.winsor(s.replace([np.inf, -np.inf], np.nan)))
for c in ["co_strong", "co_medium", "co_weak", "sc_store", "sc_home", "co_E", "co_X", "geo_all"]:
    P[c] = P[c].fillna(P.groupby("month")[c].transform("mean"))
CTL = M.CONTROLS
specs = [["sig"], ["co_E", "co_X"], ["co_E", "co_X", "geo_all"], ["sig", "sc_store", "sc_home"], ["co_strong", "co_medium", "co_weak"],
         ["sig", "RPS_pos", "HOME_pos"], ["RPS_pos"]]
lab = {"sig": "Customer overlap (CO)", "co_E": "CO: local-market component", "co_X": "CO: firm-specific (excess) component",
       "geo_all": "Geography-weighted category return",
       "sc_store": "Same-category store-footprint peers", "sc_home": "Same-category customer-home peers",
       "co_strong": "CO: strongest third of links", "co_medium": "CO: middle third", "co_weak": "CO: weakest third",
       "RPS_pos": "Shared tastes (RPS)", "HOME_pos": "Customer homes (all firms)"}
P = P.dropna(subset=["ret_f1", "sig", "RPS_pos", "HOME_pos"] + CTL)   # one sample for every column
res = [ap.fama_macbeth(P, "ret_f1", s + CTL) for s in specs]
order = ["sig", "co_E", "co_X", "geo_all", "sc_store", "sc_home", "co_strong", "co_medium", "co_weak", "RPS_pos", "HOME_pos"]
rows = []
for v in order:
    rb, rt = [lab[v]], [""]
    for (r, *_), s in zip(res, specs):
        if v in s:
            b, t = r[v]; rb.append(f"{b:.3f}{ap.stars(t)}"); rt.append(f"({t:.2f})")
        else:
            rb.append(""); rt.append("")
    rows += [rb, rt]
# direct tests of the comparisons in columns (2) and (4): monthly slope differences
C2 = ap.fama_macbeth(P, "ret_f1", specs[1] + CTL, series=True)
C4 = ap.fama_macbeth(P, "ret_f1", specs[4] + CTL, series=True)
diffs = {"local_minus_firm_specific": ap.nw_mean(C2.co_E - C2.co_X)[:2],
         "strongest_minus_weakest": ap.nw_mean(C4.co_strong - C4.co_weak)[:2]}
rows.append("\\addlinespace")
for k, lab_, col in [("local_minus_firm_specific", "Local-market $-$ firm-specific", 2),
                     ("strongest_minus_weakest", "Strongest $-$ weakest third", 5)]:
    m, t = diffs[k]
    rb = [lab_] + [""] * len(specs); rt = [""] * (len(specs) + 1)
    rb[col] = f"{m:.3f}{ap.stars(t)}"; rt[col] = f"({t:.2f})"
    rows += [rb, rt]
rows.append("\\addlinespace")
rows.append(["Controls"] + ["Yes"] * len(specs))
rows.append(["Observations"] + [f"{x[2]:,}" for x in res])
notes = ("Fama-MacBeth regressions of month-$t+1$ returns on month-$t$ peer returns built from alternative networks, with the "
         "controls of Table~\\ref{tab:fm}, column (5). The local-market component weights same-category peers by the share of "
         "the focal firm's customers that would also shop at the peer if they behaved like other same-category shoppers in the "
         "focal firm's local markets (counties); the firm-specific component weights by the excess of actual over predicted overlap (positive "
         "part). The geography-weighted category return weights each same-category peer by its share of the category's customers "
         "in each county, averaged over the focal firm's counties with the focal firm's own county customer shares; it uses only "
         "where firms' customers shop, not who shops at both. Same-category store-footprint (customer-home) peers weight same-category firms by the cosine similarity of "
         "store counts across counties (customer counts across home cities). Link-strength terciles split each firm's "
         "customer-overlap peers by the size of the overlap. Shared-taste returns weight all sample firms by RPS. "
         "All columns use the same firm-months: firms with at least one customer-overlap peer and all controls. When a "
         "firm has no peer in a component or link-strength third, that peer return is set to its monthly cross-sectional mean. "
         "The difference rows report the time-series average of the monthly difference between the two slopes in column (2) "
         "and between the strongest- and weakest-third slopes in column (5). Newey-West $t$-statistics (four lags) in parentheses.")
open(f"{R}/tables/tab_decomp.tex", "w").write(ap.latex_table([["", *[f"({k})" for k in range(1, len(specs) + 1)]]], rows,
                                                            "Shared customers, shared places, and shared tastes", "tab:decomp",
                                                            notes, size="\\scriptsize"))
json.dump({"specs": specs, "full": [{k: r[k] for k in s} for (r, *_), s in zip(res, specs)],
           "nobs": [x[2] for x in res],
           "differences": diffs},
          open(f"{R}/logs/decomp.json", "w"), default=float, indent=1)
for (r, *_), s in zip(res, specs):
    print(s, {k: (round(r[k][0], 3), round(r[k][1], 2)) for k in s})
