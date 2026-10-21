"""
16_attention.py -- H5: limited attention and the visibility of links.

 tab_visibility   peer return decomposed into visible vs hidden links: same vs other
                  FF48 industry; TNIC-3 rival vs not; shared analyst vs not
"""
import os
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")
R = ap.R
SIG = sys.argv[1] if len(sys.argv) > 1 else "co"
BASE = SIG.split("_pos")[0].split("_top")[0]
SPL = (lambda nm: f"co_{nm}") if SIG == "co" else (lambda nm: f"{BASE}_pos_{nm}")
CTL = ["ret", "ind_u", "indret", "cat_ew", "cat_vw", "STORE_pos", "tnic_ret", "size", "bm", "gpa", "ik", "mom"]


def visibility(P):
    pairs = [("same", "cross", "Same FF48 industry", "Different industry"),
             ("tnic", "ntnic", "TNIC-3 rival", "Not a TNIC-3 rival"),
             ("cov", "ncov", "Shared analyst", "No shared analyst")]
    cols = []
    for a, b, la, lb in pairs:
        D = P[P.sig.notna()].copy()                       # firms with at least one customer-overlap peer
        D["v_a"] = D[SPL(a)]; D["v_b"] = D[SPL(b)]
        for c in ["v_a", "v_b"]:                          # empty subset -> monthly cross-sectional mean
            D[c] = D.groupby("month")[c].transform(ap.winsor)
            D[c] = D[c].fillna(D.groupby("month")[c].transform("mean"))
        r, r2, n, T = ap.fama_macbeth(D, "ret_f1", ["v_a", "v_b"] + CTL + ["xind_u"])
        C = ap.fama_macbeth(D, "ret_f1", ["v_a", "v_b"] + CTL + ["xind_u"], series=True)
        d = ap.nw_mean(C.v_b - C.v_a)[:2]                  # hidden minus visible, monthly slopes
        cols.append((la, lb, r, n, d))
    hdr = [["", "(1)", "(2)", "(3)"]]
    rows = []
    for k, (la, lb, r, n, d) in enumerate(cols):
        for key, lab in [("v_a", la), ("v_b", lb)]:
            rb = [f"Peers: {lab}"] + [""] * 3; rt = [""] * 4
            b, t = r[key]; rb[k + 1] = f"{b:.3f}{ap.stars(t)}"; rt[k + 1] = f"({t:.2f})"
            rows += [rb, rt]
    rows.append("\\addlinespace")
    rows.append(["Hidden $-$ visible"] + [f"{c[4][0]:.3f}{ap.stars(c[4][1])}" for c in cols])
    rows.append([""] + [f"({c[4][1]:.2f})" for c in cols])
    rows.append("\\addlinespace")
    rows.append(["Controls", "Yes", "Yes", "Yes"])
    rows.append(["Observations"] + [f"{c[3]:,}" for c in cols])
    notes = ("Fama-MacBeth regressions of month-$t+1$ returns on the month-$t$ returns of two disjoint subsets of each firm's "
             "customer-overlap peers (each weighted by shared customers within the subset), the controls of "
             "Table~\\ref{tab:fm}, column (5), and the equal-weighted average return of sample firms outside the firm's FF48 "
             "industry. The sample is firms with at least one customer-overlap peer; when one subset is empty, its return is set "
             "to the monthly cross-sectional mean of that subset return. Column (1) splits peers by whether they share the focal firm's FF48 industry; "
             "column (2) by whether they are TNIC-3 product-market rivals; column (3) by whether at least one IBES analyst covered "
             "both firms in the previous calendar year. ``Hidden $-$ visible'' is the time-series average of the monthly difference "
             "between the slope on the hidden subset (different industry, not a TNIC-3 rival, no shared analyst) and the slope on "
             "the visible subset. Newey-West $t$-statistics (four lags) in parentheses.")
    open(f"{R}/tables/tab_visibility.tex", "w").write(
        ap.latex_table(hdr, rows, "Visible versus hidden links", "tab:visibility", notes))
    json.dump({la + " vs " + lb: {"visible": r["v_a"], "hidden": r["v_b"], "hidden_minus_visible": d, "n": n}
               for la, lb, r, n, d in cols}, open(f"{R}/logs/visibility.json", "w"), default=float, indent=1)
    return cols


if __name__ == "__main__":
    P = M.load(SIG)
    for la, lb, r, n, d in visibility(P): print(la, r["v_a"], lb, r["v_b"], "hidden-visible", d)
