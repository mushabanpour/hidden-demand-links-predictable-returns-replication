"""
16_attention.py -- H5: limited attention.

 tab_subsamples   Fama-MacBeth slopes on the peer return in
                  subsamples split at the monthly median of size, analyst coverage,
                  institutional ownership, and industry return
 tab_visibility   peer return decomposed into visible vs hidden links: same vs other
                  FF48 industry; TNIC-3 rival vs not; shared analyst vs not
 tab_bigsmall     peer return split into large-cap and
                  small-cap peers, for all, large, and small focal firms
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


def add_io(P):
    io = pd.read_parquet(f"{R}/data/raw/inst_own_13f.parquet")
    io["month"] = pd.to_datetime(io.fdate).dt.to_period("M")
    io = io.groupby(["permno", "month"]).inst_shares.sum().reset_index()
    msf = pd.read_parquet(f"{R}/data/raw/crsp_msf_v2.parquet", columns=["permno", "mthcaldt", "shrout"])
    msf["month"] = pd.to_datetime(msf.mthcaldt).dt.to_period("M")
    io = io.merge(msf[["permno", "month", "shrout"]], on=["permno", "month"], how="left")
    io["io"] = (io.inst_shares / (io.shrout * 1e3)).clip(0, 1)
    io["avail"] = io.month + 2                       # 13F filed within 45 days of quarter end
    P = P.sort_values("month")
    io = io.dropna(subset=["io"]).sort_values("avail")
    P["_t"] = P.month.dt.to_timestamp(); io["_t"] = io.avail.dt.to_timestamp()
    P = pd.merge_asof(P, io[["permno", "_t", "io"]], on="_t", by="permno", direction="backward",
                      tolerance=pd.Timedelta(days=200)).drop(columns="_t")
    return P


def fm_row(D, xs, v="sig"):
    r, r2, n, T = ap.fama_macbeth(D, "ret_f1", xs, min_obs=20)
    return r[v], n, r


def add_turnover(P):
    msf = pd.read_parquet(f"{R}/data/raw/crsp_msf_v2.parquet", columns=["permno", "mthcaldt", "mthvol", "shrout"])
    msf["month"] = pd.to_datetime(msf.mthcaldt).dt.to_period("M")
    msf = msf.sort_values(["permno", "month"]).drop_duplicates(["permno", "month"])
    msf["to"] = msf.mthvol / (msf.shrout * 1e3)
    msf["turn"] = msf.groupby("permno").to.transform(lambda s: s.rolling(12, min_periods=6).mean())
    return P.merge(msf[["permno", "month", "turn"]], on=["permno", "month"], how="left")


def subsamples(P):
    P = add_turnover(add_io(P))
    splits = [("size", "Market cap", "Small", "Large"), ("numest", "Analyst coverage", "Low", "High"),
              ("io", "Inst.\\ ownership", "Low", "High"), ("turn", "Share turnover", "Low", "High"),
              ("indret", "Industry return", "Low", "High")]
    xs = ["sig"] + CTL
    rows, hdr2 = [], [""]
    res = {}
    for var, lab, lo, hi in splits:
        med = P.groupby("month")[var].transform("median")
        lo_m = P[var] < med if var != "numest" else P[var] <= med
        for nm, msk in [(lo, lo_m), (hi, ~lo_m & P[var].notna())]:
            (b, t), n, r = fm_row(P[msk], xs)
            res[(lab, nm)] = (b, t, n, r)
    # difference (low - high) via FM on interaction
    cells_b, cells_t, cells_n = ["Customer-overlap peer return"], [""], ["Observations"]
    for var, lab, lo, hi in splits:
        for nm in (lo, hi):
            b, t, n, _ = res[(lab, nm)]
            cells_b.append(f"{b:.3f}{ap.stars(t)}"); cells_t.append(f"({t:.2f})"); cells_n.append(f"{n:,}")
    diff_b, diff_t = ["Low $-$ High"], [""]
    for var, lab, lo, hi in splits:
        med = P.groupby("month")[var].transform("median")
        D = P.copy(); D["low"] = ((D[var] <= med) if var == "numest" else (D[var] < med)).astype(float)
        D.loc[D[var].isna(), "low"] = np.nan
        D["sig_low"] = D.sig * D.low
        r, *_ = ap.fama_macbeth(D, "ret_f1", ["sig", "sig_low", "low"] + CTL, min_obs=30)
        b, t = r["sig_low"]
        diff_b += [f"\\multicolumn{{2}}{{c}}{{{b:.3f}{ap.stars(t)}}}"]; diff_t += [f"\\multicolumn{{2}}{{c}}{{({t:.2f})}}"]
    hdr = [["", *[f"\\multicolumn{{2}}{{c}}{{{s[1]}}}" for s in splits]],
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}",
           ["", *sum([[s[2], s[3]] for s in splits], [])]]
    rows = [cells_b, cells_t, "\\addlinespace", diff_b, diff_t, "\\addlinespace", ["Controls"] + ["Yes"] * 10, cells_n]
    notes = ("Fama-MacBeth regressions of month-$t+1$ returns on the month-$t$ customer-overlap peer return and the controls "
             "of Table~\\ref{tab:fm}, column (5), estimated separately for firms below and above the monthly median of market "
             "capitalization, the number of IBES analysts (firms at the median are in the low group), the 13F institutional "
             "ownership ratio, share turnover (average monthly volume divided by shares outstanding over months $t-11$ to $t$), "
             "and the FF48 industry return. "
             "``Low $-$ High'' is the coefficient on the interaction of the peer return with a below-median indicator in a pooled "
             "Fama-MacBeth regression. Newey-West $t$-statistics (four lags) in parentheses.")
    open(f"{R}/tables/tab_subsamples.tex", "w").write(
        ap.latex_table(hdr, rows, "Who underreacts? Subsamples by attention proxies", "tab:subsamples", notes, size="\\scriptsize"))
    return res


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


def bigsmall(P):
    D = P[P.sig.notna()].copy()                           # firms with at least one customer-overlap peer
    D["v_big"] = D[SPL("big")]; D["v_small"] = D[SPL("small")]
    for c in ["v_big", "v_small"]:                        # empty subset -> monthly cross-sectional mean
        D[c] = D.groupby("month")[c].transform(ap.winsor)
        D[c] = D[c].fillna(D.groupby("month")[c].transform("mean"))
    med = D.groupby("month")["size"].transform("median")
    out = []
    for nm, msk in [("All firms", D["size"].notna()), ("Large firms", D["size"] >= med), ("Small firms", D["size"] < med)]:
        for xs in [["v_big", "v_small"], ["v_big", "v_small"] + CTL]:
            r, r2, n, T = ap.fama_macbeth(D[msk], "ret_f1", xs, min_obs=20)
            C = ap.fama_macbeth(D[msk], "ret_f1", xs, min_obs=20, series=True)
            out.append((nm, xs, r, n, ap.nw_mean(C.v_small - C.v_big)[:2]))
    hdr = [["", "\\multicolumn{2}{c}{All firms}", "\\multicolumn{2}{c}{Large firms}", "\\multicolumn{2}{c}{Small firms}"],
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}", ["", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)"]]
    rows = []
    for key, lab in [("v_big", "Large-cap peers' return"), ("v_small", "Small-cap peers' return")]:
        rb, rt = [lab], [""]
        for nm, xs, r, n, d in out:
            b, t = r[key]; rb.append(f"{b:.3f}{ap.stars(t)}"); rt.append(f"({t:.2f})")
        rows += [rb, rt]
    rows.append("\\addlinespace")
    rows.append(["Small $-$ large"] + [f"{o[4][0]:.3f}{ap.stars(o[4][1])}" for o in out])
    rows.append([""] + [f"({o[4][1]:.2f})" for o in out])
    rows.append("\\addlinespace")
    rows.append(["Controls"] + ["", "Yes"] * 3)
    rows.append(["Observations"] + [f"{o[3]:,}" for o in out])
    notes = ("Each firm's customer-overlap peers are split at the month-$t$ median market capitalization of sample firms; "
             "each sub-portfolio is weighted by shared customers. Fama-MacBeth regressions of month-$t+1$ returns for all firms and "
             "for firms above and below the median size, without controls (columns (1), (3), and (5)) and with the controls of "
             "Table~\\ref{tab:fm}, column (5) (columns (2), (4), and (6)). The sample is firms with at least one customer-overlap "
             "peer; when one sub-portfolio is empty, its return is set to the monthly cross-sectional mean. ``Small $-$ large'' is the "
             "time-series average of the monthly difference between the two slopes. Newey-West $t$-statistics (four lags) in "
             "parentheses.")
    open(f"{R}/tables/tab_bigsmall.tex", "w").write(
        ap.latex_table(hdr, rows, "Large and small peers as sources of information", "tab:bigsmall", notes))
    return out


if __name__ == "__main__":
    P = M.load(SIG)
    s = subsamples(P)
    for k, v in s.items(): print(k, round(v[0], 3), round(v[1], 2), v[2])
    for la, lb, r, n, d in visibility(P): print(la, r["v_a"], lb, r["v_b"], "hidden-visible", d)
    for nm, xs, r, n, d in bigsmall(P): print(nm, len(xs), r["v_big"], r["v_small"], "small-large", d)
