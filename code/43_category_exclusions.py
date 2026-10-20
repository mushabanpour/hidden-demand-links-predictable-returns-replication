"""
43_category_exclusions.py -- the category adjustment and the placebo networks, each combined with the
exclusion of the exceptional months (January 2021; the return months of 2020; both).

Panel A: six-factor alphas of quintile sorts on the industry-adjusted peer return (Table 6, Panel B).
Panels B and C: placebo networks that reshuffle firms within their month-t SafeGraph category (as in
Figure 4), for the raw peer return (B) and for the industry-adjusted peer return (C), which is
re-estimated for every placebo network. Each placebo is evaluated on the same subsample of months as
the actual network; p is the share of placebo alphas at least as large as the actual one.
Output: tables/tab_category_excl.tex, logs/category_excl.json
"""
import os
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")
R = ap.R
NREP, SEED = 500, 7
JAN21 = pd.Period("2020-12", "M")


def samples(months):
    m = pd.PeriodIndex(list(months), freq="M")
    r2020 = (m + 1).year == 2020                      # return months in 2020
    return {"all": np.ones(len(m), bool), "ex_jan21": m != JAN21, "ex_2020": ~r2020, "ex_both": (~r2020) & (m != JAN21)}


def main():
    P = M.load("co")
    pr = pd.read_parquet(f"{R}/data/derived/panel_final.parquet", columns=["permno", "month", "ret"])   # peer returns as in the signal
    Fn = M.Fn
    # ------------------------------------------------------------- Panel A: industry-adjusted sorts
    ew, vw, _ = ap.sort_returns(P, "sig_adj", 5)
    S = samples(ew.index)
    A = {k: {w: (lambda a: (a["alpha"], a["t_alpha"]))(ap.ts_reg(x.LS[S[k]], Fn, "FF6")) for w, x in (("ew", ew), ("vw", vw))}
         for k in S}
    # ------------------------------------------------------------- Panels B, C: placebo networks
    rng = np.random.default_rng(SEED)
    mats = []
    for t, d in P.groupby("month"):
        U = pd.Index(sorted(d.permno.unique()))
        O = pd.read_parquet(f"{R}/data/derived/ovl/{t}.parquet", columns=["i", "j", "A_S"])
        O = O[O.i.isin(U) & O.j.isin(U)]
        W = np.zeros((len(U), len(U))); W[U.get_indexer(O.i), U.get_indexer(O.j)] = O.A_S.clip(lower=0).fillna(0).values
        r = pr[pr.month == t].set_index("permno").ret.reindex(U).values
        x = d.set_index("permno").reindex(U)
        Z = x[["ind_u", "indret", "cat_ew", "cat_vw"]]
        Z = np.column_stack([np.ones(len(U)), Z.fillna(Z.mean()).values])
        cat = x.cat.fillna("NA").values
        mats.append((t, W, np.nan_to_num(r), x.ret_f1.values, x.me.values, Z, [np.where(cat == c)[0] for c in np.unique(cat)]))

    def ls_series(perm):
        out = {k: {} for k in ("raw_ew", "raw_vw", "adj_ew", "adj_vw")}
        for t, W, r, y, me, Z, groups in mats:
            p = np.arange(len(r))
            if perm:
                for g in groups:
                    p[g] = rng.permutation(g)
            Wp = W[np.ix_(p, p)]
            den = Wp.sum(1)
            s = np.where(den > 0, Wp @ r / np.where(den > 0, den, 1), np.nan)
            has = ~np.isnan(s)                              # residualize over all firms with a peer return, as in the paper
            ok = has & ~np.isnan(y)
            if ok.sum() < 25:
                continue
            lo_, hi_ = np.nanquantile(s[has], [0.01, 0.99])  # winsorize as the paper's signal (13_main_tables.load)
            s = np.where(has, np.clip(s, lo_, hi_), np.nan)
            b = np.linalg.lstsq(Z[has], s[has], rcond=None)[0]
            for kind, sig in (("raw", s[ok]), ("adj", s[ok] - Z[ok] @ b)):
                q = pd.qcut(pd.Series(sig).rank(method="first"), 5, labels=False).values
                yy, ww = y[ok], me[ok]
                hi, lo = q == 4, q == 0
                out[f"{kind}_ew"][t] = yy[hi].mean() - yy[lo].mean()
                out[f"{kind}_vw"][t] = (yy[hi] * ww[hi]).sum() / ww[hi].sum() - (yy[lo] * ww[lo]).sum() / ww[lo].sum()
        return {k: pd.Series(list(v.values()), index=pd.PeriodIndex(list(v.keys()), freq="M")) for k, v in out.items()}

    def alphas(series):
        res = {}
        for k, x in series.items():
            S_ = samples(x.index)
            res[k] = {s: ap.ts_reg(x[S_[s]], Fn, "FF6")["alpha"] for s in S_}
        return res

    actual = alphas(ls_series(False))
    null = [alphas(ls_series(True)) for _ in range(NREP)]
    PB = {k: {s: {"actual": actual[k][s], "placebo_mean": float(np.mean([n[k][s] for n in null])),
                  "p": float(np.mean([n[k][s] >= actual[k][s] for n in null]))} for s in actual[k]} for k in actual}
    json.dump({"adjusted_sorts": A, "placebo": PB, "nrep": NREP, "seed": SEED}, open(f"{R}/logs/category_excl.json", "w"),
              default=float, indent=1)
    # ------------------------------------------------------------- table
    cols = ["all", "ex_jan21", "ex_2020", "ex_both"]
    hdr = [["", "All months", "Excl.\\ Jan.\\ 2021", "Excl.\\ 2020", "Excl.\\ both"]]
    rows = ["\\multicolumn{5}{l}{\\emph{Panel A: Sorts on the industry-adjusted peer return, six-factor alpha (\\%)}} \\\\"]
    for w, lab in (("ew", "Equal-weighted"), ("vw", "Value-weighted")):
        rows.append([lab] + [f"{100 * A[c][w][0]:.2f}{ap.stars(A[c][w][1])}" for c in cols])
        rows.append([""] + [f"({A[c][w][1]:.2f})" for c in cols])
    for kind, panel, title in (("raw", "B", "peer return"), ("adj", "C", "industry-adjusted peer return")):
        rows.append("\\addlinespace")
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{Panel {panel}: Placebo networks, {title}}}}} \\\\")
        for w, lab in (("ew", "Equal-weighted"), ("vw", "Value-weighted")):
            k = f"{kind}_{w}"
            rows.append([f"{lab}: actual alpha (\\%)"] + [f"{100 * PB[k][c]['actual']:.2f}" for c in cols])
            rows.append([f"\\quad placebo mean (\\%)"] + [f"{100 * PB[k][c]['placebo_mean']:.2f}" for c in cols])
            rows.append([f"\\quad $p$-value"] + [f"{PB[k][c]['p']:.3f}" for c in cols])
    months = {c: int(samples(ew.index)[c].sum()) for c in cols}
    rows.append("\\addlinespace")
    rows.append(["Months"] + [str(months[c]) for c in cols])
    notes = ("Panel A: six-factor alphas of the quintile long-short portfolio sorted on the industry-adjusted peer return "
             "(equation~\\eqref{eq:adj}; Table~\\ref{tab:alphas}, Panel B). Panels B and C: each month, firms' rows and columns of the "
             "customer-overlap matrix are reassigned at random among firms in the same SafeGraph category (Figure~\\ref{fig:perm}); "
             f"the peer return (Panel B) or the industry-adjusted peer return, re-estimated for each placebo network (Panel C), is "
             f"recomputed and the quintile long-short portfolio re-formed. Across {NREP} placebo networks, the table reports the mean "
             "six-factor alpha and the share of placebo alphas at least as large as the actual alpha ($p$). Excl.\\ 2020 drops the "
             "twelve return months of 2020; excl.\\ both also drops January 2021. Newey-West $t$-statistics (four lags) in parentheses.")
    open(f"{R}/tables/tab_category_excl.tex", "w").write(
        ap.latex_table(hdr, rows, "Category adjustment, exceptional months, and placebo networks", "tab:catexcl", notes,
                       colspec="lcccc"))
    print(json.dumps({"adjusted_sorts": A, "placebo": PB}, default=float, indent=1))


if __name__ == "__main__":
    main()
