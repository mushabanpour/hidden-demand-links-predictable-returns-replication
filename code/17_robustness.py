"""
17_robustness.py -- robustness of the main result + permutation placebo.

Each row: EW and VW quintile long-short FF6 alpha and the Fama-MacBeth slope (with the
column-(5) controls of Table 5) for one alternative specification.
"""
import sys, glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")
R = ap.R
SIG = sys.argv[1] if len(sys.argv) > 1 else "co"
CTL = ["ret", "ind_u", "indret", "cat_ew", "cat_vw", "STORE_pos", "tnic_ret", "size", "bm", "gpa", "ik", "mom"]
Fn = M.Fn


def evaluate(D, sig="sig", ret="ret_f1"):
    out = {}
    ew, vw, n = ap.sort_returns(D, sig, 5, ret=ret)
    for nm, x in [("ew", ew), ("vw", vw)]:
        a = ap.ts_reg(x.LS, Fn2 if ret == "ret_f2" else Fn, "FF6")
        out[nm] = (a["alpha"] * 100, a["t_alpha"])
    r, r2, nobs, T = ap.fama_macbeth(D, ret, [sig] + CTL)
    out["fm"] = r[sig]; out["T"] = T
    return out


Fn2 = ap.factors(); Fn2.index = Fn2.index - 2      # month t+2 factors for the skip-a-month test


def variant_signals(P, src="ovl_var", prefix="CO_"):
    """peer-return signals from alternative networks (ovl_var for customer overlap, sim_var for RPS)."""
    sm = pd.read_parquet(f"{R}/data/derived/stock_month.parquet", columns=["permno", "month", "ret"])
    rows = []
    for f in sorted(glob.glob(f"{R}/data/derived/{src}/*.parquet")):
        t = pd.Period(os.path.basename(f)[:7], "M")
        U = P.loc[P.month == t, "permno"]
        S = pd.read_parquet(f); S = S[S.i.isin(U) & S.j.isin(U)]
        r = sm[sm.month == t].set_index("permno").ret
        S["rj"] = S.j.map(r)
        o = pd.DataFrame(index=U.values)
        for c in [c for c in S.columns if c.startswith(prefix)]:
            x = S[["i", "rj", c]].dropna(); w = x[c].clip(lower=0)
            g = (w * x.rj).groupby(x.i).sum() / w.groupby(x.i).sum().replace(0, np.nan)
            o[c] = g
        o["month"] = t
        rows.append(o.reset_index().rename(columns={"index": "permno"}))
    V = pd.concat(rows)
    return P.merge(V, on=["permno", "month"], how="left")


def permutation(P, nrep=500, seed=7):
    rng = np.random.default_rng(seed)
    # peer returns exactly as in the baseline signal (monthly returns winsorized at 1%/99%, panel_final.ret)
    sm = pd.read_parquet(f"{R}/data/derived/panel_final.parquet", columns=["permno", "month", "ret"])
    mats = {}
    for t, d in P.groupby("month"):
        S = pd.read_parquet(f"{R}/data/derived/ovl/{t}.parquet", columns=["i", "j", "A_S"]).rename(columns={"A_S": "CO"})
        U = pd.Index(sorted(d.permno.unique()))
        S = S[S.i.isin(U) & S.j.isin(U)]
        W = np.zeros((len(U), len(U)))
        W[U.get_indexer(S.i), U.get_indexer(S.j)] = S.CO.clip(lower=0).fillna(0).values
        r = sm[sm.month == t].set_index("permno").ret.reindex(U).values
        y = d.set_index("permno").ret_f1.reindex(U).values
        cat = d.set_index("permno").cat.reindex(U).fillna("NA").values
        groups = [np.where(cat == c)[0] for c in np.unique(cat)]
        mats[t] = (W, np.nan_to_num(r), y, groups)
    def ls_alpha(perm):
        ls = {}
        for t, (W, r, y, groups) in mats.items():
            n = len(r); p = np.arange(n)
            if perm:                                   # shuffle firm labels within SafeGraph category
                for g in groups:
                    p[g] = rng.permutation(g)
            Wp = W[np.ix_(p, p)]
            s = Wp @ r / np.where(Wp.sum(1) > 0, Wp.sum(1), np.nan)
            ok = ~np.isnan(s) & ~np.isnan(y)
            if ok.sum() < 25:
                continue
            q = pd.qcut(pd.Series(s[ok]).rank(method="first"), 5, labels=False)
            yy = y[ok]
            ls[t] = yy[q.values == 4].mean() - yy[q.values == 0].mean()
        ls = pd.Series(ls)
        return ap.ts_reg(ls, Fn, "FF6")["alpha"]
    actual = ls_alpha(False)
    null = np.array([ls_alpha(True) for _ in range(nrep)])
    import figstyle as fs
    fs.apply()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.hist(null * 100, bins=40, color=fs.BAR_LIGHT, edgecolor="white", linewidth=0.4, label="Placebo networks")
    ax.axvline(actual * 100, color=fs.BLACK, lw=1.4, label=f"Actual network ({actual*100:.2f}%)")
    ax.set_xlim(right=actual * 100 + 0.2)
    ax.set_xlabel("Six-factor alpha of quintile long-short portfolio, percent per month"); ax.set_ylabel("Placebo networks")
    ax.legend(loc="upper left")
    plt.tight_layout(); plt.savefig(f"{R}/figures/fig_permutation.pdf"); plt.close()
    return actual, null, (null >= actual).mean()


def main():
    P = M.load(SIG)
    P = variant_signals(P, "ovl_var", "CO_")
    P = variant_signals(P, "sim_var", "RPS_")
    sm = pd.read_parquet(f"{R}/data/derived/stock_month.parquet", columns=["permno", "month", "ret", "me", "nyse"])
    f2 = sm[["permno", "month", "ret"]].copy(); f2["month"] = f2.month - 2
    P = P.merge(f2.rename(columns={"ret": "ret_f2"}), on=["permno", "month"], how="left")
    bp = sm[sm.nyse].groupby("month").me.quantile(0.2).rename("nyse20")
    P = P.merge(bp, left_on="month", right_index=True, how="left")
    nm_ = pd.read_parquet(f"{R}/data/raw/crsp_stocknames_v2.parquet").sort_values("nameenddt").groupby("permno").tail(1).set_index("permno").ticker
    P["tic"] = P.permno.map(nm_)
    P["ret_w"] = P.groupby("month").ret_f1.transform(lambda s: ap.winsor(s, 0.01))
    specs = [("Baseline", P, "sig", "ret_f1"),
             ("Excluding January 2021 (meme-stock episode)", P[P.month != pd.Period("2020-12", "M")], "sig", "ret_f1"),
             ("Excluding GME, AMC, BBBY, EXPR (all months)", P[~P.tic.isin(["GME", "AMC", "BBBY", "EXPR"])], "sig", "ret_f1"),
             ("Returns winsorized at 1\\%/99\\% each month", P, "sig", "ret_w"),
             ("Excluding return months in 2020", P[(P.month + 1).dt.year != 2020], "sig", "ret_f1"),
             ("Excluding formation months in 2020 (returns 2020:02--2021:01)", P[P.month.dt.year != 2020], "sig", "ret_f1"),
             ("First half (formation 2019:07--2022:06)", P[P.month <= pd.Period("2022-06", "M")], "sig", "ret_f1"),
             ("Second half (formation 2022:07--2025:11)", P[P.month > pd.Period("2022-06", "M")], "sig", "ret_f1"),
             ("Price $\\geq$ \\$5", P[P.prc >= 5], "sig", "ret_f1"),
             ("Excluding microcaps (NYSE 20th pct.)", P[P.me >= P.nyse20], "sig", "ret_f1"),
             ("Skip a month (return in $t+2$)", P, "sig", "ret_f2"),
             ("Card coverage $\\geq$ 0.25\\% of sales", P[P.cov_ratio >= 0.0025], "sig", "ret_f1")]
    for c, lab in [("co_eq", "Overlapping peers equal-weighted"), ("CO_ASYM", "One-directional overlap (share of $i$'s customers)"),
                   ("CO_W6", "6-month network window"), ("CO_W24", "24-month network window"),
                   ("CO_FROZEN", "Frozen 2019 network (formation 2020--2025)"),
                   ("co_strong", "Strongest third of each firm's links"), ("co_ntnic", "Excluding TNIC-3 rivals"),
                   ("RPS_pos", "Shared tastes (RPS) instead of overlap"), ("RPS_FROZEN", "Shared tastes, frozen 2019 network"),
                   ("RPS_D_pos", "Shared tastes, location-free channels only")]:
        if c in P:
            D = P.copy(); D["sig2"] = D.groupby("month")[c].transform(lambda s: ap.winsor(s))
            specs.append((lab, D, "sig2", "ret_f1"))
    rows, raw = [], []
    for lab, D, s, rt in specs:
        try:
            o = evaluate(D, s, rt)
        except Exception as e:
            print("fail", lab, e); continue
        raw.append((lab, o))
        rows.append([lab, f"{o['ew'][0]:.2f}{ap.stars(o['ew'][1])}", f"({o['ew'][1]:.2f})",
                     f"{o['vw'][0]:.2f}{ap.stars(o['vw'][1])}", f"({o['vw'][1]:.2f})",
                     f"{o['fm'][0]:.3f}{ap.stars(o['fm'][1])}", f"({o['fm'][1]:.2f})", str(o["T"])])
        print(rows[-1], flush=True)
    hdr = [["", "\\multicolumn{2}{c}{EW FF6 $\\alpha$ (\\%)}", "\\multicolumn{2}{c}{VW FF6 $\\alpha$ (\\%)}",
            "\\multicolumn{2}{c}{FM slope}", "Months"],
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}"]
    notes = ("Each row re-estimates the main tests under one change. EW/VW FF6 $\\alpha$: monthly alpha of the quintile "
             "long-short portfolio (Table~\\ref{tab:alphas}). FM slope: Fama-MacBeth coefficient on the peer return with the "
             "controls of Table~\\ref{tab:fm}, column (5). Rows 2--12 change the sample, the return, or the test assets; "
             "the next seven rows re-estimate customer overlap with the stated change (equal weights on all overlapping peers; "
             "the share of $i$'s customers who shop at $j$ instead of the symmetric average; a 6- or 24-month window; a network "
             "estimated once from 2019 card data and held fixed; only the strongest third of each firm's links; or excluding peers "
             "that are TNIC-3 rivals); the last three rows replace customer overlap by RPS. Newey-West $t$-statistics "
             "(four lags) in parentheses.")
    open(f"{R}/tables/tab_robust.tex", "w").write(
        ap.latex_table(hdr, rows, "Robustness", "tab:robust", notes, colspec="lccccccc", size="\\scriptsize"))
    import json; json.dump({k: v for k, v in raw}, open(f"{R}/logs/robust_raw.json", "w"), default=float, indent=0)
    act, null, p = permutation(P)
    open(f"{R}/logs/permutation.txt", "w").write(f"actual={act:.5f} null_mean={null.mean():.5f} null_p95={np.quantile(null,.95):.5f} p={p:.4f}\n")
    print("permutation", act, null.mean(), np.quantile(null, .95), p)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "perm":
        P = M.load(SIG)
        act, null, p = permutation(P)
        open(f"{R}/logs/permutation.txt", "w").write(f"actual={act:.5f} null_mean={null.mean():.5f} null_p95={np.quantile(null,.95):.5f} p={p:.4f}\n")
        print("permutation", act, null.mean(), np.quantile(null, .95), p)
    else:
        main()
