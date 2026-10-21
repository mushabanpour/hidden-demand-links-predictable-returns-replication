"""
56_placebo_ext.py -- placebo networks (Table: Placebo networks; Figure 4 shows the first design, equal-weighted).

Each placebo keeps the customer-overlap matrix and relabels firms within groups, recomputes the peer return (winsorized as
the paper's signal), and re-forms the quintile long-short portfolio, for equal weights, value weights, and value weights
capped at the NYSE 80th percentile of month t (Table 19). Designs, 500 replications each:
  category        firms reshuffled within their month-t SafeGraph category, independently every month (seed 7, as in
                  Figure 4 and 17_robustness.py)
  category_size   reshuffled within category x size tercile (terciles of market cap within the category in month t), so a
                  placebo peer is about the size of the firm it replaces
  persistent      one random relabeling per replication, held fixed across months: within each category, firms ordered by
                  PERMNO are mapped to the same firms ordered by a random key drawn once per replication
For the category design the industry-adjusted peer return (equation (8)) is re-estimated for every placebo network.
p is the share of placebo alphas at least as large as the actual alpha.
Output: tables/tab_placebo.tex, logs/placebo_ext.json
"""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")
R = ap.R
NREP = int(os.environ.get("NREP", 500))
SEEDS = {"category": 7, "category_size": 8, "persistent": 9}
W = ("ew", "vw", "vwc")


def build():
    P = M.load("co")
    pr = pd.read_parquet(f"{R}/data/derived/panel_final.parquet", columns=["permno", "month", "ret"])   # peer returns as in the signal
    sm = pd.read_parquet(f"{R}/data/derived/stock_month.parquet", columns=["permno", "month", "me", "nyse"])
    cap = sm[sm.nyse].groupby("month").me.quantile(0.8)
    mats = []
    for t, d in P.groupby("month"):
        U = pd.Index(sorted(d.permno.unique()))
        O = pd.read_parquet(f"{R}/data/derived/ovl/{t}.parquet", columns=["i", "j", "A_S"])
        O = O[O.i.isin(U) & O.j.isin(U)]
        Wm = np.zeros((len(U), len(U))); Wm[U.get_indexer(O.i), U.get_indexer(O.j)] = O.A_S.clip(lower=0).fillna(0).values
        x = d.set_index("permno").reindex(U)
        Z = x[["ind_u", "indret", "cat_ew", "cat_vw"]]
        Z = np.column_stack([np.ones(len(U)), Z.fillna(Z.mean()).values])
        me = x.me.values
        cat = x.cat.fillna("NA").values
        gc = [np.where(cat == c)[0] for c in np.unique(cat)]
        gs = []
        for g in gc:
            terc = pd.qcut(pd.Series(me[g]).rank(method="first"), 3, labels=False).values if len(g) >= 3 else np.zeros(len(g))
            gs += [g[terc == k] for k in np.unique(terc)]
        mats.append(dict(t=t, W=Wm, r=np.nan_to_num(pr[pr.month == t].set_index("permno").ret.reindex(U).values),
                         y=x.ret_f1.values, me=me, mec=np.minimum(me, cap.get(t, np.inf)), Z=Z, gc=gc, gs=gs, permno=U.values))
    return mats


def series(mats, design=None, rng=None, key=None, adjust=False):
    out = {f"{k}_{w}": {} for k in (("raw", "adj") if adjust else ("raw",)) for w in W}
    for m in mats:
        p = np.arange(len(m["r"]))
        if design == "category":
            for g in m["gc"]:
                p[g] = rng.permutation(g)
        elif design == "category_size":
            for g in m["gs"]:
                p[g] = rng.permutation(g)
        elif design == "persistent":
            for g in m["gc"]:
                p[g[np.argsort(m["permno"][g])]] = g[np.argsort(key.reindex(m["permno"][g]).values)]
        Wp = m["W"][np.ix_(p, p)]
        den = Wp.sum(1)
        s = np.where(den > 0, Wp @ m["r"] / np.where(den > 0, den, 1), np.nan)
        has = ~np.isnan(s); ok = has & ~np.isnan(m["y"])
        if ok.sum() < 25:
            continue
        lo_, hi_ = np.nanquantile(s[has], [0.01, 0.99]); s = np.where(has, np.clip(s, lo_, hi_), np.nan)
        sigs = [("raw", s[ok])]
        if adjust:
            b = np.linalg.lstsq(m["Z"][has], s[has], rcond=None)[0]
            sigs.append(("adj", s[ok] - m["Z"][ok] @ b))
        for kind, sig in sigs:
            q = pd.qcut(pd.Series(sig).rank(method="first"), 5, labels=False).values
            yy = m["y"][ok]; hi, lo = q == 4, q == 0
            out[f"{kind}_ew"][m["t"]] = yy[hi].mean() - yy[lo].mean()
            for w, wv in (("vw", m["me"][ok]), ("vwc", m["mec"][ok])):
                out[f"{kind}_{w}"][m["t"]] = (yy[hi] * wv[hi]).sum() / wv[hi].sum() - (yy[lo] * wv[lo]).sum() / wv[lo].sum()
    return {k: pd.Series(list(v.values()), index=pd.PeriodIndex(list(v.keys()), freq="M")) for k, v in out.items()}


def alphas(sr):
    return {k: ap.ts_reg(x, M.Fn, "FF6")["alpha"] for k, x in sr.items()}


def main():
    mats = build()
    act_s = series(mats, adjust=True)
    actual = alphas(act_s)
    res = {"actual": actual, "nrep": NREP, "seeds": SEEDS, "months": int(len(act_s["raw_ew"]))}
    permnos = np.unique(np.concatenate([m["permno"] for m in mats]))
    for design, seed in SEEDS.items():
        rng = np.random.default_rng(seed); t0 = time.time(); null = []
        for _ in range(NREP):
            key = pd.Series(rng.random(len(permnos)), index=permnos) if design == "persistent" else None
            null.append(alphas(series(mats, design, rng, key, adjust=(design == "category"))))
        res[design] = {k: {"actual": actual[k], "placebo_mean": float(np.mean([n[k] for n in null])),
                           "p": float(np.mean([n[k] >= actual[k] for n in null]))} for k in null[0]}
        print(design, f"{time.time() - t0:.0f}s", {k: (round(100 * v["placebo_mean"], 2), v["p"]) for k, v in res[design].items()}, flush=True)
    json.dump(res, open(f"{R}/logs/placebo_ext.json", "w"), default=float, indent=1)
    table(res)


def table(res):
    rows = ["\\multicolumn{4}{l}{\\emph{Panel A: Peer return}} \\\\",
            ["Actual network: six-factor alpha (\\%)"] + [f"{100 * res['actual'][f'raw_{w}']:.2f}" for w in W]]
    for d, lab in (("category", "Reshuffled within category, each month"),
                   ("category_size", "Reshuffled within category and size tercile"),
                   ("persistent", "Same relabeling within category in every month")):
        rows.append(f"\\multicolumn{{4}}{{l}}{{{lab}}} \\\\")
        rows.append(["\\quad placebo mean (\\%)"] + [f"{100 * res[d][f'raw_{w}']['placebo_mean']:.2f}" for w in W])
        rows.append(["\\quad $p$-value"] + [f"{res[d][f'raw_{w}']['p']:.3f}" for w in W])
    rows += ["\\addlinespace", "\\multicolumn{4}{l}{\\emph{Panel B: Industry-adjusted peer return}} \\\\",
             ["Actual network: six-factor alpha (\\%)"] + [f"{100 * res['actual'][f'adj_{w}']:.2f}" for w in W],
             "\\multicolumn{4}{l}{Reshuffled within category, each month} \\\\",
             ["\\quad placebo mean (\\%)"] + [f"{100 * res['category'][f'adj_{w}']['placebo_mean']:.2f}" for w in W],
             ["\\quad $p$-value"] + [f"{res['category'][f'adj_{w}']['p']:.3f}" for w in W],
             "\\addlinespace", ["Months"] + [str(res["months"])] * 3, ["Placebo networks per design"] + [str(res["nrep"])] * 3]
    notes = ("Each placebo network keeps the customer-overlap matrix but relabels the firms in its rows and columns: within "
             "their month-$t$ SafeGraph category, independently each month (the design of Figure~\\ref{fig:perm}); within category "
             "and tercile of market capitalization within the category, so that each placebo peer is about the size of the firm it "
             "replaces; or with one random relabeling within each category that is held fixed in every month. For each placebo "
             "network the peer return is recomputed and the quintile long-short portfolio re-formed; Panel B re-estimates the "
             "industry adjustment of equation~\\eqref{eq:adj} for every placebo network. Capped value weights cap market "
             "capitalization at the NYSE 80th percentile of month $t$ (Table~\\ref{tab:costs}). The table reports the six-factor "
             "alpha of the actual network, the mean alpha across placebo networks, and $p$, the share of placebo alphas at least "
             "as large as the actual alpha.")
    hdr = [["", "Equal-weighted", "Value-weighted", "Capped value-weighted"]]
    open(f"{R}/tables/tab_placebo.tex", "w").write(ap.latex_table(hdr, rows, "Placebo networks", "tab:placebo", notes))


if __name__ == "__main__":
    if os.environ.get("TABLE_ONLY"):
        table(json.load(open(f"{R}/logs/placebo_ext.json")))
    else:
        main()
