"""
46_overlap_validation.py -- validation of the customer-overlap measure.

From the POI-level detail in data/extract/val_SAME and val_brand (45_overlap_validation_extract.py), rebuild
customer overlap for every formation month with the window t-12..t-1 and the brand -> firm map of
19_overlap_family.py, check that the rebuild reproduces the baseline network, and build variants:
  CO_MAX   peer brands combined by the largest share instead of 1 - prod(1 - A_b)   (perfectly overlapping clienteles)
  CO_SUM   peer brands combined by min(1, sum_b A_b)                                (disjoint clienteles)
  CO_POI   every POI-month weighted equally instead of by its customers
  CO_W1    one-month window (card data for month t-1 only)
  CO_UB    censoring upper bound: at every POI of i whose same-category list is full (20 brands) and does not
           list the peer brand, the peer's share is set to the smallest listed share (the most it can be)
Each variant is symmetrized like the baseline, turned into a peer return with the baseline's (winsorized) peer
returns, and evaluated as in Table 16. Diagnostics describe how often the three issues arise.
Output: tables/tab_validation.tex, logs/validation.json, data/derived/co_validation.parquet
"""
import os, glob
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
R = ap.R
EX = f"{R}/data/extract"


def brand_map(mp, month):
    mid = pd.Timestamp(month + "-15")
    m = mp[(mp.start <= mid) & (mp.end >= mid)].dropna(subset=["permno"])
    return dict(zip(m.brand, m.permno.astype(int)))


def month_tables(mp, months):
    """firm-level monthly aggregates; brands assigned to firms with the map of that month"""
    F, Rl = {}, {}
    for m in months:
        b2p = brand_map(mp, m)
        vb = pd.read_parquet(f"{EX}/val_brand/{m}.parquet"); vb["i"] = vb.brand.map(b2p); vb = vb.dropna(subset=["i"])
        F[m] = vb.groupby("i").agg(W=("w_sum", "sum"), NP=("n_poi", "sum"), TW=("thr_w", "sum"), WF=("w_full", "sum"),
                                   LW=("len_w", "sum")).reset_index().astype({"i": int})
        vs = pd.read_parquet(f"{EX}/val_SAME/{m}.parquet"); vs["i"] = vs.brand.map(b2p); vs = vs.dropna(subset=["i"])
        Rl[m] = vs.groupby(["i", "b"]).agg(a=("act_num", "sum"), ap_=("act_poi", "sum"), tl=("thr_lst", "sum")).reset_index().astype({"i": int})
    return F, Rl


def window_networks(win, F, Rl, mp):
    f = pd.concat([F[m] for m in win]).groupby("i").sum()
    f = f[f.W >= 2000]
    r = pd.concat([Rl[m] for m in win]).groupby(["i", "b"]).sum().reset_index()
    r = r[r.i.isin(f.index)]
    b2p = brand_map(mp, win[-1])
    r["j"] = r.b.map(b2p); r = r.dropna(subset=["j"]); r["j"] = r.j.astype(int); r = r[r.j != r.i]
    W, NP, TW = r.i.map(f.W), r.i.map(f.NP), r.i.map(f.TW)
    r["A"] = (r.a / W / 100).clip(0, 1)                         # baseline brand-level share
    r["Apoi"] = (r.ap_ / NP / 100).clip(0, 1)                   # POI-equal weights
    r["Aub"] = ((r.a + TW - r.tl) / W / 100).clip(0, 1)         # censoring upper bound
    for c in ("A", "Apoi", "Aub"):                              # union across a peer's brands: 1 - prod(1 - A_b)
        r["l" + c] = np.log1p(-r[c].clip(upper=1 - 1e-12))
    r["pos"] = (r.A > 0).astype(int)
    g = r.groupby(["i", "j"]).agg(lA=("lA", "sum"), lApoi=("lApoi", "sum"), lAub=("lAub", "sum"), CO_MAX=("A", "max"),
                                  CO_SUM=("A", "sum"), nb=("pos", "sum")).reset_index()
    g["CO"] = -np.expm1(g.lA); g["CO_POI"] = -np.expm1(g.lApoi); g["CO_UB"] = -np.expm1(g.lAub)
    g["CO_SUM"] = g.CO_SUM.clip(upper=1.0)
    g = g[["i", "j", "CO", "CO_MAX", "CO_SUM", "CO_POI", "CO_UB", "nb"]]
    cols = ["CO", "CO_MAX", "CO_SUM", "CO_POI", "CO_UB"]
    gt = g.rename(columns={"i": "j", "j": "i", **{c: c + "_t" for c in cols}, "nb": "nb_t"})
    s = g.merge(gt, on=["i", "j"], how="outer").fillna(0.0)
    for c in cols:
        s[c] = (s[c] + s[c + "_t"]) / 2
    s["multi_brand"] = (s.nb > 1) | (s.nb_t > 1)
    diag = {"cens_share": float(f.WF.sum() / f.W.sum()), "thr_mean": float(f.TW.sum() / max(f.WF.sum(), 1)),
            "list_len": float(f.LW.sum() / f.W.sum())}
    return s[["i", "j"] + cols + ["multi_brand"]], diag


def main():
    mp = pd.read_parquet(f"{R}/data/derived/brand_firm_map.parquet")
    allm = sorted(os.path.basename(f)[:7] for f in glob.glob(f"{EX}/val_SAME/*.parquet"))
    P = pd.read_parquet(f"{R}/data/derived/panel_final.parquet", columns=["permno", "month", "ret"])
    F, Rl = month_tables(mp, allm)
    rows, diags, checks = [], [], []
    for t, d in P.groupby("month"):
        k = allm.index(str(t)) if str(t) in allm else len(allm)
        win = allm[max(0, k - 12):k]
        if len(win) < 6:
            continue
        S, dg = window_networks(win, F, Rl, mp)
        U = d.set_index("permno")
        S = S[S.i.isin(U.index) & S.j.isin(U.index)]
        L = S[S.CO > 0]
        # does the rebuild reproduce the baseline network?
        O = pd.read_parquet(f"{R}/data/derived/ovl/{t}.parquet", columns=["i", "j", "A_S"])
        O = O[O.i.isin(U.index) & O.j.isin(U.index)].merge(S[["i", "j", "CO"]], on=["i", "j"], how="outer").fillna(0)
        checks.append(float(np.corrcoef(O.A_S, O.CO)[0, 1]))
        dg.update(month=str(t), multi_share=float(L.multi_brand.mean()),
                  max_over_union=float((L.CO_MAX / L.CO).median()), sum_over_union=float((L.CO_SUM / L.CO).median()),
                  ub_over_base=float((L.CO_UB / L.CO).median()),
                  **{f"corr_{c}": float(L[["CO", c]].corr().iloc[0, 1]) for c in ["CO_MAX", "CO_SUM", "CO_POI", "CO_UB"]})
        diags.append(dg)
        S = S.assign(rj=S.j.map(U.ret))
        o = {}
        for c in ["CO", "CO_MAX", "CO_SUM", "CO_POI", "CO_UB"]:
            w = S[c].clip(lower=0)
            o[c] = (w * S.rj).groupby(S.i).sum() / w.groupby(S.i).sum().where(lambda x: x > 0)
        o = pd.DataFrame(o); o["month"] = t
        rows.append(o.reset_index().rename(columns={"i": "permno"}))
        # one-month window (card data for month t-1 only)
        S1, _ = window_networks(allm[k - 1:k], F, Rl, mp)
        S1 = S1[S1.i.isin(U.index) & S1.j.isin(U.index)].assign(rj=lambda x: x.j.map(U.ret))
        w = S1.CO.clip(lower=0)
        rows[-1] = rows[-1].merge(((w * S1.rj).groupby(S1.i).sum() / w.groupby(S1.i).sum().where(lambda x: x > 0))
                                  .rename("CO_W1").reset_index().rename(columns={"i": "permno"}), on="permno", how="left")
        print(t, len(L), round(checks[-1], 4), flush=True)
    V = pd.concat(rows, ignore_index=True).rename(columns={"CO": "CO_REBUILT"})
    V.to_parquet(f"{R}/data/derived/co_validation.parquet", index=False)
    D = pd.DataFrame(diags)
    # evaluate every variant exactly as the rows of Table 16
    RB = import_module("17_robustness"); M = import_module("13_main_tables")
    Pm = M.load("co").merge(V, on=["permno", "month"], how="left")
    res = {}
    for c in ["CO_REBUILT", "CO_MAX", "CO_SUM", "CO_POI", "CO_W1", "CO_UB"]:
        X = Pm.copy(); X["s2"] = X.groupby("month")[c].transform(lambda s: ap.winsor(s))
        res[c] = RB.evaluate(X, "s2", "ret_f1")
    summ = {"rebuild_corr_min": min(checks), "rebuild_corr_mean": float(np.mean(checks)),
            **{k: float(D[k].mean()) for k in D.columns if k != "month"}, "tests": res}
    json.dump(summ, open(f"{R}/logs/validation.json", "w"), default=float, indent=1)
    # ------------------------------------------------------------------ table
    pc = lambda x: f"{100 * x:.1f}"
    rows_ = ["\\multicolumn{8}{l}{\\emph{Panel A: How often the issues arise (averages over formation months)}} \\\\",
             ["Customers at stores whose same-category list is full (20 brands), \\%", "\\multicolumn{7}{c}{" + pc(summ["cens_share"]) + "}"],
             ["Smallest listed share at stores with full lists (censoring threshold), \\%", "\\multicolumn{7}{c}{" + f"{summ['thr_mean']:.1f}" + "}"],
             ["Linked pairs in which a firm has more than one linked brand, \\%", "\\multicolumn{7}{c}{" + pc(summ["multi_share"]) + "}"],
             *[[f"Median ratio to baseline overlap, linked pairs: {lab}", "\\multicolumn{7}{c}{" + f"{summ[k]:.2f}" + "}"]
               for k, lab in [("max_over_union", "largest brand"), ("sum_over_union", "capped sum"), ("ub_over_base", "censoring bound")]],
             *[[f"Correlation with baseline overlap, linked pairs: {lab}", "\\multicolumn{7}{c}{" + f"{summ['corr_' + k]:.3f}" + "}"]
               for k, lab in [("CO_MAX", "largest brand"), ("CO_SUM", "capped sum"), ("CO_POI", "equal store weights"),
                              ("CO_UB", "censoring bound")]],
             "\\addlinespace",
             "\\multicolumn{8}{l}{\\emph{Panel B: Tests with each version of the measure}} \\\\",
             ["", "\\multicolumn{2}{c}{EW FF6 $\\alpha$ (\\%)}", "\\multicolumn{2}{c}{VW FF6 $\\alpha$ (\\%)}",
              "\\multicolumn{2}{c}{FM slope}", "Months"],
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}"]
    for c, lab in [("CO_REBUILT", "Baseline (rebuilt from store-level detail)"),
                   ("CO_MAX", "Peer brands combined by the largest share"), ("CO_SUM", "Peer brands combined by the capped sum"),
                   ("CO_POI", "Every store-month weighted equally"), ("CO_W1", "One-month window (card data for $t-1$)"),
                   ("CO_UB", "Censoring upper bound")]:
        o = res[c]
        rows_.append([lab, f"{o['ew'][0]:.2f}{ap.stars(o['ew'][1])}", f"({o['ew'][1]:.2f})",
                      f"{o['vw'][0]:.2f}{ap.stars(o['vw'][1])}", f"({o['vw'][1]:.2f})",
                      f"{o['fm'][0]:.3f}{ap.stars(o['fm'][1])}", f"({o['fm'][1]:.2f})", str(o["T"])])
    notes = ("Panel A: shares computed from the same-category cross-shopping list of every store (POI) of the sample firms. "
             "The censoring threshold is the smallest share listed at stores whose list holds the maximum of twenty brands; a brand "
             "that is not listed there has a share below it. Ratios and correlations are medians and averages over linked pairs "
             "(positive baseline overlap). Panel B: the baseline rebuilt from store-level detail (its monthly correlation with the "
             f"network of Section~\\ref{{sec:co}} averages {summ['rebuild_corr_mean']:.4f}) and five variants: a peer firm's brands "
             "combined by the largest brand-level share (as if the brands' clienteles coincided) or by the sum capped at one (as if "
             "they were disjoint) instead of $1-\\prod_b(1-A_{ib})$; every store-month weighted equally instead of by its customers; "
             "a window of one month instead of twelve, so that no customer is counted in more than one month; and an upper bound for "
             "censoring that sets a peer brand's share at every full list that omits it to that list's smallest share. Tests as in "
             "Table~\\ref{tab:robust}. Newey-West $t$-statistics (four lags) in parentheses.")
    tex = ap.latex_table([], rows_, "Validation of the customer-overlap measure", "tab:validation", notes,
                         colspec="p{7.2cm}ccccccc", size="\\scriptsize")
    open(f"{R}/tables/tab_validation.tex", "w").write(tex)
    print(json.dumps({k: v for k, v in summ.items() if k != "tests"}, indent=1))
    for c, o in res.items():
        print(c, {k: (tuple(round(x, 3) for x in v) if isinstance(v, tuple) else v) for k, v in o.items()})


if __name__ == "__main__":
    main()
