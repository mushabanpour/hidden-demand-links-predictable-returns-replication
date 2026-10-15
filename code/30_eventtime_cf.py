"""
30_eventtime_cf.py -- event-time CAR in the style of Cohen and Frazzini (2008, Figure 2).

At the end of each formation month t, firms are sorted into quintiles on the month-t
customer-overlap peer return (k = 0 is the signal month; holding starts at k = 1).
For k = -12..24 we compute
  * the focal long-short return: Q5 minus Q1 return of the focal firms in month t+k
  * the sorting-variable long-short return: Q5 minus Q1 of the firms' peer-portfolio
    return in month t+k, with the overlap network and peer set held fixed at t
and cumulate both from k = -12. Panels: equal-weighted, value-weighted (month-t market
cap), and equal-weighted excluding calendar month January 2021 (GameStop/AMC).
Also writes a table of cumulative returns over event windows with Newey-West t-stats.
"""
import os
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")
R = ap.R
K = list(range(-12, 25))
JAN21 = pd.Period("2021-01", "M")
import figstyle as fs


def build():
    P = M.load("co")
    d = P.dropna(subset=["sig", "ret_f1"]).copy()
    d["q"] = d.groupby("month").sig.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    d = d[d.q.isin([1, 5])][["permno", "month", "q", "me"]].rename(columns={"month": "t"})
    sm = pd.read_parquet(f"{R}/data/derived/stock_month.parquet", columns=["permno", "month", "ret"]).dropna()
    # peer sets fixed at formation: same-category overlap peers within the month-t universe
    pairs = []
    for t, g in P.groupby("month"):
        if t not in set(d.t):
            continue
        U = set(g.permno)
        O = pd.read_parquet(f"{R}/data/derived/ovl/{t}.parquet", columns=["i", "j", "A_S"])
        O = O[O.i.isin(set(d[d.t == t].permno)) & O.j.isin(U) & (O.A_S > 0)]
        pairs.append(O.assign(t=t))
    pairs = pd.concat(pairs).rename(columns={"i": "permno"})
    rows = []
    for k in K:
        x = sm.rename(columns={"month": "m"}).copy(); x["t"] = x.m - k
        f = d.merge(x[["permno", "t", "ret"]], on=["permno", "t"], how="inner")          # focal return in t+k
        pj = pairs.merge(x[["permno", "t", "ret"]].rename(columns={"permno": "j"}), on=["j", "t"], how="inner")
        pr = (pj.assign(wr=pj.A_S * pj.ret).groupby(["t", "permno"]).agg(wr=("wr", "sum"), w=("A_S", "sum")))
        pr = (pr.wr / pr.w).rename("pret").reset_index()
        f = f.merge(pr, on=["t", "permno"], how="left")
        f["k"] = k; f["cal"] = f.t + k
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def ls_series(E, col, vw, drop_jan21):
    D = E.dropna(subset=[col])
    if drop_jan21:
        D = D[D.cal != JAN21]
    if vw:
        g = D.assign(wr=D[col] * D.me).groupby(["t", "k", "q"]).agg(wr=("wr", "sum"), w=("me", "sum"))
        r = (g.wr / g.w).unstack("q")
    else:
        r = D.groupby(["t", "k", "q"])[col].mean().unstack("q")
    return (r[5] - r[1]).unstack("k")            # formation month x k


def window(L, a, b):
    cols = list(range(a, b + 1))
    s = L[cols].dropna().sum(1)
    m, t, n = ap.nw_mean(s, max(4, b - a + 1))
    return m * 100, t, n


def main():
    E = build()
    specs = [("ew", False, False, "A. Equal-weighted"), ("vw", True, False, "B. Value-weighted"),
             ("ewx", False, True, "C. Equal-weighted, excluding January 2021")]
    res = {}
    fs.apply()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for (key, vw, dj, title), ax in zip(specs, axes):
        L = ls_series(E, "ret", vw, dj); S = ls_series(E, "pret", vw, dj)
        res[key] = (L, S)
        car = L.mean().cumsum() * 100; scar = S.mean().cumsum() * 100
        fs.zero_line(ax)
        ax.axvline(0, color=fs.BLACK, lw=0.7, ls=":", zorder=0)
        fs.line(ax, car.index, car.values, 0, label="Long-short portfolio (focal firms)")
        fs.line(ax, scar.index, scar.values, 1, label="Sorting variable (overlap-peer portfolio)")
        ax.set_xlabel("Month $t+k$"); ax.set_xticks(range(-12, 25, 6))
    axes[0].set_ylabel("Cumulative return, percent")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(f"{R}/figures/fig_eventtime.pdf"); plt.close()

    wins = [(-12, -1, "$[-12,-1]$"), (0, 0, "$[0]$"), (1, 1, "$[1]$"), (2, 6, "$[2,6]$"), (7, 12, "$[7,12]$"),
            (13, 24, "$[13,24]$"), (1, 12, "$[1,12]$"), (1, 24, "$[1,24]$")]
    hdr = [["", *[w[2] for w in wins]]]
    rows, raw = [], {}
    for key, lab in [("ew", "Equal-weighted"), ("vw", "Value-weighted"), ("ewx", "Equal-weighted, excluding January 2021")]:
        L, S = res[key]
        rows.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{lab}}}}} \\\\")
        for nm, X in [("Long-short (focal firms)", L), ("Sorting variable (peers)", S)]:
            rb, rt = [f"\\quad {nm}"], [""]
            for a, b, _ in wins:
                m, t, n = window(X, a, b)
                rb.append(f"{m:.2f}{ap.stars(t)}"); rt.append(f"({t:.2f})")
                raw[(key, nm[:4], a, b)] = (m, t, n)
            rows += [rb, rt]
        rows.append(["\\quad Formation months"] + [str(window(L, a, b)[2]) for a, b, _ in wins])
        if key != "ewx":
            rows.append("\\addlinespace")
    m0, t0, _ = window(res["ew"][1], 0, 0)
    notes = ("At the end of each month $t$ firms are sorted into quintiles on the month-$t$ customer-overlap peer return. "
             "The table reports the average cumulative return (in percent) over event windows in months $t+k$ of the focal "
             "long-short portfolio (top minus bottom quintile) and of the same difference in the firms' overlap-peer portfolios, "
             "whose composition and weights are fixed at formation (the sorting variable); $k=0$ is the signal month and "
             "holding begins at $k=1$. Each window "
             "uses the formation months for which all of its months are observed. In the last panel, calendar month January "
             "2021 is excluded wherever it falls. Newey-West $t$-statistics, with lags equal to the window length (at least four), "
             "in parentheses.")
    open(f"{R}/tables/tab_eventtime.tex", "w").write(
        ap.latex_table(hdr, rows, "Event-time returns of the customer-overlap long-short portfolio", "tab:eventtime", notes,
                       size="\\footnotesize"))
    for k, v in raw.items():
        print(k, np.round(v[:2], 2), v[2])
    for key in res:
        L, S = res[key]
        print(key, "CAR k=0,1,6,12,24:", np.round(L.mean().cumsum().loc[[0, 1, 6, 12, 24]].values * 100, 2),
              "peer CAR:", np.round(S.mean().cumsum().loc[[-1, 0, 1, 12, 24]].values * 100, 2))


if __name__ == "__main__":
    main()
