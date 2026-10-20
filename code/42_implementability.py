"""
42_implementability.py -- turnover, concentration, and trading-cost sensitivity of the quintile
long-short portfolio (Table 5, Panel A).

Weights: equal or market-cap (end of month t) within the top and bottom quintiles of the month-t
customer-overlap peer return. Turnover at each rebalancing is sum_i |w_new - w_drift| per leg, where
w_drift are the previous portfolio's weights after one month of returns (stocks that leave count in
full); the first portfolio is bought from cash. A one-way cost c per dollar traded lowers the month's
long-short return by c x (turnover of the long leg + turnover of the short leg). Net returns are
regressed on the six factors of Table 6; the break-even cost sets the net six-factor alpha to zero.
Output: tables/tab_costs.tex, logs/costs.json
"""
import os
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")
R = ap.R
COSTS = [0.0010, 0.0025, 0.0050]          # one-way cost per dollar traded: 10, 25, 50 basis points
JAN21 = pd.Period("2020-12", "M")          # formation month of the January 2021 return


def legs(P, nq=5):
    d = P.dropna(subset=["sig", "ret_f1"]).copy()
    d["q"] = d.groupby("month").sig.transform(lambda s: pd.qcut(s.rank(method="first"), nq, labels=False) + 1)
    return d[d.q.isin([1, nq])], nq


def leg_stats(d, weight):
    """per formation month: weights, gross return, turnover (sum |dw|) vs the drifted previous weights"""
    out, prev = [], None
    for t, g in d.groupby("month"):
        w = (pd.Series(1.0, index=g.permno) if weight == "ew" else pd.Series(g.me.values, index=g.permno))
        w = w / w.sum()
        r = pd.Series(g.ret_f1.values, index=g.permno)
        traded = 1.0 if prev is None else (w.reindex(w.index.union(prev.index), fill_value=0)
                                           - prev.reindex(w.index.union(prev.index), fill_value=0)).abs().sum()
        drift = w * (1 + r); prev = drift / drift.sum()
        out.append(dict(month=t, ret=(w * r).sum(), traded=traded, n=len(w), wmax=w.max(), top5=w.nlargest(5).sum(),
                        eff_n=1 / (w ** 2).sum()))
    return pd.DataFrame(out).set_index("month")


def summarize(L, S, Fn, label, drop=None):
    if drop is not None:
        L, S = L.drop(drop, errors="ignore"), S.drop(drop, errors="ignore")
    ls = L.ret - S.ret
    trd = L.traded + S.traded
    gross = ap.ts_reg(ls, Fn, "FF6")
    res = {"label": label, "months": len(ls), "gross": (gross["alpha"], gross["t_alpha"]),
           "turnover_long": L.traded.iloc[1:].mean() / 2, "turnover_short": S.traded.iloc[1:].mean() / 2,
           "n_long": L.n.mean(), "n_short": S.n.mean(), "wmax_long": L.wmax.mean(), "wmax_short": S.wmax.mean(),
           "top5_long": L.top5.mean(), "top5_short": S.top5.mean(), "effn_long": L.eff_n.mean(), "effn_short": S.eff_n.mean()}
    for c in COSTS:
        a = ap.ts_reg(ls - c * trd, Fn, "FF6"); res[f"net_{int(c * 1e4)}"] = (a["alpha"], a["t_alpha"])
    res["breakeven_bp"] = gross["alpha"] / trd.mean() * 1e4     # alpha is linear in c: alpha(c) = alpha(0) - c * mean(trd)
    return res


def main():
    P = M.load("co")
    # capped value weights: market capitalization capped at the NYSE 80th percentile of month t (Jensen, Kelly, and Pedersen, 2023)
    sm = pd.read_parquet(f"{R}/data/derived/stock_month.parquet", columns=["permno", "month", "me", "nyse"])
    P = P.merge(sm[sm.nyse].groupby("month").me.quantile(0.8).rename("nyse80"), left_on="month", right_index=True, how="left")
    d, nq = legs(P)
    Fn = M.Fn
    out = {}
    for w in ("ew", "vw", "vwc"):
        dd = d.assign(me=np.minimum(d.me, d.nyse80)) if w == "vwc" else d
        L = leg_stats(dd[dd.q == nq], "ew" if w == "ew" else "vw"); S = leg_stats(dd[dd.q == 1], "ew" if w == "ew" else "vw")
        out[w] = summarize(L, S, Fn, w)
        if w == "ew":
            out["ew_x"] = summarize(L, S, Fn, "ew_x", drop=[JAN21])
    # the largest names in the value-weighted legs (average weight over formation months)
    nm = pd.read_parquet(f"{R}/data/raw/crsp_stocknames_v2.parquet").sort_values("nameenddt").groupby("permno").tail(1).set_index("permno").ticker
    w_ = d.assign(w=d.me / d.groupby(["month", "q"]).me.transform("sum"))
    avg = w_.groupby(["q", "permno"]).w.sum() / d.month.nunique()
    out["vw_top_names"] = {("long" if q == nq else "short"): [(nm.get(p, str(p)), float(v)) for p, v in avg.loc[q].nlargest(5).items()]
                           for q in (1, nq)}
    top10 = P.dropna(subset=["ret_f1"]).groupby("month").me.apply(lambda s: s.nlargest(10).sum() / s.sum())
    out["top10_share_of_sample_cap"] = float(top10.mean())
    json.dump(out, open(f"{R}/logs/costs.json", "w"), default=float, indent=1)
    cols = [("ew", "EW"), ("ew_x", "EW, excl.\\ Jan.\\ 2021"), ("vw", "VW"), ("vwc", "Capped VW")]
    pc = lambda x: f"{100 * x:.1f}"
    rows = ["\\multicolumn{5}{l}{\\emph{Panel A: Portfolio composition}} \\\\"]
    for k, lab, f in [("n_long", "Stocks in the long leg", lambda x: f"{x:.1f}"),
                      ("n_short", "Stocks in the short leg", lambda x: f"{x:.1f}"),
                      ("turnover_long", "Monthly turnover, long leg (\\%)", pc),
                      ("turnover_short", "Monthly turnover, short leg (\\%)", pc),
                      ("wmax_long", "Largest weight, long leg (\\%)", pc),
                      ("top5_long", "Five largest weights, long leg (\\%)", pc),
                      ("effn_long", "Effective number of stocks, long leg", lambda x: f"{x:.1f}"),
                      ("wmax_short", "Largest weight, short leg (\\%)", pc),
                      ("top5_short", "Five largest weights, short leg (\\%)", pc),
                      ("effn_short", "Effective number of stocks, short leg", lambda x: f"{x:.1f}")]:
        rows.append([lab] + [f(out[c][k]) for c, _ in cols])
    rows.append("\\addlinespace")
    rows.append("\\multicolumn{5}{l}{\\emph{Panel B: Six-factor alpha net of trading costs (\\% per month)}} \\\\")
    for k, lab in [("gross", "Gross"), ("net_10", "One-way cost of 10 bp"), ("net_25", "One-way cost of 25 bp"),
                   ("net_50", "One-way cost of 50 bp")]:
        rows.append([lab] + [f"{100 * out[c][k][0]:.2f}{ap.stars(out[c][k][1])}" for c, _ in cols])
        rows.append([""] + [f"({out[c][k][1]:.2f})" for c, _ in cols])
    rows.append(["Break-even one-way cost (bp)"] + [f"{out[c]['breakeven_bp']:.0f}" for c, _ in cols])
    rows.append(["Months"] + [str(out[c]["months"]) for c, _ in cols])
    notes = ("Quintile long-short portfolio of Table~\\ref{tab:sorts}, Panel A. Capped VW caps each firm's market capitalization at "
             "the month-$t$ NYSE 80th percentile before value-weighting \\citep{JensenKellyPedersen2023}. Turnover is half the sum of absolute weight changes "
             "at each monthly rebalancing, measured against the previous portfolio's weights after one month of returns (stocks that "
             "leave the portfolio count in full), averaged over rebalancing months. The effective number of stocks is the inverse of "
             "the sum of squared weights. Panel B subtracts, each month, a one-way cost per dollar traded times the dollars traded in "
             "both legs (twice the one-sided turnover of each leg; the first portfolio is bought from cash) and reports the "
             "six-factor alpha of the net long-short return. The break-even cost is the one-way cost at which that alpha is zero. "
             "Newey-West $t$-statistics (four lags) in parentheses.")
    tex = ap.latex_table([["", *[c for _, c in cols]]], rows, "Turnover, concentration, and trading costs", "tab:costs", notes,
                         colspec="lcccc")
    open(f"{R}/tables/tab_costs.tex", "w").write(tex)
    print(json.dumps(out, default=float, indent=1))


if __name__ == "__main__":
    main()
