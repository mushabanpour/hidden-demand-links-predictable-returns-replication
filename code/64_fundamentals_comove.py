"""64_fundamentals_comove.py -- same-quarter comovement of firms' fundamentals with those of their network peers.

Calendar-quarter panel of the sample firms (each fiscal quarter is assigned to the calendar quarter of its middle month).
Every fundamental is a year-over-year change: log growth for positive quantities, the change scaled by total assets a
year earlier for income and cash flows, the change in the ratio for ratios, or the surprise itself; card-spending growth
is the quarterly average of the monthly series of 11_fundamentals.py (all three months required). Variables are
winsorized at 1%/99% over the sample firm-quarters. For each fundamental y and network N, a panel regression

    y_iq = d_q + b * ybar^N_iq + g * ybar^cat_-i,q + f * y_i,q-1 + c'Z_iq + e_iq,   ybar^N_iq = sum_j w_ij y_jq / sum_j w_ij

with networks formed in the first month of q (card data t-12..t-1), quarter fixed effects, standard errors clustered by
firm and quarter, and Z = log size, log book-to-market (0 if book equity <= 0, plus an indicator), momentum. Networks: customer overlap (CO); shared tastes (RPS,
positive part, all sample firms); the local-market and firm-specific components of overlap, entered jointly as in
Table 9 (a firm with no peer in a component gets that quarter's cross-sectional mean); and same-FF48 peers (equal-
weighted, leave-one-out).
Inputs: data/raw/comp_fundq.parquet, comp_fundq_extra.parquet (63), ccm_link.parquet; data/derived/earnings.parquet,
firm_spend.parquet (11), ovl/ (19), sim/ (06); panel from 13_main_tables.load.
Outputs: tables/tab_fundcomove.tex, logs/fundcomove.json
"""
import os
import sys, json, warnings
import numpy as np, pandas as pd, pyfixest as pf
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)   # column-by-column panel build
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")
R = ap.R

GROUPS = [("Earnings", [("eps", "EPS", r"$\Delta$EPS$/P$"), ("ni", "Net income", r"$\Delta$NI$/A$"),
                        ("ib", "Income before extraordinary items", r"$\Delta$IB$/A$"), ("ebt", "Pretax income", r"$\Delta$PI$/A$"),
                        ("sue", "SUE", "surprise"), ("fe", "Analyst forecast error", r"surprise")]),
          ("Sales and costs", [("sale", "Sales growth", r"$\Delta\log$"), ("card", "Card-spending growth", r"$\Delta\log$"),
                               ("rsur", "Revenue surprise", "surprise"), ("cogs", "Cost of goods sold", r"$\Delta\log$"),
                               ("sga", r"SG\&A", r"$\Delta\log$"), ("gm", "Gross margin", r"$\Delta$GP$/$sale")]),
          ("Profitability", [("roa", "ROA", r"$\Delta$IB$/A_{q-1}$"), ("roe", "ROE", r"$\Delta$IB$/BE_{q-1}$"),
                             ("gpa", "Gross profitability", r"$\Delta$GP$/A_{q-1}$"), ("op", "Operating profitability", r"$\Delta$OP$/BE_{q-1}$")]),
          ("Cash flows", [("ocf", "Operating cash flow", r"$\Delta$OCF$/A$"), ("fcf", "Free cash flow", r"$\Delta$FCF$/A$"),
                          ("acc", "Accruals", r"$\Delta$ACC$/A_{q-1}$")]),
          ("Balance sheet and valuation", [("beg", "Book equity", r"$\Delta\log$"), ("inv", "Inventory", r"$\Delta\log$"),
                                           ("bm_f", "Book-to-market", r"$\Delta BE/ME$"), ("ep", "Earnings-to-price", r"$\Delta$IB$/ME$")])]
VARS = [v for _, rows in GROUPS for v, _, _ in rows]


def fundamentals(perm):
    a = pd.read_parquet(f"{R}/data/raw/comp_fundq.parquet")
    a["datadate"] = pd.to_datetime(a.datadate)
    a = a.sort_values(["gvkey", "datadate"]).drop_duplicates(["gvkey", "datadate"], keep="last")
    x = pd.read_parquet(f"{R}/data/raw/comp_fundq_extra.parquet"); x["datadate"] = pd.to_datetime(x.datadate)
    x = x.sort_values(["gvkey", "datadate"]).drop_duplicates(["gvkey", "datadate"], keep="last")
    a = a.merge(x[["gvkey", "datadate", "xsgaq", "piq", "oancfy", "capxy", "invtq", "xintq"]], on=["gvkey", "datadate"], how="left")
    a = a.sort_values(["gvkey", "fyearq", "fqtr"])
    for v in ["oancfy", "capxy"]:                     # quarterly flows from year-to-date items
        g = a.groupby(["gvkey", "fyearq"]); prev = g[v].shift(1); pq = g.fqtr.shift(1)
        a[v[:-1] + "_q"] = np.where(a.fqtr == 1, a[v], np.where(pq == a.fqtr - 1, a[v] - prev, np.nan))
    e = pd.read_parquet(f"{R}/data/derived/earnings.parquet", columns=["gvkey", "datadate", "sue_sd", "sur_sale", "fe_p"])
    e["datadate"] = pd.to_datetime(e.datadate)
    a = a.merge(e.drop_duplicates(["gvkey", "datadate"]), on=["gvkey", "datadate"], how="left")
    lk = pd.read_parquet(f"{R}/data/raw/ccm_link.parquet")
    lk["linkdt"] = pd.to_datetime(lk.linkdt); lk["linkenddt"] = pd.to_datetime(lk.linkenddt).fillna(pd.Timestamp("2099-12-31"))
    a = a.merge(lk[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey")
    a = a[(a.datadate >= a.linkdt) & (a.datadate <= a.linkenddt)].copy(); a["permno"] = a.permno.astype(int)
    a["cq"] = (a.datadate.dt.to_period("M") - 1).dt.asfreq("Q")      # calendar quarter of the fiscal quarter's middle month
    a = a.sort_values(["permno", "datadate"]).drop_duplicates(["permno", "cq"], keep="last").set_index(["permno", "cq"])

    def lag(s, k):
        idx = pd.MultiIndex.from_arrays([s.index.get_level_values(0), s.index.get_level_values(1) - k])
        return pd.Series(s.reindex(idx).values, index=s.index)
    pos = lambda s: s.where(s > 0)
    d4 = lambda s: s - lag(s, 4)
    lg = lambda s: np.log(pos(s)) - np.log(pos(lag(s, 4)))
    at, be, me = a.atq, pos(a.ceqq), pos(a.prccq * a.cshoq)
    F = pd.DataFrame(index=a.index)
    F["eps"] = d4(a.epspxq / a.ajexq) / pos(lag(a.prccq / a.ajexq, 4))
    F["ni"] = d4(a.niq) / pos(lag(at, 4)); F["ib"] = d4(a.ibq) / pos(lag(at, 4)); F["ebt"] = d4(a.piq) / pos(lag(at, 4))
    F["sue"] = a.sue_sd; F["fe"] = a.fe_p * 100
    F["sale"] = lg(a.saleq); F["rsur"] = a.sur_sale; F["cogs"] = lg(a.cogsq); F["sga"] = lg(a.xsgaq)
    F["gm"] = d4((a.saleq - a.cogsq) / pos(a.saleq))
    F["roa"] = d4(a.ibq / pos(lag(at, 1))); F["roe"] = d4(a.ibq / lag(be, 1))
    F["gpa"] = d4((a.saleq - a.cogsq) / pos(lag(at, 1)))
    F["op"] = d4((a.revtq - a.cogsq.fillna(0) - a.xsgaq.fillna(0) - a.xintq.fillna(0))
                 .where(a.cogsq.notna() | a.xsgaq.notna()) / lag(be, 1))
    F["ocf"] = d4(a.oancf_q) / pos(lag(at, 4)); F["fcf"] = d4(a.oancf_q - a.capx_q) / pos(lag(at, 4))
    F["acc"] = d4((a.ibq - a.oancf_q) / pos(lag(at, 1)))
    F["beg"] = lg(a.ceqq); F["inv"] = lg(a.invtq); F["bm_f"] = d4(be / me); F["ep"] = d4(a.ibq / me)
    cs = pd.read_parquet(f"{R}/data/derived/firm_spend.parquet", columns=["permno", "month", "g_yoy"])
    cs = cs[np.isfinite(cs.g_yoy)]; cs["cq"] = cs.month.dt.asfreq("Q")
    cq_g = cs.groupby(["permno", "cq"]).g_yoy.agg(["mean", "count"])
    F["card"] = cq_g[cq_g["count"] == 3]["mean"].reindex(F.index).values
    F = F.replace([np.inf, -np.inf], np.nan).reset_index()
    F = F[F.permno.isin(perm) & (F.cq >= pd.Period("2019Q1", "Q")) & (F.cq <= pd.Period("2026Q4", "Q"))].copy()
    for v in VARS:
        F[v] = ap.winsor(F[v])
    return F


def wavg(O, w, vals, ids):
    O = O.assign(v=O.j.map(vals), w=w).dropna(subset=["v"]); O = O[O.w > 0]
    return ids.map((O.w * O.v).groupby(O.i).sum() / O.w.groupby(O.i).sum())


def loo_mean(own, grp):
    g = own.groupby(grp).agg(["sum", "count"])
    s, c = grp.map(g["sum"]), grp.map(g["count"]); n = c - own.notna().astype(int)
    return (s - own.fillna(0)) / n.where(n > 0)


def build_panel(P, F):
    U = {t: set(d.permno) for t, d in P.groupby("month")}
    rows = []
    for cq in sorted({pd.Period(t, "M").asfreq("Q") for t in U}):
        t = cq.asfreq("M", how="start")
        if t not in U:
            continue
        O = pd.read_parquet(f"{R}/data/derived/ovl/{t}.parquet", columns=["i", "j", "A_S", "E_S", "X_S"])
        O = O[O.i.isin(U[t]) & O.j.isin(U[t]) & (O.A_S > 0)]
        S = pd.read_parquet(f"{R}/data/derived/sim/{t}.parquet", columns=["i", "j", "RPS"])
        S = S[S.i.isin(U[t]) & S.j.isin(U[t]) & (S.i != S.j)]
        d = P[P.month == t][["permno", "cat", "ff48", "size", "bm", "negbe", "mom"]].copy()
        fq = F[F.cq == cq].set_index("permno")
        for v in VARS:
            own = d.permno.map(fq[v])
            d[f"co_{v}"] = wavg(O, O.A_S, fq[v], d.permno).values
            d[f"e_{v}"] = wavg(O, O.E_S.clip(lower=0), fq[v], d.permno).values
            d[f"x_{v}"] = wavg(O, O.X_S.clip(lower=0), fq[v], d.permno).values
            d[f"rps_{v}"] = wavg(S, S.RPS.clip(lower=0), fq[v], d.permno).values
            d[f"cat_{v}"] = loo_mean(own, d.cat).values
            d[f"ind_{v}"] = loo_mean(own, d.ff48).values
        d["cq"] = cq
        rows.append(d)
    A = pd.concat(rows).reset_index(drop=True)
    FI = F.set_index(["permno", "cq"])
    for h, nm in [(-1, "m1"), (0, "0")]:
        idx = pd.MultiIndex.from_arrays([A.permno, A.cq + h])
        for v in VARS:
            A[f"{v}_{nm}"] = FI[v].reindex(idx).values
    A["qid"] = A.cq.astype(str)
    for v in VARS:                                   # as in Table 9: no peer in a component -> quarter's mean
        for c in ["e", "x"]:
            A[f"{c}_{v}"] = A[f"{c}_{v}"].fillna(A.groupby("cq")[f"{c}_{v}"].transform("mean")).where(A[f"co_{v}"].notna())
    return A


def fit(A, y, xs, ctl):
    D = A.dropna(subset=[y] + xs + ctl)
    m = pf.feols(f"{y} ~ {' + '.join(xs + ctl)} | qid", data=D, vcov={"CRV1": "permno + qid"})
    return [[float(m.coef()[x]), float(m.tstat()[x]), int(m._N)] for x in xs]


def main():
    P = M.load("co"); P = P[P.sig.notna()][["permno", "month", "cat", "ff48", "size", "bm", "negbe", "mom"]].copy()
    A = build_panel(P, fundamentals(set(P.permno)))
    res = {}
    for v in VARS:
        ctl = [f"cat_{v}", f"{v}_m1", "size", "mom"] + ([] if v == "bm_f" else ["bm", "negbe"])
        y = f"{v}_0"; r = {}
        r["co"], = fit(A, y, [f"co_{v}"], ctl)
        r["rps"], = fit(A, y, [f"rps_{v}"], ctl)
        r["E"], r["X"] = fit(A, y, [f"e_{v}", f"x_{v}"], ctl)
        r["ind"], = fit(A, y, [f"ind_{v}"], ctl)
        r["corr_E_X"] = float(A[[f"e_{v}", f"x_{v}"]].corr().iloc[0, 1])
        res[v] = r
        print(v, {k: (round(z[0], 3), round(z[1], 2), z[2]) for k, z in r.items() if isinstance(z, list)}, flush=True)
    cols = ["co", "rps", "E", "X", "ind"]
    hdr = [["", "", "", "", "\\multicolumn{2}{c}{Overlap components}", "", ""], "\\cmidrule(lr){5-6}",
           ["", "Measure", "Customer", "Shared", "Local-", "Firm-", "Same", ""],
           ["", "", "overlap", "tastes", "market", "specific", "FF48", "Obs."]]
    body = []
    for g, rows in GROUPS:
        body.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{g}}}}} \\\\")
        for v, lab, meas in rows:
            rb, rt = [f"\\quad {lab}", meas], ["", ""]
            for k in cols:
                b, t = ap.coef_cell(*res[v][k][:2]); rb.append(b); rt.append(t)
            rb.append(f"{res[v]['co'][2]:,}"); rt.append("")
            body += [rb, rt]
    cex = [res[v]["corr_E_X"] for v in VARS]
    n_ind = [res[v]["ind"][2] for v in VARS]
    notes = ("Panel regressions of a firm's fundamental in calendar quarter $q$ (fiscal quarters assigned by their middle month) on "
             "the same fundamental of its peers in $q$, $\\overline{\\mathrm{Fundamental}}^{\\,N}_{iq}=\\sum_j w^{N}_{ijq}\\mathrm{Fundamental}_{jq}/\\sum_j w^{N}_{ijq}$, controlling "
             "for the average of the other firms in its SafeGraph category, its own value in $q-1$, log size, log book-to-market (set to "
             "zero for firms with negative book equity, with an indicator for them; both omitted for the book-to-market fundamental), "
             "and momentum, with quarter fixed effects; each network is a separate regression. Fundamentals are year-over-year changes "
             "from Compustat and IBES ($\\Delta\\log$: log growth; $\\Delta X/A$: change scaled by total assets a year earlier; other "
             "$\\Delta$: change in the ratio; GP: sales minus cost of goods sold; OP: revenue minus cost of goods sold, SG\\&A, and "
             "interest; FCF: operating cash flow minus capital expenditures; ACC: income before extraordinary items minus operating "
             "cash flow); SUE, the revenue surprise, the forecast error, and card-spending growth (SafeGraph, quarterly average) are "
             "as in Table~\\ref{tab:cfnews}. Overlap and shared tastes (positive part of RPS) are formed in the first month of $q$ "
             "from card data for the preceding twelve months; the local-market and firm-specific components ($E$ and the positive "
             "part of $X$) enter jointly, as in Table~\\ref{tab:decomp}, with the quarter's mean for firms without a peer in a "
             "component. Same-FF48 peers: the other sample firms in the firm's FF48 industry, equal-weighted "
             f"({min(n_ind):,}--{max(n_ind):,} observations). Variables are winsorized at 1\\% and 99\\%; standard errors are "
             "clustered by firm and quarter; $t$-statistics in parentheses.")
    tex = ap.latex_table(hdr, body, "Customer overlap and the comovement of fundamentals", "tab:fundcomove", notes,
                         colspec="llcccccr", size="\\scriptsize", stretch=0.8)
    open(f"{R}/tables/tab_fundcomove.tex", "w").write(tex)
    sig = {k: [v for v in VARS if abs(res[v][k][1]) >= 1.96] for k in cols}
    co_t = [res[v]["co"][1] for v in sig["co"]]
    summ = {"results": res, "n_fundamentals": len(VARS), "n_sig_5pct": {k: len(s) for k, s in sig.items()},
            "sig_5pct": sig, "co_sig_t_range": [min(co_t), max(co_t)], "corr_E_X_range": [min(cex), max(cex)],
            "firms": int(A.dropna(subset=["co_sale"]).permno.nunique()),
            "quarters": [str(A.cq.min()), str(A.cq.max())]}
    json.dump(summ, open(f"{R}/logs/fundcomove.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k != "results"}, indent=1, default=float))


if __name__ == "__main__":
    main()
