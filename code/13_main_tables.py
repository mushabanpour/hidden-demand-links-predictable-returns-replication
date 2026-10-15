"""
13_main_tables.py -- main asset-pricing results: portfolio sorts, factor alphas,
cumulative returns, Fama-MacBeth and fixed-effect panel regressions.
"""
import os
import sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
import pyfixest as pf

R = ap.R
SIG = sys.argv[1] if len(sys.argv) > 1 else "co"
TAB, FIG = f"{R}/tables", f"{R}/figures"
F = ap.factors()
Fn = F.copy(); Fn.index = Fn.index - 1          # month t+1 factors aligned with formation month t
CONTROLS = ["ret", "ind_u", "indret", "cat_ew", "cat_vw", "STORE_pos", "tnic_ret", "size", "bm", "gpa", "ik", "mom"]
LABEL = {"sig": "Customer-overlap peer return", "sig_l1": "Customer-overlap peer return, $t-1$",
         "cat_ew": "Same-category return (EW)", "cat_vw": "Same-category return (VW)", "RPS_pos": "Shared-taste (RPS) peer return", "ret": "Own return",
         "STORE_pos": "Store-footprint peer return", "tnic_ret": "TNIC-3 peer return",
         "indret": "Industry return (FF48, all stocks)", "ind_u": "Own-industry peers' return", "hq_ret": "HQ-city peer return", "cov_ret": "Shared-analyst peer return", "cust_ret": "Customer return",
         "size": "Log market cap", "bm": "Log book-to-market", "gpa": "Gross profitability", "ik": "Investment (CAPX/PPE)",
         "mom": "Momentum $r_{t-11,t-1}$", "ret_l1": "Own return, $t-1$", "HOME_pos": "Customer-home peer return"}


def load(sig=SIG, src="panel_final"):
    P = pd.read_parquet(f"{R}/data/derived/{src}.parquet")
    P = P.drop(columns=[c for c in P.columns if c in ("sig", "sig_l1", "sig_adj", "ind_u", "xind_u", "exret_f1")
                        or (c.endswith("_cross_x"))])
    P["sig"] = P[sig]
    # within-universe own-industry return (EW, leave-one-out) and the naive cross-industry average
    g = P.groupby(["month", "ff48"]).ret
    n_i = g.transform("count"); s_i = g.transform("sum")
    P["ind_u"] = ((s_i - P.ret) / (n_i - 1)).where(n_i >= 2)
    s_a = P.groupby("month").ret.transform("sum"); n_a = P.groupby("month").ret.transform("count")
    P["xind_u"] = (s_a - s_i) / (n_a - n_i)
    for c in [c for c in P.columns if c.endswith("_cross")]:
        P[c + "_x"] = P[c] - P.xind_u            # RPS-weighted cross-industry return in excess of the naive average
    P = P.sort_values(["permno", "month"])
    lag = P[["permno", "month", "sig"]].copy(); lag["month"] = lag.month + 1
    P = P.merge(lag.rename(columns={"sig": "sig_l1"}), on=["permno", "month"], how="left")
    P["ik"] = P.ik.clip(upper=P.ik.quantile(0.99))
    for c in ["sig", "sig_l1", "ret", "ret_l1", "STORE_pos", "HOME_pos", "RPS_pos", "tnic_ret", "indret", "ind_u", "hq_ret", "cov_ret", "cust_ret",
              "size", "bm", "gpa", "ik", "mom", "ag"]:
        P[c] = P.groupby("month")[c].transform(lambda s: ap.winsor(s.replace([np.inf, -np.inf], np.nan)))
    # controls that exist only for linked firms: missing -> monthly cross-sectional mean (no link = no news)
    for c in [c for c in ["tnic_ret", "hq_ret", "cov_ret", "cust_ret", "cat_ew", "cat_vw"] if c in P]:
        P[c] = P[c].fillna(P.groupby("month")[c].transform("mean"))
    if "cat_ew" not in P:                       # base panel (before the category returns are built)
        return P
    P["exret_f1"] = P.ret_f1 - P.month.map(Fn.rf)
    # industry-adjusted signal: monthly cross-sectional residual of the peer return on industry returns
    def resid(d):
        x = d[["ind_u", "indret", "cat_ew", "cat_vw"]].fillna(d[["ind_u", "indret", "cat_ew", "cat_vw"]].mean())
        X = np.column_stack([np.ones(len(d)), x.values]); y = d.sig.values; ok = ~np.isnan(y)
        b = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        return pd.Series(y - X @ b, index=d.index)
    P["sig_adj"] = P.groupby("month", group_keys=False).apply(resid)
    return P


def sorts_table(P):
    """every portfolio of the quintile, decile, and tercile sorts"""
    res, out = {}, {}
    for nq in (3, 5, 10):
        ew, vw, n = ap.sort_returns(P, "sig", nq)
        res[(nq, "ew")], res[(nq, "vw")] = ew, vw
    rows = []
    for k, (nq, nm) in enumerate([(5, "Quintiles"), (10, "Deciles"), (3, "Terciles")]):
        rows.append(f"\\multicolumn{{12}}{{l}}{{\\emph{{Panel {'ABC'[k]}: {nm} ($Q={nq}$)}}}} \\\\")
        for w, wl in (("ew", "Equal-weighted"), ("vw", "Value-weighted")):
            x = res[(nq, w)]
            rb, rt = [wl], [""]
            for q in range(1, nq + 1):
                m, t, _ = ap.nw_mean(x[q])
                rb.append(f"{m*100:.2f}"); rt.append(f"({t:.2f})"); out[f"{w}_{nq}_{q}"] = (m, t)
            rb += [""] * (10 - nq); rt += [""] * (10 - nq)
            m, t, _ = ap.nw_mean(x.LS)
            rb.append(f"{m*100:.2f}{ap.stars(t)}"); rt.append(f"({t:.2f})"); out[f"{w}_{nq}_LS"] = (m, t)
            rows += [rb, rt]
        if k < 2:
            rows.append("\\addlinespace")
    T = len(res[(5, "ew")])
    notes = (f"At the end of each month $t$, consumer-facing stocks are sorted into $Q$ portfolios on the month-$t$ "
             "return of their customer-overlap peers (other sample firms weighted by the share of customers they have in common "
             "with the firm); the network uses card data from months $t-12$ to $t-1$ only. Portfolio 1 holds the firms whose peers "
             "had the lowest returns and portfolio $Q$ those whose peers had the highest. The table reports average month-$t+1$ "
             "returns in percent per month; High$-$Low is portfolio $Q$ minus portfolio 1. Value weights use month-$t$ market "
             "capitalization. Newey-West $t$-statistics (four lags) in parentheses. "
             f"Sample: {P.dropna(subset=['ret_f1']).month.min()+1} to {P.dropna(subset=['ret_f1']).month.max()+1}, {T} months.")
    tex = ap.latex_table([["Portfolio", *[str(q) for q in range(1, 11)], "High$-$Low"]], rows,
                         "Portfolios sorted on customer-overlap peers' returns", "tab:sorts", notes,
                         colspec="l" + "c" * 11)
    open(f"{TAB}/tab_sorts.tex", "w").write(tex)
    return res, out


def alpha_table(res, P):
    models = ["Raw", "CAPM", "FF3", "FFC4", "FF5", "FF6", "HXZ4", "HXZ5"]
    rows, out = [], {}
    ewa, vwa, _ = ap.sort_returns(P, "sig_adj", 5)
    res[("adj", "ew")], res[("adj", "vw")] = ewa, vwa
    rows.append("\\multicolumn{9}{l}{\\emph{Panel A: Sorts on the peer return}} \\\\")
    for key, w, wl in ((5, "ew", "EW"), (5, "vw", "VW"), ("sep", None, None),
                       ("adj", "ew", "EW"), ("adj", "vw", "VW")):
        if key == "sep":
            rows.append("\\addlinespace")
            rows.append("\\multicolumn{9}{l}{\\emph{Panel B: Sorts on the industry-adjusted peer return}} \\\\")
            continue
        ls = res[(key, w)].LS
        if key == "adj":
            w = w + "_adj"
        r_a, r_t = [f"$\\alpha$ ({wl})"], [""]
        for mdl in models:
            a = ap.ts_reg(ls, Fn, mdl)
            r_a.append(f"{a['alpha']*100:.2f}{ap.stars(a['t_alpha'])}"); r_t.append(f"({a['t_alpha']:.2f})")
            out[f"{w}_{mdl}"] = a
        rows += [r_a, r_t]
    # loadings for the equal-weighted FF6 and HXZ5 regressions
    rows.append("\\addlinespace")
    rows.append("\\multicolumn{9}{l}{\\emph{Panel C: Factor loadings, equal-weighted long-short portfolio of Panel A}} \\\\")
    fac = [("mktrf", "MKT"), ("smb", "SMB"), ("hml", "HML"), ("umd", "MOM"), ("smb5", "SMB$_5$"), ("hml5", "HML$_5$"),
           ("rmw", "RMW"), ("cma", "CMA"), ("q_mkt", "MKT$_q$"), ("q_me", "ME"), ("q_ia", "IA"), ("q_roe", "ROE"), ("q_eg", "EG")]
    for f, fl in fac:
        r_b, r_t = [fl], [""]
        for mdl in models:
            a = out[f"ew_{mdl}"]
            if f in a:
                r_b.append(f"{a[f]:.2f}{ap.stars(a['t_'+f])}"); r_t.append(f"({a['t_'+f]:.2f})")
            else:
                r_b.append(""); r_t.append("")
        if any(c for c in r_b[1:]):
            rows += [r_b, r_t]
    r2 = ["$R^2$"] + [f"{out['ew_'+m]['r2']:.2f}" for m in models]
    rows.append(r2)
    rows.append(["Months"] + [str(out['ew_' + m]['n']) for m in models])
    notes = ("Time-series regressions of the monthly return of the quintile long-short portfolio (top minus bottom quintile "
             "of the customer-overlap peer return; Table~\\ref{tab:sorts}) on factor models: CAPM; Fama-French three-factor "
             "(FF3); FF3 plus momentum (FFC4); Fama-French five-factor (FF5); FF5 plus momentum (FF6); and the $q$-factor "
             "(HXZ4) and $q^5$ (HXZ5) models of \\citet{HouXueZhang2015} and \\citet{HouMoXueZhang2021}. Alphas in percent "
             "per month; Raw is the average long-short return without factor adjustment. Panel B sorts into quintiles on the peer "
             "return after removing, month by month, its cross-sectional projection on the own-industry, same-category (EW and VW), "
             "and FF48 industry returns. Panel C reports the factor loadings of the equal-weighted regressions in Panel A: MKT, SMB, "
             "HML, and MOM are the market, size, value, and momentum factors of the three-factor and Carhart models; SMB$_5$ and "
             "HML$_5$ are the size and value factors of the five-factor model, with RMW (profitability) and CMA (investment); "
             "MKT$_q$, ME, IA, ROE, and EG are the market, size, investment, profitability, and expected-growth factors of the "
             "$q$ and $q^5$ models. $R^2$ and Months refer to the equal-weighted regressions in Panel A. EW and VW denote equal and "
             "value weights. Newey-West $t$-statistics (four lags) in parentheses.")
    tex = ap.latex_table([["", *models]], rows, "Long-short portfolio alphas across factor models", "tab:alphas", notes)
    open(f"{TAB}/tab_alphas.tex", "w").write(tex)
    return out


def figures(P, res):
    import figstyle as fs
    fs.apply()
    ew = res[(5, "ew")]; vw = res[(5, "vw")]
    idx = (ew.index + 1).to_timestamp()
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for k, (s_, lab) in enumerate([(ew[5], "High (Q5)"), (ew[1], "Low (Q1)"), (ew.LS, "High$-$Low")]):
        fs.line(ax[0], idx, (1 + s_.values).cumprod(), k, label=lab, markevery=6)
    mk = Fn.loc[ew.index]
    for k, (s_, lab) in enumerate([(ew.LS, "Long-short, equal-weighted"), (vw.LS, "Long-short, value-weighted"),
                                   (mk.mktrf, "Market excess return")]):
        fs.line(ax[1], idx, (1 + s_.values).cumprod(), k, label=lab, markevery=6)
    for a in ax:
        fs.zero_line(a, 1.0)
        a.set_xlabel("Month")
        a.legend(loc="upper left")
    ax[0].set_ylabel("Value of \\$1")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_cumret.pdf"); plt.close()

    # event-time figure is produced by 30_eventtime_cf.py (Cohen-Frazzini style, observed returns only)
    return {}


def fm_table(P):
    base = ["ret", "ind_u", "indret", "cat_ew", "cat_vw"]
    chars = ["size", "bm", "gpa", "ik", "mom"]
    specs = [["sig"], ["sig", "ret"], ["sig"] + base,
             ["sig"] + base + ["STORE_pos", "tnic_ret"],
             ["sig"] + base + ["STORE_pos", "tnic_ret"] + chars,
             ["sig", "sig_l1"] + base + ["ret_l1", "STORE_pos", "tnic_ret"] + chars,
             ["sig"] + base + ["STORE_pos", "tnic_ret", "hq_ret", "cov_ret", "cust_ret", "RPS_pos", "HOME_pos"] + chars]
    allv = ["sig", "sig_l1", "ret", "ret_l1", "ind_u", "indret", "cat_ew", "cat_vw", "STORE_pos", "tnic_ret", "hq_ret",
            "cov_ret", "cust_ret", "RPS_pos", "HOME_pos"] + chars
    fm = []
    for s in specs:
        r, r2, n, T = ap.fama_macbeth(P, "ret_f1", s)
        fm.append((r, r2, n, T))
    # pooled panel with fixed effects, two-way clustered
    D = P.dropna(subset=["ret_f1"] + specs[4]).copy()
    D["mid"] = D.month.astype(str)
    co = pd.read_parquet(f"{R}/data/raw/comp_company.parquet", columns=["gvkey", "state"])
    D = D.merge(co, on="gvkey", how="left"); D["state"] = D.state.fillna("NA")
    fe = []
    for fe_spec in ["mid", "mid + permno", "mid + ff48", "mid + state"]:
        m = pf.feols(f"ret_f1 ~ {' + '.join(specs[4])} | {fe_spec}", data=D, vcov={"CRV1": "permno + mid"})
        fe.append(m)
    hdr = [["", *[f"({k})" for k in range(1, len(specs) + 1)], "(8)", "(9)", "(10)", "(11)"]]
    rows = []
    for v in allv:
        rb, rt = [LABEL.get(v, v)], [""]
        for (r, *_rest), s in zip(fm, specs):
            if v in s:
                b, t = r[v]; rb.append(f"{b:.3f}{ap.stars(t)}"); rt.append(f"({t:.2f})")
            else:
                rb.append(""); rt.append("")
        for m in fe:
            if v in specs[4]:
                b, t = m.coef()[v], m.tstat()[v]; rb.append(f"{b:.3f}{ap.stars(t)}"); rt.append(f"({t:.2f})")
            else:
                rb.append(""); rt.append("")
        rows += [rb, rt]
    rows.append("\\addlinespace")
    rows.append(["Estimator"] + ["FM"] * len(specs) + ["Panel"] * 4)
    rows.append(["Month FE"] + [""] * len(specs) + ["Yes", "Yes", "Yes", "Yes"])
    rows.append(["Firm FE"] + [""] * len(specs) + ["", "Yes", "", ""])
    rows.append(["Industry FE"] + [""] * len(specs) + ["", "", "Yes", ""])
    rows.append(["HQ-state FE"] + [""] * len(specs) + ["", "", "", "Yes"])
    rows.append(["Avg./within $R^2$"] + [f"{x[1]:.3f}" for x in fm] + [f"{m._r2_within:.3f}" for m in fe])
    rows.append(["Observations"] + [f"{x[2]:,}" for x in fm] + [f"{m._N:,}" for m in fe])
    notes = ("The dependent variable is the stock's month-$t+1$ return. Explanatory variables are measured at the end of month "
             "$t$. The customer-overlap peer return is the month-$t$ return of other sample firms weighted by the share of customers "
             "they have in common with the firm (equation~\\eqref{eq:peer}; card data from $t-12$ to $t-1$). Own-industry peers' and "
             "same-category returns are leave-one-out averages over sample firms in the firm's FF48 industry and in its SafeGraph "
             "category (the most frequent category of its stores over months $t-12$ to $t-1$), equal- or value-weighted. "
             "The shared-taste return weights peers by RPS (Section~\\ref{sec:rps}) and the customer-home return by \\textsc{Home}; "
             "the store-footprint return weights peers by \\textsc{Store}; the TNIC-3 return is the equal-weighted return of "
             "TNIC-3 rivals; the industry return is the value-weighted FF48 return of all CRSP stocks excluding the firm; the "
             "customer return follows \\citet{CohenFrazzini2008}; the HQ-city and shared-analyst returns follow \\citet{ParsonsSabbatucciTitman2020} and \\citet{AliHirshleifer2020}. "
             "When a firm has no link in a network (TNIC-3, HQ city, shared analysts, customers, same category), that peer return "
             "is set to its monthly cross-sectional mean. Columns (1)--(7): "
             "Fama-MacBeth regressions with Newey-West (four lags) $t$-statistics. Columns (8)--(11): pooled OLS with fixed "
             "effects and standard errors clustered by firm and month. All regressors are winsorized at 1\\% and 99\\% each month.")
    tex = ap.latex_table(hdr, rows, "Customer-overlap peer returns and future returns: regressions", "tab:fm", notes,
                         size="\\scriptsize", side=True, stretch=0.82)
    open(f"{TAB}/tab_fm.tex", "w").write(tex)
    return fm, fe


if __name__ == "__main__":
    P = load()
    res, srt = sorts_table(P)
    al = alpha_table(res, P)
    ev = figures(P, res)
    fm, fe = fm_table(P)
    summ = {"sorts": {k: list(v) for k, v in srt.items()},
            "alphas": {k: {kk: vv for kk, vv in v.items()} for k, v in al.items()}, **ev,
            "fm": [{k: v for k, v in x[0].items()} for x in fm]}
    json.dump(summ, open(f"{R}/logs/main_results_{SIG}.json", "w"), indent=1, default=float)
    print(json.dumps({k: summ["alphas"][k]["alpha"] for k in summ["alphas"]}, indent=0, default=float))
    print("FM:", [(round(x[0]["sig"][0], 3), round(x[0]["sig"][1], 2)) for x in fm])
