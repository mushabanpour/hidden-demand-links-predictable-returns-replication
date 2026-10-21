"""
55_card_power.py -- do the card data measure the firm's sales news? A first stage for the cash-flow tests (Table 10).

For each fiscal quarter of a sample firm, card growth is the average over the quarter's three months of the
year-over-year log growth of the firm's share of panel card spending (the measure of Table 10, Panel A). It is regressed
on the quarter's year-over-year sales growth (Compustat) and used to predict the announcement outcomes of Table 10,
Panel B: the revenue surprise, SUE, the analyst forecast error, and CAR[-1,+1], for the same quarter and for the next
two quarters (h = 1, 2: card growth of quarter q predicting the surprise announced for quarter q+h).
Card growth is standardized; quarter fixed effects; standard errors clustered by firm and quarter.
Output: logs/card_power.json, tables/tab_cardpower.tex
"""
import os, sys, json
import numpy as np, pandas as pd, pyfixest as pf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
R = ap.R


def main():
    P = pd.read_parquet(f"{R}/data/derived/panel_final.parquet", columns=["permno", "month"])
    fs = pd.read_parquet(f"{R}/data/derived/firm_spend.parquet", columns=["permno", "month", "g_yoy"])
    fs = fs[fs.set_index(["permno", "month"]).index.isin(P.set_index(["permno", "month"]).index)]   # sample firm-months
    q = pd.read_parquet(f"{R}/data/derived/earnings.parquet")
    q["qend"] = pd.to_datetime(q.datadate).dt.to_period("M")
    q = q.sort_values(["permno", "qend"]).drop_duplicates(["permno", "qend"], keep="last")
    g = []
    for k in range(3):
        x = fs.copy(); x["qend"] = x.month + k; g.append(x)
    g = pd.concat(g).groupby(["permno", "qend"]).g_yoy.agg(["mean", "count"]).reset_index()
    g = g[g["count"] == 3].rename(columns={"mean": "card_g"})
    D = q.merge(g[["permno", "qend", "card_g"]], on=["permno", "qend"], how="inner")
    D["card_g"] = ap.winsor(D.card_g)
    D["card_z"] = (D.card_g - D.card_g.mean()) / D.card_g.std()
    for c in ["sue_sd", "sur_sale", "sale_g", "fe_p", "car_m1p1"]:
        D[c] = ap.winsor(D[c].replace([np.inf, -np.inf], np.nan))
    D["fe_p"] = D.fe_p * 100; D["car_m1p1"] = D.car_m1p1 * 100; D["sale_g"] = D.sale_g * 100
    D["qtr"] = D.qend.astype(str)
    D = D.sort_values(["permno", "qend"])
    res = {"n_firm_quarters": len(D), "n_firms": int(D.permno.nunique()),
           "corr_card_sale_g": float(D[["card_g", "sale_g"]].corr().iloc[0, 1])}
    for h in (0, 1, 2):
        lead = D.groupby("permno")[["sue_sd", "sur_sale", "fe_p", "car_m1p1", "sale_g", "qend"]].shift(-h)
        ok_gap = (lead.qend - D.qend).apply(lambda x: x.n if pd.notna(x) else np.nan) == 3 * h if h else True
        E = D[["permno", "qtr", "card_z"]].join(lead.drop(columns="qend"))
        E = E[ok_gap] if h else E
        for y in ["sale_g", "sur_sale", "sue_sd", "fe_p", "car_m1p1"]:
            d = E.dropna(subset=[y, "card_z"])
            m = pf.feols(f"{y} ~ card_z | qtr", data=d, vcov={"CRV1": "permno + qtr"})
            res[f"h{h}_{y}"] = {"b": float(m.coef()["card_z"]), "t": float(m.tstat()["card_z"]), "N": int(m._N),
                                "sd_y": float(d[y].std())}
            print(f"h={h} {y:9s} b={m.coef()['card_z']:.3f} t={m.tstat()['card_z']:.2f} N={m._N} (sd of y {d[y].std():.3f})")
    res["bounds"] = bounds()
    json.dump(res, open(f"{R}/logs/card_power.json", "w"), indent=1)
    print({k: v for k, v in res.items() if not isinstance(v, dict)}, res["bounds"])
    table(res)


def bounds():
    """upper end of the 95% confidence interval of the peer-return coefficient in Table 10, in standard deviations of the
    outcome, next to the effect of a one-standard-deviation increase in own card growth (quarter q)"""
    from importlib import import_module
    M = import_module("13_main_tables"); C = import_module("15_cashflow_news")
    P = C.build(M.load("co")).rename(columns={"g_-1": "g_lag"})
    ctl = "ret + ind_u + indret + cat_ew + cat_vw + size + bm + mom"
    out = {}
    for y in ["sur_sale", "sue_sd", "g_fwd"]:
        rhs = f"sig_sd + g_lag + {ctl}" if y.startswith("g") else f"sig_sd + {ctl}"
        D = P.dropna(subset=[y] + (["g_lag"] if y.startswith("g") else []))
        m = pf.feols(f"{y} ~ {rhs} | mid", data=D, vcov={"CRV1": "permno + mid"})
        b, se, sd = m.coef()["sig_sd"], m.se()["sig_sd"], D[y].std()
        out[y] = {"b": float(b), "se": float(se), "sd_y": float(sd), "upper_sd": float((b + 1.96 * se) / sd)}
    return out


def table(res):
    cols = [(0, "sale_g"), (0, "sur_sale"), (0, "sue_sd"), (1, "sur_sale"), (1, "sue_sd"), (2, "sur_sale"), (2, "sue_sd")]
    hdr = [["", "\\multicolumn{3}{c}{Quarter $q$}", "\\multicolumn{2}{c}{Quarter $q+1$}", "\\multicolumn{2}{c}{Quarter $q+2$}"],
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
           ["", "Sales growth", "Rev.\\ surprise", "SUE", "Rev.\\ surprise", "SUE", "Rev.\\ surprise", "SUE"]]
    g = [res[f"h{h}_{y}"] for h, y in cols]
    rows = [["Card-spending growth in $q$ (std.)"] + [f"{x['b']:.3f}{ap.stars(x['t'])}" for x in g],
            [""] + [f"({x['t']:.2f})" for x in g], "\\addlinespace",
            ["Std.\\ dev.\\ of dep.\\ var."] + [f"{x['sd_y']:.2f}" for x in g],
            ["Quarter FE"] + ["Yes"] * len(g), ["Firm-quarters"] + [f"{x['N']:,}" for x in g]]
    notes = ("Panel regressions of the outcomes of a sample firm's earnings announcements on its own card-spending growth. "
             "Card-spending growth is the average, over the three months of fiscal quarter $q$, of the year-over-year log growth of "
             "the firm's share of total panel card spending (the measure of Table~\\ref{tab:cfnews}), standardized. Sales growth is "
             "the year-over-year growth of quarterly sales in percent (Compustat); the revenue surprise and SUE are those of "
             "Table~\\ref{tab:cfnews}, for the announcement of quarter $q$, $q+1$, or $q+2$. Firm-quarters of sample firms with card "
             f"data in all three months of the quarter ({res['n_firms']} firms). Standard errors clustered by firm and quarter; "
             "$t$-statistics in parentheses.")
    open(f"{R}/tables/tab_cardpower.tex", "w").write(
        ap.latex_table(hdr, rows, "Card spending and the firm's own sales and earnings news", "tab:cardpower", notes,
                       size="\\scriptsize"))


if __name__ == "__main__":
    main()
